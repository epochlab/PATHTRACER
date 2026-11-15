#include "smith_walk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

#include "microfacet.h"
#include "pathtracer/scene/fresnel_dielectric.h"
#include "shading_math.h"

namespace pathtracer::scene {

namespace {

// A walk along ray (slab frame) on one side, its height as log C1, C1 = (h+1)/2 (Heitz 2016 sec. 4.1), exact where h rounds to +-1.
struct Walker {
    glm::vec3 ray;
    float logCdf;
    bool upper;  // wo's medium; the far side's walk is the near side's under the point reflection through the mean plane
    glm::vec3 weight;
};

// A side's own frame: GGX and the uniform heights are centrally symmetric, so the far side is the near side point-reflected.
glm::vec3 own(const glm::vec3& v, bool upper) { return upper ? v : -v; }

// Smith Lambda for any w.z != 0 (Heitz 2016 eq. 6): negative below the horizon, where the walk descends.
float lambdaOf(const glm::vec3& w, const glm::vec2& alpha) { return 0.5F * ((smithRadical(w, alpha) / w.z) - 1.0F); }

// The far side's log C1 under the point reflection h -> -h: log(1 - e^x) without cancellation at either end (Maechler 2012).
float farSide(float logCdf) {
    return logCdf > -std::numbers::ln2_v<float> ? std::log(-std::expm1(logCdf)) : std::log1p(-std::exp(logCdf));
}

// The bottom C1 = 0, reached only by refracting at the very top (a level ray's collision): a zero-measure state no walk occupies.
bool bottom(float logCdf) { return !(logCdf > -std::numeric_limits<float>::infinity()); }

// The odds a ray along w (own frame) leaves: G1 = C1^Lambda rising (Heitz 2016 eq. 9), zero level, falling or from the bottom.
float escapeOdds(const glm::vec3& w, float logCdf, const glm::vec2& alpha) {
    return w.z > 0.0F && !bottom(logCdf) ? std::exp(lambdaOf(w, alpha) * logCdf) : 0.0F;
}

// Moves the walker to its next collision, u uniform on [0, 1); false where u falls in the escaping share G1. A level ray keeps its height.
bool collides(const glm::vec2& alpha, Walker& walker, float u) {
    const glm::vec3 d = own(walker.ray, walker.upper);
    if (u >= 1.0F - escapeOdds(d, walker.logCdf, alpha)) {
        return false;
    }
    // C1' = C1 (1 - u)^(-1/Lambda) in logs, where the power alone overflows as Lambda -> 0.
    if (d.z != 0.0F) {
        walker.logCdf = std::min(walker.logCdf - (std::log1p(-u) / lambdaOf(d, alpha)), 0.0F);
    }
    return true;
}

// The bend at a facet seen from a side, n_here / n_there.
float bendRatio(const DielectricSlab& slab, bool upper) { return upper ? slab.etaI / slab.etaT : slab.etaT / slab.etaI; }

// One collision's scattering from toward t to o, both in the side's own frame: its value per channel and its sampling density.
struct Phase {
    glm::vec3 value{0.0F};
    float pdf = 0.0F;
};

// The facet taking t to o, its visible-normal density times the branch's Jacobian (Walter 2007 eqs. 14, 17) and the Fresnel branch odds.
Phase phase(const DielectricSlab& slab, const glm::vec3& t, const glm::vec3& o, bool upper, bool refracted) {
    const float eta = bendRatio(slab, upper);
    glm::vec3 m = refracted ? (eta * t) + o : t + o;
    const float length = glm::length(m);
    if (!(length > 0.0F)) {
        return {};
    }
    m = (glm::dot(t, m) < 0.0F ? -1.0F : 1.0F) * m / length;
    const float tm = glm::dot(t, m);
    const float om = glm::dot(o, m);
    // sigma(t) = (t.z + radical)/2 is the facets' projected area toward t over the whole sphere, zero only seen from straight below.
    const float projected = 0.5F * (t.z + smithRadical(t, slab.alpha));
    if (!(m.z > 0.0F && tm > 0.0F && projected > 0.0F) || (refracted ? om >= 0.0F : om <= 0.0F)) {
        return {};
    }
    const float denom = (eta * tm) + om;
    const float density = (tm * distributionGGX(m, slab.alpha) / projected) * (refracted ? -om / (denom * denom) : 0.25F / tm);
    const glm::vec3 fresnel = interfaceFresnel(slab, tm, !upper);
    const float reflectOdds = channelMean(fresnel);
    return refracted ? Phase{density * (1.0F - fresnel), density * (1.0F - reflectOdds)} : Phase{density * fresnel, density * reflectOdds};
}

// How a collision picks reflection given its facet's Fresnel; the throughput carries the channels' ratio to it.
using ReflectOdds = float (*)(const DielectricSlab&, const glm::vec3& fresnel);

// The facet's own odds, its mean Fresnel: what every collision past the first draws by, and what phase() densities assume.
float meanFresnel(const DielectricSlab& /*slab*/, const glm::vec3& fresnel) { return channelMean(fresnel); }

// First-collision odds: glass by tinted Fresnel; where only wo's side counts (opaque, coat) a/(a+b), a = F, b = (1-F) sqrt(rho).
float firstReflectOdds(const DielectricSlab& slab, const glm::vec3& fresnel) {
    if (slab.refractWeight > 0.0F) {
        return facetReflectProbability(slab, fresnel);
    }
    // rho, a refraction's odds of returning as 2+ reflection, is at most multiReflect over the light not singly reflected.
    const float through = 1.0F - channelMean(slab.reflectSingle);
    const float rho = through > 0.0F ? slab.multiReflect / through : 0.0F;
    // a^2/q + b^2/(1 - q), each branch's second moment over its odds, is least at q = a/(a + b).
    const float reflect = channelMean(slab.tint * fresnel);
    const float returns = slab.tintMean * channelMean(1.0F - fresnel) * std::sqrt(rho);
    return reflect + returns > 0.0F ? reflect / (reflect + returns) : 1.0F;
}

// A visible facet's draw, reflected or refracted by odds, the throughput by its channels; false where it carries no measure.
bool scatter(const DielectricSlab& slab, Walker& walker, Sampler& walk, bool& refracted, ReflectOdds odds = meanFresnel) {
    const glm::vec3 t = own(-walker.ray, walker.upper);
    const glm::vec3 m = sampleVisibleNormal(t, slab.alpha, walk.next2D());
    const float tm = glm::dot(t, m);
    // Round-off near the cap's pole can leave the facets' support (or give NaN), where phase() has zero density: such draws end.
    if (!(tm > 0.0F && m.z > 0.0F)) {
        return false;
    }
    const glm::vec3 fresnel = interfaceFresnel(slab, tm, !walker.upper);
    const float reflectOdds = odds(slab, fresnel);
    refracted = walk.next1D() >= reflectOdds;
    if (!refracted) {
        walker.ray = own((2.0F * tm * m) - t, walker.upper);
        walker.weight *= fresnel / reflectOdds;
        return true;
    }
    // Only a coat-shifted Fresnel ratio below the bend's lets the facet transmit past the bend's critical angle: no direction carries it.
    const float eta = bendRatio(slab, walker.upper);
    const float cos2T = cos2Transmitted(tm, eta);
    if (cos2T < 0.0F) {
        return false;
    }
    walker.ray = own(glm::normalize((((eta * tm) - std::sqrt(cos2T)) * m) - (eta * t)), walker.upper);
    walker.weight *= (1.0F - fresnel) / (1.0F - reflectOdds);
    walker.upper = !walker.upper;
    walker.logCdf = farSide(walker.logCdf);
    return !bottom(walker.logCdf);
}

// Past order 1 only: the walker collides again, weighed by those odds 1 - G1, at a height drawn from the colliding rays alone.
bool forceCollision(const DielectricSlab& slab, Walker& walker, float u) {
    const float collision = 1.0F - escapeOdds(own(walker.ray, walker.upper), walker.logCdf, slab.alpha);
    if (!(collision > 0.0F)) {
        return false;
    }
    walker.weight *= collision;
    static_cast<void>(collides(slab.alpha, walker, u * collision));
    return true;
}

// A walk entering from direction from at the top of its side, to its first collision: a falling ray never escapes.
Walker enter(const DielectricSlab& slab, const glm::vec3& from, Sampler& walk) {
    Walker walker{-from, 0.0F, from.z >= 0.0F, glm::vec3(1.0F)};
    static_cast<void>(collides(slab.alpha, walker, walk.next1D()));
    return walker;
}

// One walk's connections toward `to` from orders >= 2, each weighted P_this / (P_this + P_other) where weighed (Wang et al. 2022 eq. 15).
glm::vec3 connections(const DielectricSlab& slab, const glm::vec3& from, const glm::vec3& to, Sampler& walk, bool weighed) {
    const bool toUpper = to.z >= 0.0F;
    Walker walker = enter(slab, from, walk);
    glm::vec3 sum(0.0F);
    // The other end's density of the interior directions over this walk's, in double: products of peaked densities overrun float.
    double ratio = 1.0;
    for (int order = 1;; ++order) {
        const glm::vec3 toward = -walker.ray;
        if (order > 1) {
            const bool refracted = toUpper != walker.upper;
            const Phase exit = phase(slab, own(toward, walker.upper), own(to, walker.upper), walker.upper, refracted);
            if (exit.pdf > 0.0F) {
                const float leaves = escapeOdds(own(to, toUpper), refracted ? farSide(walker.logCdf) : walker.logCdf, slab.alpha);
                const double other = weighed ? ratio * phase(slab, own(to, toUpper), own(toward, toUpper), toUpper, refracted).pdf : 0.0;
                sum += walker.weight * exit.value * static_cast<float>(leaves / (1.0 + other));
            }
        }
        const bool before = walker.upper;
        bool refracted = false;
        if (!scatter(slab, walker, walk, refracted)) {
            return sum;
        }
        const glm::vec3 out = walker.ray;
        const float drawn = phase(slab, own(toward, before), own(out, before), before, refracted).pdf;
        if (!(drawn > 0.0F)) {
            return sum;
        }
        // The other end meets vertex 1 by connecting, not drawing, so its density enters from vertex 2 (Wang et al. 2022 eq. 14).
        if (order > 1) {
            ratio *= phase(slab, own(out, walker.upper), own(toward, walker.upper), walker.upper, refracted).pdf;
        }
        ratio /= drawn;
        // Order 1's escape adds nothing here, so forcing past it spares the 1/(1 - G1) second moment of a near-specular ray's rare return.
        if (!(order == 1 ? forceCollision(slab, walker, walk.next1D()) : collides(slab.alpha, walker, walk.next1D()))) {
            return sum;
        }
    }
}

}  // namespace

std::optional<WalkExit> sampleInterfaceWalk(const DielectricSlab& slab, const glm::vec3& wo, Sampler& walk) {
    Walker walker = enter(slab, wo, walk);
    bool refracted = false;
    // The first collision draws its branch by what each side's exits carry (Walter 2007 5.3), returning refraction included.
    if (!scatter(slab, walker, walk, refracted, firstReflectOdds)) {
        return std::nullopt;
    }
    while (collides(slab.alpha, walker, walk.next1D())) {
        if (!scatter(slab, walker, walk, refracted)) {
            return std::nullopt;
        }
    }
    // Radiance crossing into the far medium is compressed by (etaI/etaT)^2 (Veach 1997 sec. 5.2), as single scattering's is.
    return WalkExit{walker.ray, walker.upper ? walker.weight : walker.weight * slab.etaSq, walker.upper};
}

glm::vec3 evaluateInterfaceMulti(const DielectricSlab& slab, const glm::vec3& wo, const glm::vec3& wi, Sampler& walk) {
    // Exit densities are f |cos| up to eta^2 across: from wo the radiance compression, from wi f(wo, wi) eta_wi^2 = f(wi, wo) eta_wo^2.
    const bool twoEnded = wo.z > 0.0F;
    const glm::vec3 forward = connections(slab, wo, wi, walk, twoEnded) * (wi.z < 0.0F ? slab.etaSq : 1.0F);
    // A level wo is met only at the top, where no walk from wi can leave toward it: the walk from wo carries the whole value alone.
    if (!twoEnded) {
        return forward;
    }
    return forward + (connections(slab, wi, wo, walk, true) * (std::abs(wi.z) / wo.z));
}

}  // namespace pathtracer::scene
