#pragma once

#include <cmath>

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

namespace pathtracer::config {

// Every JSON float read: get<float>() silently narrows a finite double past FLT_MAX (1e39) to inf, so the overflow is refused here.
inline float toFloat(const nlohmann::json& j) {
    // IEEE 754 round-to-nearest: a double at or past FLT_MAX + ulp/2 narrows to inf, every other finite double to a finite float.
    const auto narrowed = static_cast<float>(j.get<double>());
    if (!std::isfinite(narrowed)) {
        throw nlohmann::json::out_of_range::create(406, "number " + j.dump() + " is outside the float range", &j);
    }
    return narrowed;
}

// toFloat for an optional key, the json::value("key", fallback) form.
inline float floatOr(const nlohmann::json& object, const char* key, float fallback) {
    const auto it = object.find(key);
    return it == object.end() ? fallback : toFloat(*it);
}

}  // namespace pathtracer::config

// ADL hook (found via glm's namespace) so nlohmann can do j.at("key").get<glm::vec3>() instead of indexing components by hand.
namespace glm {

inline void from_json(const nlohmann::json& j, vec2& v) {
    if (!j.is_array() || j.size() != 2) {
        throw nlohmann::json::type_error::create(302, "expected a 2-element array for glm::vec2", &j);
    }
    v = vec2{pathtracer::config::toFloat(j[0]), pathtracer::config::toFloat(j[1])};
}

inline void from_json(const nlohmann::json& j, vec3& v) {
    if (!j.is_array() || j.size() != 3) {
        throw nlohmann::json::type_error::create(302, "expected a 3-element array for glm::vec3", &j);
    }
    v = vec3{pathtracer::config::toFloat(j[0]), pathtracer::config::toFloat(j[1]), pathtracer::config::toFloat(j[2])};
}

}  // namespace glm
