#include "pathtracer/debug/denoiser.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <OpenImageDenoise/oidn.hpp>

namespace pathtracer::debug {

using pathtracer::gfx::HdrImage;

namespace {

oidn::Quality oidnQuality(DenoiseQuality quality) {
    switch (quality) {
        case DenoiseQuality::Fast:
            return oidn::Quality::Fast;
        case DenoiseQuality::Balanced:
            return oidn::Quality::Balanced;
        case DenoiseQuality::High:
            return oidn::Quality::High;
    }
    return oidn::Quality::High;
}

}  // namespace

struct Denoiser::Impl {
    oidn::Quality quality;
    bool prefilterAuxiliary;
    oidn::DeviceRef device;
    // Built with prefilterAuxiliary only: albedo and normal are noisy at edges and behind blur, so each is cleaned before Beauty trusts it.
    oidn::FilterRef albedoFilter;
    oidn::FilterRef normalFilter;
    oidn::FilterRef beautyFilter;
    // Copies of the inputs at stable addresses: the filters bind them once per size, and the prefilters overwrite the aux in place.
    HdrImage colour;
    HdrImage albedo;
    HdrImage normal;
    HdrImage output;

    // The first error since the last check, OIDN keeping only that one; every failure surfaces with its own message.
    void throwIfFailed() {
        const char* message = nullptr;
        if (device.getError(message) != oidn::Error::None) {
            throw std::runtime_error(std::string("OIDN: ") + (message != nullptr ? message : "unknown error"));
        }
    }

    void bind(oidn::FilterRef& filter, const char* name, HdrImage& image) {
        filter.setImage(name, image.texels.data(), oidn::Format::Float3, static_cast<std::size_t>(image.width),
                        static_cast<std::size_t>(image.height));
    }

    // Reallocates the scratch and rebinds every filter only on a size change: commit is the filter's expensive setup step.
    void resize(int width, int height) {
        if (beautyFilter && colour.width == width && colour.height == height) {
            return;
        }
        for (HdrImage* image : {&colour, &albedo, &normal, &output}) {
            *image = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
        }
        beautyFilter = device.newFilter("RT");
        bind(beautyFilter, "color", colour);
        bind(beautyFilter, "albedo", albedo);
        bind(beautyFilter, "normal", normal);
        bind(beautyFilter, "output", output);
        beautyFilter.set("hdr", true);
        beautyFilter.set("cleanAux", prefilterAuxiliary);
        beautyFilter.set("quality", quality);
        beautyFilter.commit();
        if (prefilterAuxiliary) {
            albedoFilter = device.newFilter("RT");
            bind(albedoFilter, "albedo", albedo);
            bind(albedoFilter, "output", albedo);
            normalFilter = device.newFilter("RT");
            bind(normalFilter, "normal", normal);
            bind(normalFilter, "output", normal);
            for (oidn::FilterRef* filter : {&albedoFilter, &normalFilter}) {
                filter->set("quality", quality);
                filter->commit();
            }
        }
        throwIfFailed();
    }
};

Denoiser::Denoiser(const DenoiserOptions& options) : impl_(std::make_unique<Impl>()) {
    impl_->quality = oidnQuality(options.quality);
    impl_->prefilterAuxiliary = options.prefilterAuxiliary;
    impl_->device = oidn::newDevice(oidn::DeviceType::CPU);
    impl_->device.set("numThreads", options.numThreads);
    impl_->device.commit();
    impl_->throwIfFailed();
}

Denoiser::~Denoiser() = default;

HdrImage Denoiser::denoise(const HdrImage& beauty, const HdrImage& albedo, const HdrImage& normal) {
    for (const HdrImage* image : {&beauty, &albedo, &normal}) {
        if (image->width != beauty.width || image->height != beauty.height || image->channels != pathtracer::gfx::kRgbChannels) {
            throw std::invalid_argument("denoise: beauty, albedo and normal must be RGB images of one size");
        }
    }
    impl_->resize(beauty.width, beauty.height);
    std::copy(beauty.texels.begin(), beauty.texels.end(), impl_->colour.texels.begin());
    std::copy(albedo.texels.begin(), albedo.texels.end(), impl_->albedo.texels.begin());
    std::copy(normal.texels.begin(), normal.texels.end(), impl_->normal.texels.begin());
    if (impl_->prefilterAuxiliary) {
        impl_->albedoFilter.execute();
        impl_->normalFilter.execute();
    }
    impl_->beautyFilter.execute();
    impl_->throwIfFailed();
    return impl_->output;
}

}  // namespace pathtracer::debug
