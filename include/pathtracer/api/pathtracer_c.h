#ifndef PATHTRACER_API_PATHTRACER_C_H
#define PATHTRACER_API_PATHTRACER_C_H

/* Flat C ABI over headless_renderer.h. One PtRenderer per thread, never shared; the caller allocates all output buffers and frees none. */

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque scene handle: one loaded scene, its BVH, its thread pool and its reusable framebuffers. */
typedef struct PtRenderer PtRenderer;

#define PT_OK 0
#define PT_ERROR 1
/* PtCamera is passed by value, so a caller built against a different layout must be caught rather than reading the fields as garbage. */
#define PT_ABI_VERSION 4
/* Lens projections, PtCamera.lens_projection: rectilinear is the pinhole, fisheye is Kannala & Brandt's polynomial. */
#define PT_LENS_RECTILINEAR 0
#define PT_LENS_FISHEYE_POLYNOMIAL 1
/* Tri-state sentinel for the optional request fields: defer to the default rather than forcing on or off. Any other value is rejected. */
#define PT_DEFAULT (-1)

/* Loads a scene; NULL on failure with a NUL-terminated reason in err (truncated to err_cap). scene_path is relative to asset_root. */
PtRenderer* pt_renderer_open(const char* asset_root, const char* scene_path, char* err, int err_cap);
void pt_renderer_close(PtRenderer* renderer);

/* The AOV table, exposed so a caller never hardcodes a parallel copy of it. Ids are dense in [0, pt_aov_count). */
int pt_aov_count(void);
const char* pt_aov_name(int aov);
/* Exact, as pt_aov_name spells it: "bounceCount" resolves, "bounce-count" does not. -1 if unknown. */
int pt_aov_id(const char* name);
/* Channels the AOV carries, what pt_render writes per texel: 1 for depth or a scalar filter, 2 for UV or motion, 3 for radiance. */
int pt_aov_channels(int aov);
/* Non-zero if this AOV needs light transport, so a caller can tell which requests the `samples` field affects. */
int pt_aov_needs_samples(int aov);

/* PT_ABI_VERSION this library was built with: a caller compares it to its own header's and refuses to call on a mismatch. */
int pt_abi_version(void);

/* Pose, lens and exposure; aperture/shutter_seconds/iso set EV only (no depth of field). pt_render names any field out of range. */
typedef struct {
    float position[3]; /* finite */
    float yaw_degrees; /* finite */
    float pitch_degrees; /* strictly inside (-90, 90) */
    float film_back_mm[2]; /* sensor gate width, height; finite, > 0 */
    float focal_length_mm; /* finite, > 0 */
    /* 0 < near_clip < far_clip, near_clip finite; far_clip may be +inf, the unbounded ray. */
    float near_clip;
    float far_clip;
    float aperture; /* f-number; aperture, shutter_seconds and iso each finite, > 0, with a finite EV100 */
    float shutter_seconds;
    float iso;
    /* PT_LENS_*. Every AOV renders under either projection; G-buffer depth is distance along the primary ray. */
    int lens_projection;
    /* k1..k4 of r(theta) = focal_length_mm * (theta + k1*t^3 + k2*t^5 + k3*t^7 + k4*t^9), as OpenCV `fisheye` reports them. */
    float fisheye_coefficients[4];
    /* Full angle across the image circle; half it is the polynomial's domain. Must lie in (0, 360] whichever projection is selected. */
    float fisheye_field_of_view_degrees;
} PtCamera;

/* profile.json's authored camera and window size -- the defaults a caller overrides one field at a time. */
void pt_renderer_default_camera(const PtRenderer* renderer, PtCamera* out);
int pt_renderer_default_width(const PtRenderer* renderer);
int pt_renderer_default_height(const PtRenderer* renderer);

typedef struct {
    PtCamera camera;
    /* The view motionVector measures motion from, either lens; NULL is camera itself, so motion reads exactly zero. */
    const PtCamera* previous_camera;
    int width;
    int height;
    /* Path-traced passes at one sample each, averaged. Ignored by a request whose AOVs are all G-buffer-backed. */
    int samples;
    /* Fixes the sampler's scramble. The same seed and request reproduce the same floats exactly. */
    unsigned int seed;
    const int* aovs;
    int aov_count;
    /* Tri-state, PT_DEFAULT hides the sky: whether a camera miss returns environment radiance. Primary miss only; it unlights nothing. */
    int show_sky;
    /* Tri-state, PT_DEFAULT keeping the scene's authored environment.lightEnabled: whether the environment is a light at all. */
    int env_light_enabled;
} PtRenderRequest;

/* Renders every requested AOV. out parallels request->aovs at width * height * pt_aov_channels(aovs[i]) floats, row-major top-left. */
int pt_render(PtRenderer* renderer, const PtRenderRequest* request, float* const* out, char* err, int err_cap);

/* Scene-referred linear to display-referred 8-bit sRGB, the viewer's chain and render_beauty's PNG encode. Buffers hold w*h*3. */
int pt_display_encode(const float* rgb, int width, int height, float exposure_ev, int display_transform,
                      unsigned char* out, char* err, int err_cap);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PATHTRACER_API_PATHTRACER_C_H */
