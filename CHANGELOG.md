# Changelog

Newest first. The `Phase 0`-`Phase 5` blocks at the end are the original ordered build-out and keep
their own sequence; every entry above them is standalone, most recent first.

## Viewer benchmark sizing: `-size`, `-max-samples`, and no lost pass records

A viewer `-bench` at the authored 2048x1152 and 64 samples takes about 93 s per run, so a 7-round A/B took about 22 minutes. The
run is now sized from argv; the review benchmark is `-size 512x256 -max-samples 32`, **3.9 s per run**. Evidence is in
`results/viewer_size`.

- feat: `-size WxH` and `-max-samples N` replace `render.width/height` and `pathTracer.maxSamples` after the profile loads.
  They are parsed whole at the argv boundary with the profile's own bounds (extents >= 1, samples >= 0). The bench config
  already records the traced width, height and `max_samples`, so records of different sizes never compare.
- fix: `PathTraceDriver::takePassRecords()` returns every pass record since the last call, cancelled included; the frame
  loop takes them every frame. The bench read `lastPassRecord()` once per frame and lost records whenever two passes finished
  within one frame, which is every frame at 512x256. `lastPassRecord()` stays for the HUD and dashboard.
- fix: only a request at the settled `renderScale` restarts a bench capture. Before, the interactive accumulation
  (51x26 at 512x256) could reach `maxSamples` inside the 0.25 s settle window and be logged as the workload. Full-size runs
  were unaffected: their interactive accumulation outlasts the settle window.
- test: `driver_validate` `every_pass_record_is_taken_once_in_order` drains a 12-pass accumulation through
  `takePassRecords` and requires exactly passes 1..12, in order, with nothing left. `ctest` 214/214.

## Display encode cost: one shared OCIO processor, a row-wise encode into the caller's buffer

`pt_display_encode` re-parsed the builtin OCIO config and rebuilt its processor on every call. It encoded into a full-frame float
copy and a full-frame byte vector, then copied the bytes into the caller's buffer. Measured against `main` with an interleaved
process-per-round harness, 11 rounds. Evidence is in `results/display_encode`.

- perf: the sRGB and Rec.1886 CPU processors are each built once on first use, as thread-safe statics, in `ocio_cpu_transform.cpp`.
  `applyOcioDisplayTransform` takes a `Lut`, and Raw is the identity. The viewer's Beauty probe uses it, which deletes
  `main.cpp`'s duplicate cache and its OCIO include.
- perf: `encodeForDisplay` writes into a caller `std::span<unsigned char>`, one row at a time through a `width x 4` float scratch.
  RGBA is OCIO's native layout, so the CPU processor transforms the row in place without a conversion pass.
  `pt_display_encode` hands it the caller's `out`, so the byte temporary and the `memcpy` go. Peak scratch at 2048x1152 drops from
  28.3 MB plus 7.1 MB to 32 KB.
- measured: steady-state encode **1x1 5.57 ms -> 1.7 us**, the processor rebuild, 512x288 **0.397x [0.396, 0.398]**,
  2048x1152 **0.835x [0.835, 0.836]**. At 2048x1152, 39.3 of the remaining 62.5 ms is the per-texel sine-hash dither (two `sin`
  per texel, mirroring the shader) and the quantize.
- image: all 32 AOVs on cornell, macbeth and stump are byte-identical through `render_beauty`, 96 of 96 PNGs.
- test: `display_validate` gains two checks:
  - `encode_matches_whole_image_transform`: the row-wise encode is bit-equal to a whole-image OCIO pass followed by an
    untransformed encode; every code lies within the TPDF dither's 1.5-code reach of `255 clamp(v)`; Raw is the identity.
  - `rec709_lut_is_the_bt1886_inverse_eotf`: `V = L^(1/2.4)` within 1e-5 (ITU-R BT.1886 Annex 1, Lw = 1, Lb = 0).
  - Five mutations: four are caught. A dither keyed on the wrong row survives inside its own bound, so exact dither parity rests on
    the byte identity above.

## C ABI coverage: a validator that calls only the exported ABI

`api_validate` drives `HeadlessRenderer` in C++, so nothing in `ctest` crossed the C ABI that Python and other foreign
callers bind to. Evidence is in `results/c_abi_validate`.

- test: `c_abi_validate` links only the shared `pathtracer_c` and includes only `pathtracer_c.h`. It has 9 checks:
  - the library's `pt_abi_version` matches the `PT_ABI_VERSION` a C caller compiles
  - the AOV table round-trips, and out-of-range or unknown queries return their sentinels
  - `err_cap` truncates, NUL-terminates and never writes past
  - a request for every AOV writes exactly `w*h*channels` finite floats, checked with NaN-payload guards
  - 8 malformed requests return `PT_ERROR` naming the field and write nothing; null renderer, request or output return
    `PT_ERROR` with a reason
  - a fixed seed reproduces bit for bit and another seed differs
  - `pt_display_encode` writes exactly `w*h*3` bytes, clamps exactly at both ends, and rejects null or empty input
  - every handle query accepts NULL
- test: `tools/c_abi_header.c` compiles the header as strict ISO C11 (`-pedantic-errors`). Its `ptHeaderAbiVersion` is the
  C side of the version check.
- note: 6 of 7 mutations are caught. The survivor drops the encode's zero-extent check; OCIO and the allocation still raise,
  and the error is caught as `PT_ERROR`, so it is not observable at the boundary. `ctest` 222/222.
- note: Python tests under `ctest` and packaging are deferred to the Python phase.

## Sky default: hidden in the viewer and headless alike

The viewer hid the sky by default and headless showed it, so the same scene, camera and seed gave two backgrounds depending
on the caller. Both now hide it, from one definition. Evidence is in `results/sky_default`.

- fix!: `kDefaultShowSky = false` (`path_tracer.h`) is the default for `PathTraceDriver::Request::showSky`, the viewer's HUD
  checkbox and `HeadlessRenderer::Request::showSky` when unset. Through the C ABI (`show_sky = PT_DEFAULT`) and Python
  (`show_sky=None`), a camera miss now returns black. `PT_ABI_VERSION` stays 4, because no layout changed.
- image: only primary-miss radiance is removed. On cornell and macbeth at 512x288, every changed texel is within the 2 px
  sample reach (0.5 px jitter + 1.5 px Blackman-Harris radius) of a pixel-centre miss, and every texel beyond it is
  bit-identical. Stump also changes 51 texels further out, through sub-pixel gaps that the pixel-centre alpha cannot see. No
  texel gains radiance in any scene. `render_beauty` takes the headless default, so its default captures now have black
  backgrounds.
- test: `api_validate` `show_sky_changes_only_the_background` asserts the default equals `showSky=false`. pytest
  `test_show_sky_default_hides_the_sky` replaces `test_show_sky_default_is_unchanged`. `ctest` all green, pytest 42 passed,
  1 skipped.

## Geometry loader: mirrored transforms, generated tangents, every triangle topology

Three loader gaps are closed:
- A node with a negative determinant (a mirror) was loaded without re-winding its triangles, so the face normal pointed against
  the shading normal.
- A primitive without TANGENT failed the whole load.
- A point, line, strip or fan primitive failed the whole load, although the log said "skipping".

Evidence is in `results/geometry_loader`.

- fix: under a negative global determinant, the loader swaps v1 and v2 of every triangle (glTF 2.0 3.7.2.1) and multiplies
  `tangent.w` by `sign(det M)`, because sign det[M^-T N, M T, M B] = sign(det M).
  - Before, on a mirrored instance, NEE's `nearSide` test (`geoCos > 0 && shadingCos > 0`) never passed on the lit side. The
    light-leak rejection and every ray-origin offset also picked the wrong side, and the normal map's bitangent was inverted.
  - Mirrored cornell, environment off, measured against the horizontal flip of cornell: beauty mean |d| was 4.4x the two-seed
    noise floor with mean radiance at 0.23x; it is now 0.99x the floor with mean radiance at 1.001x.
  - Mirrored stump, with scene rotation zeroed so the mirror is the world x mirror, measured against the exact flip in the
    G-buffer: the normal lane was off by a mean of 11.6 deg (max 148 deg) and is now off by a mean of 0.001 deg. The residual is
    pixel-centre rounding of at most 1 ulp in UV, amplified by the 4K normal and bump maps; `geomNormal` matches exactly.
- feat: a primitive without TANGENT gets MikkTSpace tangents (Mikkelsen 2008). The reference implementation is vendored as the
  `third_party/mikktspace` submodule (zlib).
  - Generation runs in object space on the authored winding, before the mirror re-wind, as a baker sees the mesh.
  - It stores `w = -sign`, because glTF v runs down the image while tangent-space +y is image-up (glTF 2.0 3.9.3).
  - On the stump, regenerated tangents match the exporter's MikkTSpace handedness on all 61683 corners, with a max per-corner
    angle of 0.44 deg (the exporter ran MikkTSpace on quads). In the image, beauty moves by 1.9e-6 of its mean.
  - Cost: loading the 20561-triangle stump takes 31.9 ms instead of 1.9 ms. It is single-threaded and paid only by primitives
    that lack TANGENT.
- feat: TRIANGLE_STRIP and TRIANGLE_FAN expand to lists in the spec's per-triangle vertex order. POINTS and LINE* primitives,
  which have no area, are skipped with a log line. A primitive without an index accessor uses the implied 0..count-1. Macbeth
  rebuilt as a fan plus a strip, with a point and a line primitive added, renders its normal, geomNormal, tangent and shadow lanes
  bit-identical to the shipped list; beauty and directDiffuse differ only by rounding (max 1.4e-6).
- chore: `tools/gltf_tangent` is removed. It patched missing tangents in with a UV-unaware basis, which the loader now replaces.
- chore: the tangents it had written into cornell and macbeth are deleted (their accessors, views and trailing buffer bytes), so
  both now load with MikkTSpace tangents.
  - macbeth: all 32 AOVs are byte-identical, because MikkTSpace gives the same +x tangent on its axis-aligned charts.
  - cornell: only the tangent lane changes deterministically; every other G-buffer lane and shadow is byte-identical. The
    path-traced lanes change only their Monte Carlo realization, because the isotropic BSDFs sample in the rotated frame:
    - every lane differs per pixel by less than its two-seed noise floor
    - beauty's image mean moves by z = +1.25 and every radiance lobe by |z| <= 1.1, so no bias
  - Loading cornell takes 2.32 ms instead of 0.275 ms.
- test: five `io_validate` checks, built on a glTF fixture writer:
  - `gltf_mirrored_transform_keeps_geometric_and_shading_normals_agreed`
  - `gltf_strip_and_fan_triangulate_with_consistent_winding`
  - `gltf_points_and_lines_are_skipped_not_fatal`
  - `gltf_generated_tangent_follows_the_gltf_convention`
  - `gltf_generated_tangents_agree_with_authored_mikktspace`
  Each fails under the matching mutation (see `results/geometry_loader/break_tests.txt`).
- Shipped scenes are unaffected: all 32 AOVs on cornell, macbeth and stump are byte-identical before and after. Render
  `pass_ms` B/A = 1.0217 [0.9932, 1.0634], not resolved.

## Accumulation arithmetic: one running mean for the viewer and headless

The viewer averaged passes with an incremental running mean. Headless summed them in float and divided once at the end. Both
produce a mean, but they round differently, so the comments calling the two bit-identical were false. Headless now uses the
viewer's arithmetic. Evidence is in `results/accumulation_arithmetic`.

- fix: `foldRunningMean` and `foldLuminanceM2` (`path_trace_driver.h`) are the driver's two inner loops, lifted verbatim:
  - `foldRunningMean` is m_n = m_{n-1} + (x_n - m_{n-1}) / n
  - `foldLuminanceM2` is Welford's (1962) M2 in West's (1979) form
  - `PathTraceDriver` folds out of place into the fresh pass buffer, and `HeadlessRenderer` folds in place on the lanes a request
    needs. Both run the same kernels, invN and per-element order, and both keep the first pass as drawn.
- fix: `averageAccumulators` and headless's float sums are removed, along with the Welford mean it rebuilt as `sum * (1/(n-1))`
- why the running mean, of the two float32 options:
  - Both have an O(nu) worst-case forward error (Higham 2002, §4.2).
  - The running mean is exact on a constant signal, where `x - m = 0`. A float sum stops being exact once `k·x` needs more than
    24 bits.
  - Its update is centred: it loses a sample only when `|x - m|/n` is below half an ulp of m, whereas a sum loses one when `|x|`
    is below half an ulp of n·m (Chan, Golub & LeVeque 1983).
  - It cannot overflow. It also needs no final divide pass and costs the same 12 B per element per pass.
- why not a double or Kahan-compensated accumulator: it would double the accumulator memory and the viewer's accumulation
  bandwidth for a typical error already about 100x below binary16 output resolution at 65 536 passes
- image:
  - the viewer is unchanged: its converged Beauty CRC (1462477921), ray counts and window capture are identical
  - headless Beauty at the viewer's `-bench` workload now has the viewer's CRC, where before it differed
  - headless path-traced lanes move by rounding only: median 1 ulp, max 7 ulp, one texel at 72 ulp (relative 7e-6)
  - SNR moves only where the measured dispersion is rounding residue (SNR >= 1.5e5)
  - two macbeth texels whose samples were all identical now have M2 = 0 and so SNR 0, the defined converged value, instead of a
    spurious 9.3e7
  - G-buffer AOVs are byte-identical
- perf: no resolved change. Viewer `pass_accumulate_ms` B/A = 1.0015 [0.9917, 1.0090], and headless `user_s` B/A = 0.9996
  [0.9980, 1.0034].
- test: `driver_validate` `in_place_fold_is_bit_identical_to_driver` requires the driver's 10 published lanes and M2 to equal, bit
  for bit, headless's in-place fold. It fails on the old sum-then-divide (525/1536 floats) and on a mis-ordered M2 fold (64/64).

## Camera validation: every parameter, at every boundary

The C ABI checked only the lens, so pose, focal length, aperture, shutter, ISO, film back and clips went through unchecked.
`profile.json` checked four of them with `<= 0`, which lets `inf` through, and no path checked the clips or pitch. A camera's
invariants are now defined once and checked wherever a camera is built from outside input. Evidence is in
`results/camera_validation`.

- feat: `Camera::validate(error)` holds every rule, and its error names the first bad field and its value:
  - position and yaw must be finite
  - pitch must lie strictly inside (-90, 90). The yaw/pitch basis is undefined at the pole (gimbal lock), and past it a pitch is
    an alias for yaw + 180. `float(pi/2)` is the first float past pi/2, so a strict `<` is exact and no margin is needed.
  - film back, focal length, aperture, shutter and ISO must each be finite and positive
  - `0 < near < far` with near finite; far may be `+inf`, the unbounded ray
  - the pinhole half-height `h/2f` and `ev100()` must be finite, because finite extremes can overflow them
  - the lens field of view must lie in (0, 360] and the polynomial must be provably monotone, under either projection
- feat: the checks run at four places:
  - `HeadlessRenderer::render`, for the camera and the previous camera. This is the boundary for C++ callers and for `pt_render`,
    and Python raises `RuntimeError` with the reason.
  - `HeadlessRenderer::open`, for the profile camera
  - viewer startup, which aborts with the reason
  - `loadFilmBackPresets`, which checks each preset with `Camera::validFilmBack`, because the HUD can switch to any of them
- **breaking**: a camera these rules reject is now refused where it used to render. Examples: `near_clip` 0 or negative,
  `far_clip <= near_clip`, pitch of ±90 or beyond, a NaN or infinite pose, a zero film back, a zero or NaN exposure scalar.
- fix: `viewBasis` computes the pinhole half-height as `h/2f` directly instead of `tan(atan(h/2f))`. At small focal lengths the
  round trip saturates where atan rounds to a float at pi/2. On this libm the half-height then caps at 1.3e7 (off by more than 1%
  below f = 7e-5 mm); a correctly rounded atan makes it negative instead, which inverts the frame. Across the HUD's 6-300 mm range
  and every `sensor.json` gate, 13% of combinations change by up to 8 ulp of the half-height.
- refactor: the lens range checks duplicated in `toCamera` and `parseLens`, and `profile_config`'s `validCamera`, are removed.
  `toCamera` now only decodes.
- note: rendering is unchanged for the shipped camera. All 192 AOV PNG/EXR files and the viewer captures are byte-identical.

## Camera reset during an orbit releases the cursor

An orbit is two pieces of state: the controller's orbit flag and the window's disabled cursor. The LMB release cleared both, but
`resetToDefault()` cleared only the flag. Pressing `0` mid-orbit therefore left the cursor hidden and its motion detached from the
pointer, and the later release saw no orbit and never unlocked it. Evidence is in `results/reset_during_orbit`.

- fix: `endOrbit(window, app)` (`main.cpp`) is the one place the orbit flag and the cursor lock are cleared. The LMB release and
  the `0` key both call it, so the reset first ends any orbit in progress, whose pivot was picked from the pose being discarded
- fix: `resetToDefault()` restores only the pose, so the controller cannot end an orbit without the window releasing its cursor
- fix: a side effect of the stale lock is also gone. Because GLFW ignores a `DISABLED` to `DISABLED` change, the next orbit kept the
  restore point from the first one, and its release warped the pointer back to where that earlier drag began
- note: rendering is unchanged. All 192 AOV PNG/EXR files are byte-identical before and after, `libpathtracer_c` is byte-identical,
  and the viewer's converged Beauty CRC and ray counts match

## Primary-ray G-buffer: every AOV under every lens, the rasterizer removed

The 15 G-buffer AOVs came from a scan converter that projects by a perspective divide, so a fisheye refused them all. Each pixel
now casts one unjittered pixel-centre camera ray through the path tracer's own accel and bounce-0 sampling calls, which is correct
for any projection by construction. Nothing else used the rasterizer, so it is deleted. Evidence is in `results/primary_ray_gbuffer`.

- **breaking**: `depth` is the distance along the primary ray to the first hit, not planar view z: the one depth a fisheye past
  90 degrees off-axis still defines. `lookahead` ramps on that distance. Rectilinear depth grows by `1/cos` of the off-axis angle
- feat: all 15 G-buffer AOVs render under the Kannala-Brandt fisheye in the viewer, the headless renderer, the C ABI and Python. The
  refusals (API error, startup error, HUD greying, the switch back to Beauty) are removed. Past the image circle the lanes read the
  background: 0, IOR -1, zero motion
- feat: `Camera::primaryRayDifferential` returns a primary ray with its closed-form d(dir)/d(ndc) (Igehy 1999). The pinhole arm
  projects the affine rate off the ray; the fisheye arm chains d(theta)/dr = 1/(f theta_d'(theta)) and sin(theta)/r azimuthally.
  `fisheyeSample` is the one inverse-lens solve the ray and its differential share, and primaryRay's arithmetic is unchanged
- feat: wireframe and instance boxes use one projection-general first-order line metric. A mesh edge is `|n.d| / |J^T n|` over
  the hit triangle's three eye planes (Baerentzen et al. 2006). A box edge is its gnomonic image at d, a segment or a half-line
  where it crosses behind the view, pulled back to pixels by J's pseudo-inverse and range-tested against the first hit. A
  per-instance bounding cone and a vectorised per-edge plane reject keep the full test to pixels within reach of an edge
- refactor: `RasterGBuffer`/`renderRasterGBuffer` become `GBuffer`/`renderGBuffer` (`gbuffer.h`, taking the `EmbreeAccel`).
  Stage timing `rasterMs` is `gbufferMs`, `RenderStats::rasterMilliseconds` is `gbufferMilliseconds`, the dashboard and spec rows
  read `gbuffer`, and the bench column `raster_ms` is `gbuffer_ms`. `raster_bench` is `gbuffer_bench`. `aovSelectionAvoidsRasterizer`
  and the unread `lastRasterMs` are removed
- test: `gbuffer_validate` replaces `rasterizer_validate`. Every lane equals a pixel-centre ray oracle bit for bit on 5 poses (2
  pinhole, near clip, 180 and 220 degree fisheyes), and worldPos reprojects onto its pixel within its rounding over the incidence.
  Wireframe and box flags agree with exact projected-edge distances on every pixel outside the first-order model's Taylor band
  (including half-lines and an oblique occluder). Watertightness, motion (now also from a fisheye current view, budget relative to
  the projection's conditioning), occlusion, per-instance boxes and constant inputs are kept
- test: `camera_validate` checks the differential against double central differences on 3 lenses. `api_validate` renders every AOV
  through both lenses and checks the on-axis depth agrees across them. `image.gbuffer_aov_fisheye` gates the fisheye end to end.
  Python checks depth equals |worldPos - eye| under both lenses
- note: break tests were run. A flipped fisheye azimuthal term, planar depth, an over-tight edge reject, a non-metric line
  distance and a swapped current projection each fail their check; a dropped projective edge parameter was missed until the box
  check gained an occluder, and now fails 15 pixels
- image: the 17 path-traced and filter AOVs and fisheye Beauty are byte-identical on cornell and macbeth (72/72 files). Rectilinear
  G-buffer lanes match the rasterizer at every interior pixel to float rounding, except depth's new meaning and 36 cornell pixels on
  the room's four corner creases, where a pixel centre ties two walls and the two producers break the tie differently
- perf: the G-buffer costs more. At 2048x1152 `gbuffer_ms` B/A is 1.49 [1.47, 1.51] on macbeth (44.8 to 66.6 ms) and 1.67
  [1.66, 1.68] on cornell (65.3 to 108.8 ms), and 2.59 / 2.09 on gbuffer_bench's dense synthetic layers: Embree single-ray traversal
  replaces coherent scan conversion. An 8-wide `rtcIntersect8` packet path was measured, matched bit for bit and gained nothing on
  this arm64 build, so it is not kept. The viewer pays it once per camera settle. Peak RSS falls 2-7%; Beauty `pass_ms` is unchanged

## Quad light placement: position, rotation and size around a centred gimbal

A quad light was authored as a corner `origin` plus two edge vectors, so cornell's panel read as offset by half its size
(`origin [-0.15, 0.49, -0.15]`) while centred on `(0, 0.49, 0)`. A light now places a unit quad centred on its own origin with the
same translate-then-Euler convention as the model root, so `position` is the panel's centre and perpendicular edges hold by
construction. Evidence is in `results/quad_light_placement`.

- **breaking**: `lights[]` keys `origin`, `edge0` and `edge1` are replaced by `position` (centre), `rotation` (degrees,
  `Rz * Ry * Rx`, X first, as `model.rotation`) and `size` (`[w, h]`). At rest the quad lies in local XY and emits along local -Z.
  World placement is `rootTransform * T(position) * Rz * Ry * Rx`. A file in the old form fails to load with a named error
- feat: each light object has a closed key set, as `model` and `environment` already do, so a stale `edge0` or a misspelt
  `twoSided` is rejected instead of being ignored
- refactor: `placementTransform(position, rotation)` (`material_binding.cpp`) is the one `T * Rz * Ry * Rx` that `rootTransformOf`
  and `buildQuadLights` share. `QuadLight` is unchanged: corner, edges and cached normal, so the sampler and BVH injection are untouched
- refactor: the perpendicularity check and its `kMaxEdgeCosine` bound are gone. Edges are rotation columns scaled by `size`, so
  `|cos|` is float rounding (measured 0), not authored precision. The zero-length check becomes `size > 0`, which also rejects a
  negative extent, since mirroring the quad flips its emitting face
- feat: `json_glm.h` parses `glm::vec2`
- test: `quad_light_placement` (`integrator_validate`) checks the centre, the normal, `|edge0| = w`, `|edge1| = h` and edge
  orthogonality at rest, under `Rx(-90)`, under `Rx(90)` then `Ry(90)` (which pins the Euler order: Y first would give -X, not +Y)
  and under a moved scene root. It also checks that cornell's new light reproduces the retired corner and edges. Its tolerance is
  Higham's `gamma_n` for 8 chained 4-term products, and the measured worst case is 5.96e-8 against 1.53e-5
- test: `scene_config_rejects_malformed_input` gains a zero size, a negative size, the retired corner form and a stale `edge0`
  beside a full placement. The skew row is removed, since skew can no longer be authored
- note: break tests were run. Flipping edge1's sign (emits +Z) fails 5 of 5 assertions, swapping the Euler order fails the
  order row, and dropping the scene root fails the moved-root row
- note: cornell's `Rx(-90)` puts `cos(90 deg)` at its float value of -4.37e-8, so the panel is tilted by that many radians and its
  corner moves by about 1e-8. The 32 AOVs at 512x288, 16 passes: 8 are byte-identical in EXR and 17 in PNG. Beauty moves by 1/255
  in 10 pixels, and its EXR by at most 3.2e-5 absolute (4.9e-4 relative) through NEE samples that move by ulps. The G-buffer
  moves only inside the panel's own pixels (rows 18-23), and only one pixel, a primary ray grazing the panel's corner, changes
  beyond 1e-4. Beauty `pass_ms` B/A is 1.0085 [0.967, 1.029], not resolved, and peak RSS is unchanged

## Function size: every first-party function inside the 60-line limit, and glm for the shading frame and environment rotation

clang-tidy's `readability-function-size` (60 lines, `.clang-tidy`) flagged 14 functions in `src/`. All 14 are now inside it, plus
`buildSphericalRectangle` and `readExrChannels`, which sat at the edge. `tracePath` and `renderPathTraced` stay whole for the GPU
port and are the only two left. Every split moves whole statements, so no expression is re-associated, and output is unchanged
apart from the environment rotation below. Evidence is in `results/function_size`.

- refactor: `ShadingFrame` is `glm::mat3`, its columns tangent, bitangent and normal. `frame * v` is local to world and
  `v * frame`, the transpose, is world to local. glm 1.0.3 evaluates both in the operand order `toWorld`/`toLocal` used, so every
  AOV is bit-identical
- refactor: the environment's Y rotation is one `glm::mat3` from `glm::rotate`, built once per `LightSet`. The inverse is the
  row-vector product, its exact transpose. `YRotation` and `rotateAboutY` are removed
- refactor: `computeLobeProbabilities` splits into `conductorAtWo`, `coatTerms`, `selectOpaqueLobes` and
  `selectTransmissiveLobes`. `sampleBsdf` splits into `weighSample` and one function per strategy, with the sampler draw order
  unchanged. `buildSphericalRectangle` splits into `sphericalRectangleFrame` and `sphericalRectangleAngles`
- refactor: `loadProfileConfig` reads each block through its own parser, then runs `validCamera`/`validRender`/`validPathTracer`/
  `validCounts`. Read and check order are unchanged, so every diagnostic is too. `readExrChannels` moves out `bindExrChannels` and
  `allFinite`
- refactor: `rootTransformOf` and `baseSettingsOf(profile, material, samplesPerPixel)` move to `material_binding`, replacing
  `initializeApp`'s copies
  - `HeadlessRenderer::open` loads its files through `loadSceneInputs`
  - `render` validates, then accumulates, rasterizes and filters in private members
  - `pt_render` decodes through `decodeRequest`
- refactor: `driverLoop` traces through `tracePass` and folds, publishes and records through `finishPass`. The active request is
  dereferenced once, under one NOLINT. `drawStageRows` writes each left pane through `appendStageCell`/`appendBurstCell`
- refactor: main.cpp
  - the immutable scene is one `AppScene` member, built by `loadAppScene`, as `HeadlessRenderer` holds it
  - `pathTraceSettings` becomes `scene.baseSettings`, since nothing writes it after startup
  - `totalPoints` is derived where it is shown
  - session-constant `AppResources` fields take in-class initializers (C++ Core Guidelines C.48)
  - `initializeApp`, `updateHud`, `renderFrame` and `main` delegate to `loadFilmBackCatalogue`, `makeDebugCamera`,
    `printStartupSpec`, `CameraEdits`, `beginFrame`, `validStartup`, `configureBench` and `runApp`
- behaviour: a startup failure now names either the scene load or OCIO, not one combined line. OCIO is created after the scene
  loads, and the GL presentation objects are constructed with `AppResources`, after the BVH build
- image: bit-identical everywhere at environment rotation 0, which is the headless, Python and viewer default
  - 128/128 raw AOV dumps, all 32 AOVs on cornell and macbeth with the environment light off and on, plus a moved previous camera
  - render_beauty PNG and EXR byte-identical; the viewer's `-bench` converged-image CRC is unchanged (3814828525)
  - the first dashboard redraw from fixed inputs is byte-identical
  - profile diagnostics are identical over 1129 malformed profiles (every single fault and every pair); `pt_render` errors are
    identical over 14 malformed requests
- image: at a nonzero HUD rotation, environment lookups move by rounding
  - the old closed form `(v.x * c) + (v.z * s)` contracted to an FMA; glm's `mat3 * vec3` rounds each product separately, and
    `glm::rotate`'s diagonal `c + (1 - c)` is 1 - 2^-24 at 46 of 360 integer angles
  - rotated directions: 21% of components differ, 357/360 angles
  - `LightSet`: Le relative change median 5e-7, max 6.5e-4; 0.04% of `pdfEnvironment` evaluations land in the neighbouring texel
    of the piecewise-constant CDF; NEE pdfs unchanged
  - 64-pass Beauty at 37, 90 and 211 degrees: the before/after RMSE is 1e-6 to 7e-5 of the seed-to-seed Monte Carlo RMSE, max abs
    1.1e-4
- measured, paired against `main` (4903001), 11 rounds: render_beauty `pass_ms` cornell 0.999x [0.970, 1.022], macbeth 0.975x
  [0.900, 1.042], both unresolved; RSS 1.000x
- test: no new checks. Each split is covered by the exact before/after comparisons above, the existing `bsdf`, `nee`,
  `integrator`, `io`, `api` and `driver` checks, and pytest. `ctest` 195/195, pytest 32/32 with the OpenEXR module installed

## motionVector: geometric screen-space motion replaces Optic Flow

Optic Flow estimated motion from Luminance pixels, so it was coarse, noise-driven, the prior wherever texture was missing, and
not zero for a static view. `motionVector` is the geometric motion field instead: the displacement of the point each pixel sees,
reprojected through the previous camera (reverse reprojection, Nehab et al. 2007). It is exact, dense and zero when nothing moves.
Evidence is in `results/motion_vector`.

- **breaking**: Optic Flow is removed, with `opticFlow.source` in profile.json, `AovSource::Derived`, `optic_flow.{h,cpp}`,
  `flow_validate`, `luminanceMeanVariance` and the scale-space octave API (`octaveLevel`, `octaveLevelCount`, `addExpanded`,
  `ScaleSpaceLevel`) that only it used. `aovNeedsLightTransport(aov)` takes one argument again
- **breaking**: C ABI 4. `PtRenderRequest.previous_camera` is a `const PtCamera*`, where NULL means the request's own camera.
  `pt_aov_needs_samples(aov)` and Python's module-level `aov_needs_samples` return, since no answer depends on the profile.
  `motionVector` takes id 12 and `colourOpponent` moves to 13; every other id is unchanged and the count stays 32
- feat: `AovId::MotionVector` (`"motionVector"`) is a G-buffer lane with 2 channels, `(dx, dy)` in current-frame pixels, +x right,
  +y down: `x_now - x_previous`, so `current(x) ~ previous(x - d)`
  - a hit reprojects its rasterized world position, `(p, 1)`. A miss reprojects its pixel-centre direction as a point at infinity,
    `(d, 0)`, so the environment moves under rotation and zoom and never under translation
  - both terms are one projection of one point, so identical cameras give exactly zero with no special case. Divides, not
    reciprocals: `x * (1/z) - x' * (1/z')` contracts to an FMA that rounds the two sides differently
  - zero where the previous view has no image of the point: behind a pinhole, or past a fisheye's thetaMax
  - both views are projected at the current aspect, so a resize or render-scale change reads as no motion
  - display: dx and dy share one pooled, zero-centred range, so zero is (0.5, 0.5, 0)
- feat: `Camera::pinholeMatrix` returns the pinhole matrix `P = K [R | -R c]` (Hartley & Zisserman 2004, eq. 6.8) as NDC rows.
  `Camera::project(basis, (p, w))` is `primaryRay`'s inverse for both lenses, with Kannala & Brandt's forward model for a fisheye.
  An inline static `project(matrix, point)` serves per-pixel callers
- feat: the request carries the previous view. `HeadlessRenderer::Request::previousCamera` defaults to the camera itself, and
  Python has `render(previous_camera=...)`; either lens is accepted. The viewer passes the previous frame's camera, and the raster
  trigger includes that view, so the first still frame after a move re-rasterizes to zero
- perf: motion is a per-row pass after shading. It reads `worldPos`, projects through each view's pinhole matrix and dispatches on
  the lens once per row, not per pixel
- measured, paired against `main` (594ed45), 11 rounds:
  - macbeth G-buffer `raster_ms` at 2048x1152: 1.110x [1.065, 1.176]
  - `raster_bench` `frame_ms`: 1.074x [1.036, 1.124]
  - Beauty `pass_ms`: 1.015x [0.974, 1.376], unresolved
  - RSS +3.3% (macbeth) / +6.0% (raster_bench): the 2-channel lane, 18.9 MB at 2048x1152
- image: the 31 shared AOVs are byte-identical on cornell and macbeth, in EXR and PNG (124/124). Against an independent NumPy
  reprojection, the largest error is 1.0e-4 px over pans up to 138 px, dollies, tilts and trucks. A static view has no non-zero
  texel
- test: `camera_validate` gains `project_inverts_primary_rays` and `project_refuses_points_outside_the_lens_domain`.
  `rasterizer_validate` gains four motion checks:
  - exact zero for an unmoved camera
  - a double-precision pinhole and equidistant-fisheye oracle over rotation, translation, zoom and a lens toggle, within
    16 ulps x kWidth
  - the environment held still under translation
  - zero from a view facing away

  `api_validate` gains `motion_vector_display_pools_its_two_lanes` and `motion_vector_follows_the_previous_camera`. pytest gains
  `test_motion_vector_follows_the_previous_camera`. `ctest` 195/195, pytest 31 passed plus 1 skipped (no OpenEXR module in the
  test env)

## camelCase AOV names, matched exactly

- **breaking**: `kAovNames` is camelCase with acronyms kept upper case (`beauty`, `worldPos`, `indirectDiffuse`, `HSV`, `DoG`,
  `objectID`), the one spelling the HUD, `--aov`, `-bench-aovs`, profile.json's `opticFlow.source`, `pt_aov_name`/`pt_aov_id`
  and Python's `AOVS`/`aovs=` share. `aovIdFromName` is an exact match: `"Bounce Count"`, `"bounce-count"` and `"BounceCount"`
  no longer resolve and fail with the valid names listed. Bench logs record the new names in `"aov"`, so records logged before
  this no longer match earlier ones by AOV name
- chore: the shipped profile's `opticFlow.source` is `"luminance"`

## Optic Flow: image motion between consecutive views of a profile-named source

A new AOV, `Optic Flow`, measures apparent motion from the pixels of the AOV named by profile.json's `opticFlow.source`
(default `Luminance`, the eye's luminance-driven motion pathway, Livingstone & Hubel 1987). This is optical flow in Horn's
sense, not the geometric motion field: it works under either lens, and it is what a retina-to-MT pathway measures. The estimator
stacks three standard pieces:
- phase constancy on the shipped Morlet bank (Fleet & Jepson 1990);
- per-texel intersection of constraints across orientations and source channels (Simoncelli & Heeger 1998);
- a Bayesian coarse-to-fine posterior from a slow-speed prior (Simoncelli 1999; Weiss, Simoncelli & Adelson 2002).

It is evidenced by `results/optic_flow`: 62 before/after EXRs, paired benches, and real renders scored against the exact motion field.

- feat: `AovId::OpticFlow` (after `ColourOpponent`, so later ids shift by one), 3 channels: `(dx, dy)` in current-frame pixels,
  `current(x) ~ previous(x - d)`, and sigma, the posterior std along the least-determined axis. A view with no same-size
  predecessor reads the prior, zero flow at the prior std, never a fabricated zero. `AovSource::Derived` is its producer;
  `aovNeedsLightTransport(aov, opticFlowSource)` answers for its source, so viewer, HUD and headless gating follow the source
- feat: `debug/optic_flow.{h,cpp}`, `opticFlowAov({mean, variance}, {mean, variance}, pool)`; every constant derives from the bank:
  - octave k demodulates at `morletCarrier(t0 4^k)` and cascades the baseband, exactly the shipped bank at 2^k (level 0 is
    bit-identical to the Gabor AOV's), with the levels that fit given by `octaveLevelCount`
  - the previous frame's responses are read at an integer warp with the carrier rotation restored, so nothing is resampled. The
    slope and signal power come from the current frame, so the information cannot change with the warp
  - weights are Rice `2P/(v0 + v1)` with `P = |C|^2 - v`, where v is the known texel variance through the envelope's energy,
    unseen content behind the mirror (`2 V_I sum_out h^2`) and float rounding; overdispersion is pooled over the envelope
    (McCullagh & Nelder 1989)
  - a square-root information filter (Bierman 1977) folds eigen-rows by Givens rotations. Re-linearisation re-reads only texels
    whose warp moved and freezes 2-cycles. The prior is the coarsest octave's unambiguous range, pi/omega_K
- feat: profile.json `"opticFlow": {"source": ...}` names any AOV but Optic Flow itself (`aovIdFromName`), and is refused at
  load otherwise. Luminance's noise comes from the Welford second moment (`luminanceMeanVariance`); other sources are
  weighed as deterministic
- feat: pairing. Headless pairs each `render()` that requests flow with the previous one that did; the viewer pairs successive
  generations of the source, holding copies so no pool slot is pinned, in its own cache beside the filter cache. A failed
  render leaves the history alone
- feat: C ABI 3: `pt_renderer_aov_needs_samples(renderer, aov)` replaces `pt_aov_needs_samples`, the answer now depending on
  the renderer's profile. Python's `Renderer.aov_needs_samples` replaces the module function. `render_beauty` refuses
  `--aov "Optic Flow"`, having one view
- feat: display pools dx and dy into one signed range, so the direction survives, and shows sigma against the prior, an exact
  ceiling. `bipolarDisplay`'s range factors into `pooledRange`
- refactor: `gaborAov` uses the extracted `morletPlaneWave` / `demodulate` / `morletBaseband`; `octaveLevel`,
  `octaveLevelCount`, `ScaleSpaceLevel` and `addExpanded` are restored to the scale space (single diffusion per grid, exact by
  the semigroup)
- measured: Gabor `filter_ms` 1.0010x [0.9754, 1.0133] and Beauty `pass_ms` 0.9974x [0.9927, 1.0100], both unresolved; RSS
  +0.1%. Flow costs 39 ms at 256x128, 0.26 s at 512x288 and 5.5 s at 2048x1152 (8 threads)
- image: the 31 existing AOVs are byte-identical on cornell and macbeth, EXR and PNG. Against the geometric motion field, macbeth
  at 64 spp reads a median end-point error of 0.010-0.036 px with 3-sigma coverage 0.89-1.00, and 0.03-0.11 px with 0.93-0.99
  at 8 spp
- note: cornell under-covers, at 0.60-0.82. Glossy clay, grazing walls and creases are systematic misfits along weakly
  constrained directions, and noiseless WorldPos shows the same shortfall (0.84). Range is the coarsest octave's: 5.1 px at
  256x128, 82 px at 2048x1152
- note: a 1-pass view has unknown Luminance noise (Welford needs n >= 2) and is weighed as deterministic. Mid-motion in the
  viewer, a view is often 1-2 passes, so sigma is overconfident: with either view at 1 pass, 3-sigma coverage at 256x128 is
  0.53-0.69 on cornell and 0.54-0.85 on stump, against 0.87-0.96 at 8/64 passes
- test: `flow_validate` (10 checks):
  - exact zero for identical frames, the prior bit for bit, and a joint 2^k gain changing no bit
  - integer shifts converging at every interior texel
  - 3-sigma coverage at or above the F(2,3) floor 0.875 for translations, the border band, a grating, a mirror pair and a
    known-noise flat half
  - nine mutations, each failing a check
- test: `filter_validate` gains the octave cascade against direct diffusion, the level-count steps and constant expansion.
  `io_validate` gains `profile_config_optic_flow_source`. `api_validate` gains consecutive pairing, request-mix independence and
  the pooled display. pytest checks AOV count 32, flow pairing and `aov_needs_samples`. `ctest` 204/204, pytest 32/32

## Constant material inputs: unbound slots resolve to constants at load

An unbound `Material` slot was a 1x1 identity texture, bilinear-filtered at every shading vertex, and its bump slot cost four
more taps whose difference was zero by construction. Each slot now holds a constant or a bound texture, so only bound maps are
filtered: constant folding of unconnected shader parameters (Gritz et al., *Open Shading Language*, runtime optimisation).
Measured against `main` (3b06993) with `bench_compare run`, 11 interleaved rounds, `results/constant_inputs`.

- perf: `MaterialInput<T> = std::variant<T, TextureHandle>`; `Material`'s slots are `baseColor`, `normal` (vec3), `bump`,
  `roughness` (float) and `specular` (vec3), and default member initializers state the neutral constants, so `Material{}`
  replaces `makeDefaultMaterial()` and `material.cpp`. The scene JSON keys (`baseColorTexture`, ...) are unchanged
- perf: `gbuffer_shading.cpp`'s `evaluate` returns a constant input as is and filters only a bound texture. `buildShadingFrame`
  skips bump mapping for a constant height, whose gradient is zero analytically (Blinn 1978), not by four taps cancelling
- refactor: `bindSceneTextures` takes each slot's channel count from its input type (`float` -> R, `vec3` -> RGB), not from the
  default material's 1x1 texture; the load-once cache, atomicity and diagnostics are unchanged
- measured: cornell (every slot unbound) `pass_ms` **0.949x [0.946, 0.950]**, the rasterizer's `frame_ms` (`raster_bench`)
  **0.704x [0.687, 0.716]**. macbeth 0.975x [0.926, 1.028] and stump, every slot bound, 0.995x [0.982, 1.003]: both unresolved.
  RSS unchanged. In cornell's CPU profile `sampleBilinear` falls from 9.4% to 2.1% of busy samples, the remainder the HDRI alone
- image: macbeth and stump byte-identical in Beauty, Albedo, Normal and Roughness, cornell in Albedo and Roughness. cornell's
  Normal is within 2 ulp on 1% of components: the old path renormalised the already-unit mapped normal after its zero bump tilt.
  The perturbed normals reroute 5 of 46.4M rays, so Beauty decorrelates per path: RMSE 3.9e-5 (PSNR 112 dB), mean signed
  difference -4.5e-8 against a standard error of 2.9e-8, no bias. Viewer captures differ at 28 Beauty pixels by 1/255
- test: `constant_inputs_match_unit_textures` asserts over 256 random materials, frames, uvs and settings that constants resolve
  `BsdfParams` and the shading frame bit-identically to the equivalent 1x1 textures, and that a bound constant bump texture
  only renormalises the normal. Mutation-checked: decoding a scalar texture's G channel fails it. `texture_binding_resolution`
  asserts an unbound slot stays the neutral constant, not a texture. `ctest` 187/187

## Framebuffer memory: every AOV stored at its declared channel count

Each AOV image holds the 1-3 channels `aovChannels` declares rather than four floats, and the display texture is uploaded
in its own storage type. Measured against `main` (96ccd41) with `bench_compare run`, 7 interleaved rounds,
`results/framebuffer_memory`.

- perf: `HdrImage` is `{width, height, channels, texels}`. `makePathTraceResult` and the G-buffer allocate every lane from
  `aovChannels` through `pathTracedLane`/`gbufferLane`; the new `pathTracedLanes()`/`gbufferLanes()` replace the
  hand-kept 10- and 14-entry lane arrays in `accumulateMean` and the row clear. A result slot drops from 40 to 24 floats per
  texel (360 -> 216 MiB at 2048x1152) and the G-buffer from 56 to 29 (504 -> 261 MiB). No alpha is stored anywhere:
  nothing read it, and Alpha is its own lane
- refactor: `writeTexel` takes the value at the lane's arity (`float`, `vec2`, `vec3`) and asserts it equals the image's
  channel count; it and `makeImage` move to `hdr_image.h`. The filters allocate at `aovChannels`; `writeScalar` is gone
- refactor: `HdrImage::rgb(pixel)` states the display expansion once: one channel broadcasts, a second leaves blue 0. The
  texture swizzle (RRR1, RG01, RGB1), the HUD probe and the PNG export all follow it. `aovDisplay` and `bipolarDisplay`
  take an `HdrImage`; the SNR pre-map is one channel, Bounce Count's three
- perf: the display texture is staged in its storage type and uploaded without driver conversion: R and RG as they are,
  RGB padded to RGBA, since Metal, Vulkan and D3D have no sampled three-lane half or float format. Uploading RGB directly
  cost 8.0 ms against main's 4.8, because the driver expands RGB sources on the CPU; staged, it is 2.3 ms.
  `GL_UNPACK_ALIGNMENT` is set to 1 for the odd-width R16F rows the interactive scale produces
- fix: the display texture's half values round to nearest even. The macOS GL driver truncated float uploads toward zero
  (all 1,048,576 texels of a probe), dimming every texel by up to one half-precision step; on screen this moves at most
  1/255, and only upward for unsigned AOVs
- feat: `writeExr` writes the image's first `channels` of R, G, B, and `loadExr` reads the file's leading R, RG or RGB
  planes, so every AOV round-trips bit for bit. A scalar AOV's EXR is now a single R plane, not RGBA broadcast
- perf: `HeadlessRenderer` accumulates each lane at its own channel count, and the C API output is a straight copy
- test: `every_aov_renders_finite` asserts each lane's stored channel count; `exr_write_keeps_the_declared_channels` and
  `hdr_image_rgb_expands_as_the_display_swizzle` are new; the display, filter, driver and EXR checks follow the new layout
- measured: viewer peak RSS **1870 -> 1191 MiB** (0.637x [0.637, 0.638]) at 2048x1152 over six AOV stages;
  `pass_accumulate_ms` 22.9 -> 14.0 (0.610x), `upload_ms` 5.22 -> 2.93 (0.564x), `pass_ms` 0.978x [0.957, 0.984]; ray
  counts identical. All 31 AOVs on cornell, macbeth and stump are byte-identical through the C API, 51 display PNGs are
  byte-identical, and 51 EXRs are equal plane for plane

## Texture pipeline: geometry-only glTF, per-slot channel count, shared decodes

The scene JSON is now the only texture source, each map stores only the channels its reader takes, and one decode serves every
binding. Measured against `main` (3b960ee) with `bench_compare run`, 11 interleaved rounds, `results/texture_pipeline`.

- refactor: `Material::aoTexture` removed: AO is ray-traced, and the map was loaded, stored and read by nothing
- refactor: `loadGltf` reads geometry only. The core and `extras` texture slots, the hand-parsed `extras` JSON scan,
  `textureDir` and `model.texturePath` are gone; every instance starts at `makeDefaultMaterial()`.
  `textureOverrides`/`applyTextureOverrides` become `textures`/`bindSceneTextures`, since nothing is overridden any more
- fix: `loadSceneConfig` rejects unknown top-level, `model` and `environment` keys. A retired `textureOverrides` or
  `texturePath` used to load as its default, rendering the scene untextured with no diagnostic
- feat: `stump.json` restores the 4K stump (rotation and material from the retired `tree.json`) through `textures`; its AO map
  has no slot and is not bound. Through the new binding it renders byte-identical to the old glTF-bound path at both depths
- perf: `ImageTexture::channels` (`kScalarChannels` 1, `kRgbChannels` 3), fixed per slot by `makeDefaultMaterial()`: R for
  roughness and bump, RGB for colour, normal, specular and the environment, never alpha, which nothing read. `texel()` and
  `sampleBilinear` return `vec3` and dispatch on type and count, so the stride is compile-time and a scalar map filters one lane
- fix: `readExrChannels` rejects a file lacking a channel its slot reads. OpenEXR zero-fills an absent slice, so an RGB slot
  bound to an R- or Y-only EXR rendered black silently; `loadExr` now also requires R, G and B
- fix: `bindSceneTextures` copied each decode into every slot binding it while its cache was alive, holding every bound map twice
  at peak and once per primitive on a multi-primitive node. Slots are `TextureHandle` (`shared_ptr<const ImageTexture>`), one
  decode per (file, channel count); sampling dereferences the handle, never copies it
- measured: stump peak RSS **1366 -> 923 MB** at `textureBitDepth 16` (0.676x [0.674, 0.678]) and **2186 -> 1303 MB** at 32
  (0.596x [0.596, 0.597]); analytic map savings 436 and 872 MB. Before the shared-decode fix the 16-bit cut was only 73 MB
  (0.947x), which is how the copy was found. `pass_ms` 0.979x [0.944, 0.992] at 32, unresolved at 16 (0.992x [0.875, 1.036]).
  cornell and macbeth: `pass_ms` unresolved (0.995x, 1.000x), output CRC identical, RSS -4.2 MB (the HDRI's dropped alpha)
- image: cornell and macbeth byte-identical in Beauty, Albedo, Normal and Roughness at both depths; stump Albedo byte-identical.
  Stump scalar lookups round once where the vec4 path rounded twice: clang fuses the one-lane lerp under `-ffp-contract=on`
  and not glm's vector `mix`, shown by an unfused scalar lerp restoring byte-identity. Roughness within 1 ulp (rel 1.4e-7),
  Normal within 2.4e-7, Beauty RMSE 7.7e-7 (PSNR 113 dB) with mean signed difference -1.5e-9, path decorrelation without bias
- test: `image_texture_channel_selection` (R alone at one float per texel, exact RGB at three, an R-only file accepted by a
  scalar slot and rejected by an RGB slot and by `loadExr`); `image_texture_bilinear_half_bound` at both channel counts;
  `texture_binding_resolution` gains a file shared by an RGB and a scalar slot and one decode shared by two primitives;
  `scene_config_accepts_the_shipped_scenes` covers all three scenes, `scene_config_rejects_malformed_input` the retired keys and
  an environment typo. Mutation-checked: dropping the missing-channel rejection or keying the cache by path alone each fails its
  check. `ctest` 184/184

## Constant shading model and a spectral ColorChecker for pixel measurement

A surface that returns its authored value untouched, and a chart whose authored values are the measured ones, so rendered pixels can be
checked against a reference. `macbeth.json` places a clay ColorChecker (`grid00`) beside a constant one (`grid01`).

- feat: `shadingModel` (`"standard"` default, `"constant"`) in material JSON, carried by `MaterialConfig` and `PathTraceSettings`.
  A constant hit emits `resolveBaseColor` (texture × `diffuseColour` × COLOR_0) two-sided and ends the path, before the depth cap
  like an emitter. It is not in `LightSet`, so the BSDF-sampled hit takes MIS weight 1 and NEE never double-counts it.
  `constant.json` admits only `diffuseColour` and rejects any BSDF key; an unknown model is rejected
- feat: `tools/colorchecker_texture` writes `assets/textures/macbeth.exr` (6x4, linear Rec.709) from BabelColor's 30-chart
  average reflectances (`tools/data/colorchecker_babelcolor_average.csv`). Each spectrum is resampled to the CIE 1 nm grid with
  the CIE 167:2005 Sprague interpolant, extended by nearest value beyond 380-730 nm, and integrated with `cie::reflectanceToRec709`
  under D65, so no chromatic adaptation is involved. Cyan lies outside Rec.709 and keeps its negative red (-0.0286). Against
  BabelColor's published D50 chromaticities, Bradford-adapted to D65, the neutrals agree within ΔE2000 0.35 and every patch
  within 1.38
- feat: `textureOverrides` in the scene JSON binds EXRs to a glTF node's `Material` slots (`applyTextureOverrides`), replacing
  the glTF's own, so Houdini's export-specific image paths never reach the binding. `macbeth.json` binds the chart that way and
  its glTF is geometry only (tangents from `gltf_tangent`, UVs on [0,1]). Rendering is identical to the glTF-bound chart (RMSE 0)
- test: `constant_surface_returns_texel` (texel × tint exact under black and bright skies), `constant_surface_emits_indirect`
  (furnace: a floor under a constant ceiling matches the open sky; it fails when indirect constant hits are dropped),
  `material_binding_resolution` at 14 fields, `texture_binding_resolution` (slots, multi-primitive nodes, atomic rejection), three
  `io_validate` material rows and one scene row, and in `colour_validate`
  `sprague_reproduces_quartics`, `colorchecker_table_rejects_invalid_input`, and `colorchecker_matches_colour_science`, which
  agrees within 1e-12 with colour-science 0.4.6 fed the same CIE CSVs. `ctest` 182/182
- measured: on `grid01`, beauty equals the binary16-stored texel within 3.2e-10 at every patch centre, and the Albedo AOV within
  3.1e-9. At `textureBitDepth: 16` the remaining gap to the float reference is texture storage alone, at most 2.3e-4. cornell's
  Beauty is byte-identical to before (linear RMSE 0, 64 passes)
- perf: cornell `pass_ms` B/A is 0.992 [0.919, 1.331] over 11 paired rounds, not resolved: the constant branch costs one
  settings load per hit

## LoG, Retinex and CLAHE removed, and the octave pyramid with them

The three AOVs went beyond what the project needs. With them gone, DoG was the octave pyramid's only consumer and read just its
two base-grid levels, so the pyramid goes too. 31 AOVs remain; 7 are Beauty filters.

- refactor: `AovId::LoG`, `AovId::Retinex` and `AovId::CLAHE` are removed from the enum, the HUD, `pt_aov_id`, Python and
  `--aov`. Every later id shifts down. `pathtracer_c.h` documents ids as discovered at runtime, so callers that resolve by
  name are unaffected and `PT_ABI_VERSION` stays 2, as it did when LoG was added
- refactor: `scale_space.h` keeps `innerScaleVariance`, `discreteGaussianKernel` and `diffuse`. `buildOctavePyramid`,
  `ScaleSpaceLevel`, `expandToBase`, `addExpanded`, `octaveMean`, `laplacian5` and `decimationVariance` are gone. DoG is now
  two diffusions, `t0` and then `3·t0`, which by the semi-group is exactly the cascade's second level, empty where that
  step's kernel does not fit the frame. `Camera::pixelsPerRadian` and `FilterInput::pixelsPerRadian` existed only for CLAHE
  and are removed, along with the `Camera` argument that `presentFrame`, `samplePixelProbe`, `resolveAovImage` and
  `ensureFilterImage` only forwarded
- perf: DoG `filter_ms` is 0.930x [0.919, 0.947] at 2048x1152, from no longer copying two pyramid levels. Beauty `pass_ms` is
  0.993x [0.981, 1.002], not resolved
- test: every remaining filter AOV and Beauty is byte-identical to before (linear RMSE 0, identical PNGs, cornell 640x360,
  32 passes). The checks for the removed operators and the pyramid are gone; the DoG checks stand alone, and
  `dog_is_empty_exactly_below_its_coarse_support` pins the empty-band boundary. `ctest` 177/177

## Review of the perceptual AOVs and the fisheye: Ward Larson CLAHE, Land's surround, scale interpolation, Rectilinear

A code review of the scale-space AOV and fisheye waves against the literature they cite. Five operators or bounds did not
match their references, and the fixes change pixels. Every change carries a validator and a paired measurement.

- fix: **CLAHE could expand contrast without limit; the AOV keeps its name, the operator is replaced.** Its ceiling, `1/K`
  over the `K` bins a tile occupied, capped a tile's slope at `B/K`: a flat tile inside a high-range frame stretched a tenth of
  a stop of noise across the whole range. Ward Larson 1997 measures the ceiling against the *display* range, which a
  range-preserving operator lacks. `CLAHE` (`AovId::CLAHE`, the same name for `pt_aov_id`, Python and `--aov`) is now Ward
  Larson's 1-degree foveal means, Freedman-Diaconis bins, the linear ceiling `f_b <= T·Δb/D` solved at its fixed point in
  closed form, mapped onto `D = log2(80/0.2)` stops (IEC 61966-2-1 sRGB reference medium, verified against the ICC registry).
  It is the identity where the occupied world range fits, as on the shipped 70 mm Cornell frame, and compresses under a
  fisheye (relMSE 0.92 against Beauty). 7.6x faster than the tiled CLAHE (7.3 vs 55 ms at 2048x1152)
- fix: the visual-angle anchor is `Camera::pixelsPerRadian = f·H/h`, the pitch on the optical axis, exact for both projections
  since `tan'(0) = theta_d'(0) = 1`. `height / verticalAngularExtentRadians()` counted the black rows outside a fisheye's image
  circle and gave the mean rather than the fixation pitch under the pinhole. The extent stays, as the HUD's FOV readout
- fix: **Retinex's surround is Land 1986's inverse square**, not the single Gaussian at the pyramid's coarsest level that was
  cited to him. `∫ G_t dt/t = 1/(π r²)` and the rungs are uniform in `ln t`, so the surround is the plain mean of every level.
  The old scale doubled whenever the render height crossed `2^k·(2r+1)`, so the interactive render scale previewed a different
  operator. `octaveMean` collapses the mean coarse to fine, exact because bilinear restricts to a nested 2x lattice, at ~4/3 of
  one base pass: Retinex costs 1.10x the old single-level surround [0.98, 1.17], unresolved
- feat: LoG refines its extremum with the parabola through the argmax rung and its neighbours in `log t` (Lowe 2004 §4). A
  blob midway between octave rungs read up to 11.1% low on the continuous Gaussian-blob model, `t·t0/(t+t0)²`, and now up to
  4.9%. It is never above the true peak (`log_interpolates_the_scale_peak_between_rungs`, against the exact discrete peak).
  Cost 1.11x [1.08, 1.17]
- refactor: `LensProjection::Spherical` is **`Rectilinear`**, in the enum, the HUD, `profile.json` (`"rectilinear"`),
  `PT_LENS_RECTILINEAR` and Python. Arnold's and RenderMan's spherical camera is lat-long, and the name would collide with a
  real one. The enum values are unchanged, so `PT_ABI_VERSION` stays 2. The JSON key and the macro name are a source-level break
- fix: `kannalaBrandtTheta`'s 30-iteration cap cited an all-bisection bound that was wrong twice (`ulp(pi)` is `2^-22`, and a
  relative stop has no fixed bisection count). The stop is now `4·eps·thetaMax`, so 21 halvings, and a Newton step that fails
  to halve the bracket forces a bisection next, so `2·21` iterations bound any lens. The fisheye ray drops a redundant normalize
- perf/refactor: DoG builds only the two rungs it reads (`buildOctavePyramid(..., maxLevels)`), 0.83x [0.82, 0.87]. The
  expansion mirrors past a level's last sample instead of clamping, folding only at that edge. `laplacian5` folds only at its
  edge columns. `bipolarDisplay` ranges a scalar AOV once. DoG is re-cited to Burt & Adelson's first Laplacian-pyramid band.
  The cone-space and discrete-Gaussian comments are corrected where they overclaimed (S per unit L+M, not luminance; affine
  exactness only for representable fields)
- test: `render_beauty` records `filter_ms` and takes `--fisheye FOCAL_MM`. New checks for the Ward Larson ceiling, identity
  and display span, the axis pitch, octave-mean exactness, pyramid truncation and LoG scale interpolation; `ctest` 195/195.
  Rectilinear Beauty, DoG and Colour Opponent are byte-identical to before. Fisheye Beauty moves by relMSE 1.4e-13 (4 mm) to
  1.4e-12 (8 mm) from the new stop rule

## A polynomial fisheye lens, and a projection the camera selects

The camera had exactly one projection and no lens model at all: `primaryRay` built `forward + ndc·halfExtent·basis`, and
`aperture`/`shutterSeconds`/`iso` reached `ev100()` and nothing else. `profile.json` and the HUD now select between that
rectilinear pinhole -- named **Spherical** throughout, in JSON, the enum and the dropdown, so there is one vocabulary --
and a **Fisheye Polynomial**.

- feat: `scene/lens.h` carries Kannala & Brandt 2006's `r(theta) = f·(theta + k1·theta³ + k2·theta⁵ + k3·theta⁷ + k4·theta⁹)`
  in the parameterisation OpenCV's `fisheye` module and COLMAP's `OPENCV_FISHEYE` report, so a measured lens's `k1..k4` drop
  in unscaled. `k = 0` *is* the equidistant family exactly. What transfers is the radial geometry alone: one `focalLengthMm`
  means `fx == fy`, and the image circle is centred on the sensor, so a calibration's principal point and pixel aspect do not
- feat: the inverse is per ray, not a fitted `theta(r)`: safeguarded Newton (Numerical Recipes §9.4 `rtsafe`) on the bracket
  `[0, thetaMax]`, seeded at the radius itself -- first-order exact, and bitwise exact for an equidistant lens, which returns
  on iteration zero. Both constants are derived rather than tuned: the 30-iteration cap is the all-bisection worst case to
  float resolution on the widest possible bracket (`thetaMax <= pi`, `ulp(pi) = 2^-21`, so `ceil(log2(pi/2^-21)) = 23`), and
  the stopping rule is a 4-ulp bracket, i.e. float precision with no scene-dependent term. Convergence is unconditional:
  Newton accelerates, and any step leaving the bracket bisects instead
- feat: `kannalaBrandtIsInvertible` *proves* `r'(theta) > 0` rather than sampling for it. `r'` in `u = theta²` is a quartic,
  so its coefficients go to the degree-4 Bernstein basis and the convex-hull property decides the sign, de Casteljau-subdividing
  only the undecided halves. It is conservative by construction -- a slope that merely grazes zero is rejected at depth 24,
  which is the right answer, because such a lens has no unique inverse there. Load, the C ABI and Python all refuse a lens it
  cannot prove, so `primaryRay` never meets a radius it cannot invert
- feat: samples outside the image circle return `std::nullopt` and splat an all-zero `TraceResult`. The filter weight still
  accumulates -- it belongs to the film sample's position, which exists whether or not the lens formed a ray -- so the pixel
  average stays a true average, `invWeight` cannot go infinite, and the circle's edge antialiases for free. A circular fisheye's
  corners are black in every lane, which is what the lens does
- feat: framing stays physical: `r_max = f·theta_d(thetaMax)`, the authored focal length against the gate, with **no fit mode**.
  An intermediate revision derived the focal length from the gate (`Circular`/`FullFrame`, `f = extent / theta_d(thetaMax)`) and
  was removed before landing: its defining property is that the focal length no longer does anything, and a mode that ignores
  the control a lens is authored by is a worse interface than none. The consequence to live with is that a long lens hides the
  projection -- at the shipped 70 mm a 180-degree lens puts a 110 mm circle behind a 13.4 mm gate, so the frame is the central
  11 degrees and the render is **visually indistinguishable** from the pinhole (relMSE 3.1%, corner ray 1.3% apart) -- so the
  fisheye is a short-focal-length instrument, as it is on a real camera
- feat: the HUD's Camera section gains the projection dropdown and prints the active projection's vertical field of view,
  `verticalAngularExtentRadians()` itself, so the readout cannot disagree with the render; the polynomial stays authored in
  `profile.json`, being measured data rather than slider material. `ViewInputState` tracks the projection, so switching retraces. The
  focal-length slider's floor drops 10 mm -> 6 mm: 10 mm was a spherical assumption sitting above the 8.68 mm threshold at
  which the circle enters the shipped gate, so the fisheye's own regime was unreachable from the HUD at any slider position.
  6 mm is a real circular-fisheye focal length (Nikon 6 mm f/2.8) and clears that threshold on 12 of the 15 shipped presets
- refactor: `maxThetaRadians(lens)` replaces four hand-written copies of `0.5F * glm::radians(maxFieldOfViewDegrees)` across
  the camera, the config parser and the C ABI, so the lens turns degrees into radians in exactly one place
- fix: the rasterizer's `projectToScreen` is a perspective divide with no fisheye equivalent, so the 14 scan-converted G-buffer
  AOVs are **rejected, never approximated**: `HeadlessRenderer::render` fails before it allocates, the GUI greys those entries
  out and falls back to Beauty, and startup refuses a profile whose `defaultAOV` or `-bench-aovs` schedule would park the
  driver on a G-buffer generation that can never arrive. `claheAov`'s `pixelsPerDegree` is the one filter that reads an angle,
  and `verticalFovRadians` (`2·atan(h/2f)`) is not a fisheye frame's extent: it keeps its exact body for `viewBasis`, and the
  new `verticalAngularExtentRadians()` -- the same expression under Spherical, so CLAHE is bit-identical -- feeds the filters
- feat: `PtCamera` gains `lens_projection`, `fisheye_coefficients[4]` and `fisheye_field_of_view_degrees`. It crosses by value,
  so this is a **breaking ABI layout change**: `PT_ABI_VERSION` (2; 1 is the pre-lens layout) and `pt_abi_version()` land with
  it, and the Python binding checks them at load, before any struct is passed. `toCamera` became validating, since an unknown
  projection or an unprovable polynomial has no defensible coercion
- test: `camera_validate`, 8 checks and the first suite the camera has had. The strongest is
  `fisheye_direction_inverts_the_forward_model`: the test projects a known direction *forward* through the model and requires
  `primaryRay` to return it, so the forward map lives in the test and the inverse in the library. The inverse is asserted
  against an independent bisection on the forward polynomial over the degree-9 Taylor coefficients of the equisolid,
  stereographic and orthographic families -- derived, not invented calibrations -- to a forward-error bound computed per case
  from the stopping bracket and the local slope, never a hand-picked tolerance. The monotonicity gate is tested against a slope
  whose root is known exactly by construction (`k1 = -1/(3u0)` gives `r' = 1 - u/u0`), plus one-sided soundness over 2048 random
  coefficient sets. `fisheye_theta_distribution_matches_the_area_jacobian` catches what no per-ray check can: uniform sensor
  area must land on `theta` with CDF `(r(theta)/r(thetaMax))²`, by chi-square at the family-wise alpha.
  `fisheye_vertical_extent_saturates_at_the_image_circle` pins what CLAHE reads: the extent saturates at the full field of view
  once the circle falls inside the gate, is the top edge's angle by independent bisection when it does not, and is bitwise
  `verticalFovRadians()` under Spherical. `io_validate` and `api_validate` cover the config boundary and the G-buffer
  rejection; `ctest` 191/191
- perf: not measurable on the spherical path, which is the same expression text behind one branch on a per-pass constant:
  `bench_compare` puts the paired `pass_ms` ratio at **1.004 [0.981, 1.022]** over 7 rounds, unresolved at the 95% interval,
  with identical ray counts every round, and the Beauty EXR is byte-identical to a binary built from `7ea662d` (crc32
  `fccef5e8`). The fisheye's cost is three Horner pairs on primary rays only, 3-4 iterations from the first-order seed

## A log display for SNR, and one shared display decision

`SNR` previewed as a solid white frame. `aovCarriesRadiance` correctly reports false for it, which pins the display at unity
gain, and the AOV is unbounded — the HUD probe read 22.216 on a Cornell box — so every lit texel encoded to 255. The preview
carried exactly one bit: is this texel above its own noise floor.

- feat: `SNR` previews in decibels, `20·log10(SNR)`, the factor being 20 rather than 10 because `mu/SE` is a ratio of
  like-dimensioned amplitudes whose power ratio is its square (EMVA 1288). **Both anchors of the window are definitional.**
  The floor is 0 dB, the ratio 1, where a texel's value equals its own uncertainty. The ceiling is `10·log10(n)` dB, the ratio
  `sqrt(n)`, which since `SNR = sqrt(n)·(mu/sigma)` is the SNR of a texel at unit per-sample coefficient of variation. The
  decibel factor cancels, so the implementation is `2·ln(SNR)/ln(n)`. Measured on a Cornell box at 64 passes: **1 distinct
  display level before, 211 after**. The window saturates the converged majority by design — 78% of texels at 64 passes,
  67% at 1024 — and resolves the noisy tail, which is what one selects `SNR` to find. Nothing lands on the floor
- fix: `render_beauty --aov "Bounce Count"` wrote a raw scalar PNG while the viewer showed Turbo false colour, contradicting
  the pipeline's own "a PNG and the viewer agree by construction". **This is the one visible output change**: Bounce Count
  PNGs are now false-coloured, matching the viewer byte for byte
- refactor: `aovDisplay` is the one function making every display decision that has to read an AOV's values — the bipolar
  per-lane range, `Depth`'s auto-range, and the two nonlinear pre-maps a `vec3` uniform pair cannot express. It replaces two
  hand-synchronised copies of that decision in `main.cpp` and `render_beauty.cpp`, deletes `AppResources::pathTraceDisplayedDepthMax`
  and `render_beauty`'s now-dead `maxChannel`, and returns an empty buffer where the source texels pass through unchanged, so
  the common path copies nothing. The photographic exposure stays with the caller: the slider moves without rebuilding the texture
- fix: `aovTakesDisplayExposure` states once that an auto-ranged AOV does not also take the exposure. `DoG` and `LoG` are both
  bipolar *and* degree one in radiance, so collapsing the old `if/else` into two independent arms double-ranged them; caught by
  a byte-identity check against the previous build, and now gated for all 34 AOVs
- test: 8 new checks in `api_validate` — the log window's two anchors, its logarithmicity (`grey(r²) == 2·grey(r)`, exact in
  float), its absence below two passes, Bounce Count against the colormap, Depth's migrated auto-range, that a pre-map applies
  to exactly two of the 34 AOVs, and that no AOV both auto-ranges and takes the exposure. `ctest` 181/181
- perf: not measurable. The pre-map is one pass over the frame with one `log` per texel, at upload rather than per frame.
  Interleaved A/B over three runs put `SNR` at +1.5% and the `Beauty` control, which never touches this path, at +0.3%,
  both inside a spread larger than either

Unchanged: every EXR, the HUD probe, the C ABI (`pt_aov_count()` 34, `pt_display_encode` untouched) and the PNGs of `Beauty`,
`Depth`, `LoG`, `DoG`, `Colour Opponent`, `Sobel`, `Normal` and `Luminance`, all verified byte-identical against binaries built
from the previous commit.

## Colour Opponent, and a per-lane range for the signed preview

`Opponent` previewed as a near-uniform magenta wash with no visible red/green separation on a Cornell box. The filter was exact;
the preview was pooling two axes that have no common unit, and painting a lane the AOV never defined.

- fix: the bipolar auto-range is now per lane. `Opponent`'s two axes are different physical quantities — `l - l_white` is a
  dimensionless cone fraction spanning `[-0.150, +0.170]` over the whole Rec.709 gamut, `s - s_white` is S excitation per unit
  luminance and reaches `+13.087` on the blue primary, a **77× disparity** — and one shared range let the second own the display.
  Measured on a Cornell box, the `l` axis held **20 of 256 levels**; it now holds 167, and the red and green walls are visible
- fix: a lane the AOV does not define renders black, not mid-grey. `Opponent`'s third lane is a structural zero, never a
  measurement, and sending it to mid-grey put a constant blue floor over every frame — the magenta. `UV`, also two-channel,
  already read black. A scalar AOV is exempt: `writeScalar` broadcasts it across all three lanes, so all three are ranged
- refactor: the display path is one affine map, `gain ⊙ value + offset`, with both terms per channel. `uExposure` and
  `uDisplayOffset` become `vec3`, `setExposureEv`/`setDisplayOffset` collapse into `setDisplayAffine`, and
  `bipolarDisplayRange`/`bipolarDisplayExposureEv` collapse into `bipolarDisplay`. Radiance is the case where the gain is a
  scalar exposure broadcast to three lanes and the offset is zero; `pow(2, ev)` moves out of the per-frame `bind()`
- feat: `Opponent` is renamed **`Colour Opponent`** — `AovId::ColourOpponent`, `colourOpponentAov`. `normalizeAovName` ignores
  case and separators, so `colour-opponent`, `colour_opponent` and `colourOpponent` all resolve
- **breaking**: the bare name `Opponent` no longer resolves. `aovIdFromName("Opponent")` returns `AovId::Count` and
  `pt_aov_name` reports `"Colour Opponent"`. No alias is kept, so a stale caller fails loudly rather than silently
- fix: the channel-isolation view now runs *after* the affine map, not before it. A per-lane gain would otherwise multiply the
  broadcast lane by three different gains and tint the grey, and an isolated view of an undefined lane would read as black.
  With a scalar gain and a zero offset the two orders agree exactly, so no non-bipolar preview moves
- test: `bipolar_lanes_range_independently` and `bipolar_absent_lanes_render_black` are new; the two existing bipolar checks move
  to the collapsed entry point. Every non-bipolar preview is byte-identical against binaries built from the previous commit
- fix: `results/log-signed-bipolar/after-DoG.png` was stale — captured mid-wave, before the Cramér range landed, and it does not
  reproduce from its own commit. A fresh render of that commit is byte-identical to this one's, so `DoG`'s preview is unchanged

## LoG as a signed operator, and a bipolar preview for signed AOVs

`LoG` shipped as a three-channel scale-selection blob detector — magnitude, the winning scale's frequency in cycles/degree, and
its polarity. Two of those three channels were wrong to report, and the packing made the AOV unreadable as an image.

- fix: the scale channel measured the wrong quantity. With γ = 1 the response of a step edge of height `h` at its peak offset is
  `0.242 h t^(γ-1)`, identically independent of `t`, so the scale-normalised family is flat on edge structure and the argmax is
  undetermined; at a fixed offset `x` from an edge the response is stationary at `t ≈ x²`, making the channel a map of squared
  distance to the nearest edge. γ = 1 is the right exponent for blobs and the wrong one for edges, where Lindeberg 1998 gives
  γ = 1/2, and a rendered image is mostly edges. Measured on a Cornell box the channel took **4 distinct values, with 63.5% of
  the frame on the single coarsest rung** — the posterised bands the AOV showed on screen
- fix: the polarity channel was exactly `-sign` of the signed response and carried nothing. It read -1 on 52.61% of texels,
  matching the new signed channel's negative fraction to the digit
- feat: `LoG` is now one signed channel, the scale-normalised extremum over the octave ladder. Signed because the zero crossings
  are the edges (Marr & Hildreth 1980), which a magnitude erases. 3 channels to 1
- fix: the sign is negated once, so positive means bright-on-dark and agrees with `DoG`. Lowe 2004 §3 gives
  `G(kσ) - G(σ) ≈ (k-1)σ²∇²G`, so `fine - coarse` is *minus* the Laplacian: the two operators of one family previously read
  opposite polarity on the same feature
- fix: `aovCarriesRadiance` is now true for `LoG`. `t·laplacian5` is linear in luminance, hence positively homogeneous of
  degree one, exactly as Sobel, Gabor and DoG already are
- feat: `aovIsBipolar` routes `DoG`, `LoG` and `Opponent` through an affine display map, `0.5 + value/(2·range)`, putting zero
  on mid-grey instead of clipping the negative half to black. The offset is a uniform beside the exposure multiply, shared by
  the three display shaders and by the CPU encode, so `render_beauty`'s PNG and the viewer agree by construction
- feat: the preview auto-ranges to `min(peak, σ·sqrt(2 ln n))` rather than the peak, σ being the RMS about zero and the cap the
  concentration point of the maximum of `n` standard normals (Cramér 1946). A rendered signed response is heavy-tailed — LoG's
  maximum sits 32x above its own 99th percentile — so a peak scan left 97% of the display range unused; the cap recovers about
  9x of usable contrast, and a field with no tail still keeps its true peak and never clips
- note: preview only. `aovCarriesRadiance` still describes the value, and the EXR, the HUD probe and the C ABI all read the raw
  signed float. `DoG`'s preview no longer tracks the photographic exposure, being auto-ranged and so exposure-invariant
- note: `LoG`'s cost is unchanged. Dropping a full-frame array and one scatter write per ladder rung is real but sits below this
  machine's noise floor — interleaved A/B medians ran 60-74 ms either side at 2048x1152, with no resolvable difference
- test: `log_and_dog_agree_on_polarity` replaces the polarity check, plus `log_is_homogeneous_of_degree_one`,
  `log_responds_where_dog_cannot`, `bipolar_aovs_produce_both_signs`, `bipolar_range_caps_a_lone_outlier` and
  `bipolar_display_maps_the_range_to_the_unit_interval`. 170 checks, each new gate mutation-tested in both directions

## The Gabor AOV re-derived as a 2-D Morlet bank

The last authored constants in the AOV filters. `buildGaborKernel` carried `sigma = 1.4`, `lambda = 4.0`, `gamma = 0.5`, four
hand-picked orientations and a 5x5 support; all of them now follow from the shared scale space and from one another. This
changes the AOV's output and its unit, which is why it was held back as its own change with a before/after capture.

- feat: the bank is a 2-D Morlet wavelet (Morlet 1982; Antoine & Murenzi 1996) — an isotropic Gaussian envelope times a plane
  wave, minus its own mean. `sigma^2 = innerScaleVariance()`, the finest scale the grid resolves and the rung the octave
  ladder starts from, so the bank tiles the frequency plane the pyramid already does
- feat: the carrier follows from the bandwidth alone (Petkov 1995 eq. 4), and the bandwidth is the ladder's own octave
  spacing: `sigma*omega0 = 2 sqrt(ln2/2) * 3 = 3.532`, `lambda = 5.130 px`, 2.57x inside the grid Nyquist — asserted, not
  assumed
- feat: **the orientation count is not a choice either.** `sigma*omega0` depends only on the bandwidth, so the angular
  half-response width does too: 38.4 degrees, so covering a half turn takes `ceil(180/38.4) = 5` orientations. A half turn
  suffices because the bank is complex and opposite directions are conjugates
- feat: the response is the quadrature magnitude, so the old bank's odd-versus-even carrier decision disappears with it
- feat: `(f * G_theta) = exp(i omega0 x.u) [(f exp(-i omega0 x.u)) * g_sigma]` exactly, so each orientation is two separable
  `diffuse` calls rather than a rotated 2-D kernel — 11 separable passes against the 6125 taps a texel a direct 35x35 bank at
  this scale would need. The closing remodulation is a rotation the magnitude discards, so it is not computed at all
- feat: admissibility is exact **at the border too**. Instead of the ideal `exp(-sigma^2 omega0^2 / 2)`, the subtracted term
  is the blurred plane wave under the same mirror the image gets — what a constant field actually produces at that texel.
  That field is separable, so it is one width-long and one height-long 1-D blur and costs nothing.
  Zero to rounding rather than bitwise -- the image takes one fused 2-D blur and the correction a product of two 1-D ones, and
  float multiplication does not distribute -- measured at 1.0 to 1.4 float epsilons of the field over four decades of field.
  `filters_are_zero_on_a_constant_field` now asserts it on a 24x16 frame, smaller than the kernel, where every texel is a
  border texel; the old bank only cancelled to 1e-6 and only away from the edge
- perf: 16.0 -> 100.0 ms at 2048x1152, 8 threads. The envelope is 2.06x wider in each axis and there is one more orientation,
  and the operator is exact rather than a 5x5 approximation of it; it now sits between CLAHE (58) and Retinex (119)
- fix: **the old bank was largely a noise detector.** Its passband sat near pixel Nyquist, so on an unconverged frame most of
  what it reported was Monte Carlo noise, not scene structure — visible directly in `results/wave-morlet-gabor/`. The unit
  changed with it: the envelope has unit DC gain, so the magnitude is the oriented component's amplitude in radiance units,
  comparable against Luminance, where the old unnormalised weights ran about 43x higher with no unit attached
- test: `gabor_bank_rejects_dc` is replaced by `morlet_bank_is_admissible_and_covers_every_orientation`, which reads the
  bandwidth relation back out of the shipped carrier over four envelope variances, checks the carrier is inside the grid
  Nyquist, and asserts the orientation count is both sufficient and minimal -- one fewer must fail to cover. The first draft
  of that last assertion compared `ceil(x)` against a recomputed `x` and was a tautology; review caught it
- test: `the_blurred_plane_wave_separates_into_its_two_axes` gates the separable implementation directly -- the product of a
  width-long and a height-long 1-D blur must equal the full 2-D blur of the plane wave under the same mirror, over every
  orientation at five shapes including the degenerate `1x40` and `40x1`

## Monte Carlo noise: the Welford second moment and the SNR AOV

The last AOV of the perceptual wave, and the only one needing a new lane in `PathTraceResult` rather than a new read of Beauty.
33 AOVs become 34, nine filters become ten. It also unblocks the parked variance-driven adaptive-sampling item, which needed
exactly this measurement.

- feat: `PathTraceResult::beautyLuminanceM2` -- the Welford second moment of each texel's per-pass Rec.709 luminance. One
  float per texel, not an eleventh image: an SNR needs no chromaticity, and luminance M2 computed on the per-pass luminance
  is exact for luminance, where three per-channel moments would need the inter-channel covariance shared paths induce
- feat: the recurrence lives in `accumulateMean`, which already holds this pass's radiance in the destination and the previous
  mean in the source -- precisely Welford's two operands, so the moment is one extra pass over a buffer already being streamed.
  `renderPathTraced` is untouched, so the hot path pays nothing. Welford 1962 in West 1979's `(x - m_prev)(x - m_new)` form,
  the same recurrence and the same `invN` the ten RGB lanes use
- feat: `AovId::SNR` (1 ch) reports `mu / sqrt(M2 / (n(n-1)))` -- the standard error **of the mean**, because `beauty` is a
  mean and the question is how well its published value is known, not how noisy one sample was. Zero below two passes, where
  a variance is undefined, and zero on a texel every pass agreed on, rather than an infinity. Linear, not dB
- feat: single-image spatial estimators are recorded as evaluated and rejected. Immerkaer 1996 and Donoho & Johnstone 1994
  both estimate one global sigma under additive white Gaussian noise; Monte Carlo render noise is heteroscedastic by orders of
  magnitude, signal-dependent, and spatially correlated through NEE, environment importance sampling and the shared
  Owen-scrambled Sobol sequence. Both would also report a tessellated silhouette as noise
- feat: the headless path keeps its naive summation and single divide, with the moment carried *beside* the sum and the
  previous mean read back out of it. Verified: `render_beauty --compare-exr` against the pre-change Beauty is RMSE 0
- perf: peak RSS 1296.3 -> 1327.8 MiB at 2048x1152 on a matched three-stage benchmark, **+31.5 MiB, +2.43%**. The SNR filter
  itself is 6.95 ms, one streaming pass with a square root
- test: `running_m2_matches_batch_variance` gates the driver's recurrence against the two-pass sum of squared deviations in
  double, under a forward-error bound built the same way `running_mean_matches_batch_mean` builds its own -- the float mean's
  per-lane error carried through the luminance dot and into both Welford operands. Mutation-tested: dropping the mean update
  and dropping the carried moment are both caught. It also asserts the moment is somewhere non-zero, so a frame the passes
  agreed on cannot satisfy the bound vacuously
- fix: `samples * (samples - 1)` was evaluated in `int` before widening, which is signed overflow past 46341 passes -- an
  uncapped interactive accumulation or a large headless request reaches it, and the result is a negative degrees-of-freedom
  count, a NaN standard error and a silently black AOV. Found by review, confirmed undefined by UBSan on a reduction.
  The sweep now runs to 65536 passes, though it cannot gate the overflow itself: the count is a compile-time constant there
  and clang folds the product, so `PATHTRACER_SANITIZE` is what covers that class
- test: 3 SNR checks in `filter_validate`, all closed form and exact -- the alternating sequence `mu +/- delta` whose moment is
  `n delta^2`, the `sigma/sqrt(n)` fall of the standard error, and the four undefined cases reading zero

## Colour opponency, retinex and CLAHE: the three perceptual AOVs

The second wave on `scale_space.h`. All three are observer models rather than image-space derivative operators, so they get
their own `// Perceptual.` block in the enum, between Utility and Material. 30 AOVs become 33, six filters become nine. Three
of the designs recorded in [ROADMAP](ROADMAP.md) did not survive contact and were replaced; each replacement is stated below
with what was wrong, because the rejected reasoning is the part worth keeping.

- feat: `include/pathtracer/scene/cone_space.{h,cpp}` (new) -- linear Rec.709 to cone excitations to the two cardinal
  chromatic axes, every link exact on data this repo already holds. `AovId::Opponent` reports `(l - l_white, s - s_white)`,
  the L-versus-M and S-versus-(L+M) displacements from Rec.709 white
- fix: **Smith & Pokorny 1975 was the planned basis and cannot be used.** The plan's justification was that it is *by
  definition* an exact linear transform of the CIE 1931 2-degree CMFs; it is not -- it is defined on the **Judd-Vos modified**
  CMFs. Nor is that bridgeable: Rec.709's primaries are specified as CIE 1931 chromaticities, not spectra, so they have no
  Judd-Vos tristimulus values at all, and no exact route from this codebase's RGB to any Judd-Vos-based cone space exists.
  Hunt-Pointer-Estevez (Estevez 1979; Hunt 1998 App. 1) is stated **as** a 3x3 on CIE 1931 XYZ, so it is the only exact choice.
  The cost is stated rather than hidden: `L + M` is not `V(lambda)` -- the Judd modification exists precisely to make it so --
  and the `(l, s)` plane is therefore not strictly isoluminant. The `s` axis is rescaled so one unit is one S excitation per
  unit luminance at the achromatic point, recovering MacLeod & Boynton's unit as far as this observer permits
- feat: each opponent numerator's three RGB coefficients sum to zero, and a zero-sum row is `r.x (R-G) + r.z (B-G)`
  identically, so the basis stores those two coefficients. An achromatic texel then reads **exactly** zero at any intensity,
  not within a rounding of it -- the same structural device as the convolution's centre-relative form
- feat: `AovId::Retinex` is Land 1986's Gaussian-surround formulation (equivalently Stockham 1972 homomorphic filtering), per
  channel, with the surround at the pyramid's coarsest level -- its endpoint, so no extent is chosen. Jobson et al. 1997 MSR
  (`G = 192`, `b = -30`, display-referred), Land & McCann 1971 (path-ensemble dependent, no closed-form invariant) and
  Horn 1974 / Blake 1985 (a gradient threshold with no defensible derivation) are recorded as rejected
- feat: zero radiance is exact, not floored. Validity is per channel, and the surround is Knutsson & Westin 1993 normalised
  convolution: cascade `l*m` and `m`, divide at the end. An everywhere-valid channel skips the mask cascade, bit-identically,
  because the cascade returns a constant field exactly
- feat: `AovId::CLAHE` is Zuiderveld 1994's structure on log2 luminance, with all four normally-authored parameters derived:
  bin width from Freedman & Diaconis 1981 (Scott 1979 where the quartiles coincide), the clip ceiling from Ward Larson et al.
  1997, the tile grid from one degree of visual angle through the camera's vertical FOV, and the blend written as nested lerps
  so the four weights are a partition of unity by construction. Output is chromaticity-preserving
- fix: **Ward Larson's ceiling against the global range makes CLAHE the identity, exactly.** The ceiling is then `1/B`, and
  since the clipped densities must still total 1 over `B` bins, the only feasible distribution is the uniform one. This was
  not hypothetical -- the first implementation returned Beauty bit for bit through the C ABI. The ceiling is now measured
  against the support each tile actually occupies: `1/K` with `K <= B`, capping the slope at `B/K`, and reducing to the
  identity exactly when a tile already spans the whole range. It is the same criterion read adaptively, and the only
  non-vacuous reading for an operator that preserves the overall range
- fix: **`overRangeBin()` is not a log2 lattice** and the plan's reuse of it for CLAHE's binning is rejected. It is uniform in
  the float's bit pattern, so the true width of a bin varies by a factor of two within a binade (`d log2(1+f)/df` runs 1.443
  to 0.721), which would make the uniform reference density the ceiling compares against wrong by up to 2x
- fix: **the plan's tile-grid derivation degenerates.** Sizing tiles from histogram adequacy alone fixes only the product of
  bin count and tile area, and at any ordinary resolution drives the grid to a single tile -- global equalisation, not
  adaptive. The grid is now anchored in degrees, so the tile count is the FOV in degrees and the operator's angular extent is
  invariant to render scale, the same anchoring LoG's cycles-per-degree channel uses
- fix: both operators now work in a **relative** log, `log2(L / L_min)` as an exact integer exponent difference plus a mantissa
  term. With an absolute `log2`, `(e + k) + log2(m)` and `(e + log2(m)) + k` round differently, so a power-of-two gain on the
  frame perturbed every bin index and CLAHE was not exactly homogeneous. The relative form cancels the gain bitwise, which is
  what makes the two invariance checks exact rather than toleranced
- refactor: `pathtracer_cie` (new OBJECT library) holds `cie.cpp` and `cone_space.cpp`, deliberately without `-march=native`
  or IPO so `albedo_table`'s committed output still reproduces exactly. `pathtracer_core` and `metal_fit` both consume its
  objects and `metal_fit_core` no longer carries `cie.cpp`, which is what keeps `colour_validate` free of duplicate symbols
- refactor: the planned `diffuseNormalised` entry point was dropped. Normalised convolution at the coarsest scale is two calls
  to `buildOctavePyramid` and a divide, so a function with one caller composing two existing ones earns nothing
- fix: the tile grid is capped at one tile per pixel. Below one pixel per degree the pitch falls under a sample, a tile row can hold
  no rows, and its scatter then runs on into the next row's pixels. Found by review; it cannot change a pixel, because that is
  exactly the regime where the one-sample-per-tile bin cap makes the transfer linear, so the fix is to the structure's own
  contiguity invariant and to an O(height x tiles) blow-up in the scatter, not to an output
- test: 13 checks in `filter_validate` and 2 in `colour_validate`. Exactly-zero or bit-identical: opponency on any achromatic
  texel at any intensity, opponency under a gain, retinex on a uniform field, retinex against normalised convolution spelled
  out independently against the facility, retinex under a gain, CLAHE on a uniform field, CLAHE's homogeneity under a gain,
  CLAHE's blend mapping equal radiances equally, and the cascade returning a constant level intact. `api_validate`'s
  `aov_filter_dispatch_is_total` picked the three up with no change, at 45 assertions. Two checks carry explicit anti-vacuity
  assertions -- that dropping retinex's mask changes the result, and that CLAHE is not the identity where its homogeneity is
  asserted -- because the first draft of each would have passed with the feature removed
- docs: PIPELINE gains a Perceptual AOV table and three derivation sections (Cone space, Retinex, CLAHE) with 11 references;
  ROADMAP retires the shipped item and replaces it with the measured-observer gap that only a spectral path can close

## A shared discrete scale space, the Beauty filters unified on the CPU, and the DoG/LoG AOVs

Seven perceptual AOVs were asked for. Two structural problems stood in front of all of them: the four Beauty filters existed
twice, once in `debug/aov_filters.cpp` for headless and once in GLSL for the viewer, and no Gaussian scale space existed
anywhere in the repo. CLAHE and Retinex are multi-scale with global state, which a single-pass fragment shader cannot express,
so unifying on the CPU was a precondition rather than a cleanup. This entry is the foundation plus the two AOVs that need
nothing else; the rest are tracked in [ROADMAP](ROADMAP.md).

- feat: `include/pathtracer/debug/scale_space.{h,cpp}` (new) -- Lindeberg's **discrete** Gaussian `T(n;t) = e^-t I_n(t)`
  (Lindeberg 1990), not a sampled continuous one. It is the unique kernel satisfying the discrete scale-space axioms, and the
  exactness is the point: unit mass by `sum I_n(t) = e^t`, variance exactly `t`, an exact semi-group, the transfer function
  exactly `exp(-t(1-cos w))`, and `dL/dt = laplacian5(L)/2`, which makes `laplacian5` of a level *be* its discrete LoG rather
  than approximate it. Every one of those is a validator, and each holds to float rounding rather than to a tolerance
- feat: `std::cyl_bessel_i` does not exist on libc++, so `I_n(t)` comes from the ascending series, whose terms are all positive
  and therefore free of cancellation. Magnitude is carried in the exponent and the series summed relative to its own first term,
  so neither a large order nor a large `t` overflows or underflows. Miller's downward recurrence was rejected: its starting
  order is a rule of thumb, which is the authored constant this codebase's numerics avoid
- feat: **every threshold in the file is float32's unit roundoff, 2^-24, and nothing else.** Truncation emits taps until the
  discarded tail mass drops below it (~5.4 sigma, wider than the customary 4). The inner scale is where the transfer at the grid
  Nyquist reaches it, `t = 12 ln2`. The decimation scale is the same criterion at the halved Nyquist, `t = 24 ln2`, exactly
  twice the inner scale because `(1-cos pi) = 2(1-cos pi/2)`. That last figure lands within 15% of SIFT's empirically chosen
  `sigma_0 = 1.6` per octave -- corroboration from the representation rather than from repeatability experiments
- feat: the cascade is octave-spaced and decimated (Burt & Adelson 1983's structure, Lindeberg's kernels; the binomial kernel is
  only approximately Gaussian and its variance is not `t`). One octave per level is not tuning: successive-octave DoG has a
  ~1.2-octave bandwidth matching measured human spatial-frequency channels (Wilson & Bergen 1979), successive octave DoGs sum to
  `(1 - lowpass)` exactly so the decomposition is complete, and it is what permits decimation at all. The ladder ends where a
  step's kernel support outgrows its own plane -- such a level reports the mirrored boundary at every sample, not the image --
  giving 6 levels at 2048x1152, sigma 2.88 to 92.3 base pixels, with no authored cutoff
- feat: convolution is **centre-relative**, `out = c + sum_n w_n ((l-c) + (r-c))`. On an affine field every tap pair cancels
  bit-exactly, so diffusion reproduces a ramp identically and DC gain is exactly 1 structurally rather than by normalisation; it
  also cuts cancellation in DoG, which differences two nearly equal blurs. The boundary mirrors about the edge sample without
  repeating it, so every tap lands on real data and the finite operator stays diagonal in the cosine basis
- feat: recursive IIR Gaussians (Deriche 1993; Young & van Vliet 1995; Alvarez & Mazorra 1994; Triggs & Sdika 2006) surveyed and
  rejected on **correctness**, not cost: an approximate semi-group, a boundary initialisation that is itself an approximation,
  and an asymmetric kernel that does not annihilate affine fields exactly. Their O(1)/px advantage evaporates once decimation
  makes the FIR path O(N) anyway
- feat: `AovId::DoG` (1 channel) -- the signed difference of the two finest octaves (Marr & Hildreth 1980), the retinal
  centre-surround band next to pixel Nyquist. Signed, not a magnitude: on `cornell.json` **51.07% of texels are negative**, so a
  magnitude form would fold half the image onto the other half, and Sobel already reports magnitude
- feat: `AovId::LoG` (3 channels) -- `t * laplacian5(L_t)` with gamma = 1 (Lindeberg 1998), maximised over the ladder. **R** is
  the extremal magnitude, **G** the winning scale as a frequency in cycles/degree through the camera's vertical FOV and the
  *traced* height (so the axis holds at any render scale), **B** the polarity. gamma = 1 is not a knob: it is the unique exponent
  for which the response to a blob of scale `t0` peaks at exactly `t = t0`. A response *field*, not detected extrema -- detection
  needs a magnitude threshold, which is the authored constant the brief forbids, and a sparse point set is not an image
- refactor: `assets/shaders/edge_filter.frag` and `hsv_display.frag` **deleted**, with `RequiredShaders`, `loadShaders`,
  `EdgeFilterUniforms`, `HsvDisplayUniforms`, both setup functions and seven cached uniform locations in `AppResources`. The
  viewer and the headless API now share one CPU implementation. `fullscreen_triangle.vert` and `PostProcessPass` stay -- OCIO
  uses them
- refactor: `evaluateFilterAov(AovId, FilterInput, ThreadPool)` is the one dispatch, with no `default` arm, so `-Werror` rejects
  a filter AOV that nothing routes. `headless_renderer.cpp` previously had `default: -> hsvAov`, so a new filter AOV would have
  silently returned HSV; `api.aov_filter_dispatch_is_total` is the check that would have caught it
- refactor: both open-coded `isPostFilterAov` OR chains in `main.cpp` are gone. `selectPathTracedImage` becomes `resolveAovImage`
  and answers for all three producers, so `presentFrame` and `samplePixelProbe` have **no filter special case at all**
- fix: the pixel probe no longer does a synchronous `glReadPixels` for filter AOVs -- the one place the render thread blocked. It
  reads the cached HdrImage texel like every other AOV, so it reports the true scene-referred value instead of the 8-bit
  composited framebuffer
- fix: `ensurePathTraceDisplayTexture` was passed a hardcoded generation of `0` for filter AOVs, a latent staleness bug the GLSL
  path masked. It now receives `FilterCache::revision`, which the image's fixed address cannot supply
- feat: `FilterCache` in `AppResources` evaluates the selected filter once per published pass, keyed on owner, generation and
  sample count -- one entry, not a map, since exactly one AOV is on screen. Closes the roadmap's "cache the post-filter AOVs"
  item. New `filterMs` stage, reported as a bursty "aov filter" dashboard row with a duty cycle and as `filter_ms` in the
  benchmark log; `blitMs` subtracts it so the "present blit" row stays truthful
- perf: register-blocking the tap loop (`kConvolutionBlock = 32`) takes the pyramid from **45.6 ms to 30.1 ms** at 2048x1152 on
  8 threads, DoG 46.9 -> 36.4 and LoG 74.8 -> 55.5. The naive form re-streams the accumulator and the centre once per tap, 20
  bytes per tap-pair against 8 -- 5.1 GB against 2.0 GB per pyramid. `__restrict` and a precomputed fold table in place of a
  per-tap modulo were worth a further 4%, so aliasing was never the bottleneck. Output is **bit-identical** either way
  (`--compare-exr` reports RMSE 0): only the memory access pattern moved, not the accumulation order. `results/wave-perceptual-aovs`
- fix: **the GLSL clamp to `[0,1]` is gone, and it was discarding real signal.** On `cornell.json` at 512x288, Sobel reached
  51.04 with **5.92% of texels clamped**, Gabor 43.42 with 2.84%, Luminance 13.01 with 1.22%. A 51:1 highlight edge read exactly
  1.0 on screen and through the probe; it now reads its true value, so the light panel's border stops saturating to a flat band
- feat: `aovCarriesRadiance` in `aov.h` replaces the producer as the test for whether display exposure applies. The six radiance
  lanes and the four filters that are positively homogeneous of degree one in radiance take it; ratios, reflectances, counts,
  lengths, directions and frequencies do not. `relativeExposureEv()` is 0 at the authored camera, so no AOV's default appearance
  changes -- the eight lanes that newly respond differ only once the exposure controls move, which is what was wanted
- change: exposure now applies **after** filtering rather than before. It commutes exactly for every linear filter. HSV is no
  longer scaled at all, which is a correction: the previous path scaled RGB first, so V moved while H and S did not
- change: R/G/B channel isolation now acts on the filter **output** rather than its input. `R + Sobel` meant "Sobel of the red
  channel" and now means "the red channel of the Sobel response". Accepted: these AOVs are defined on luminance. Isolation stays
  meaningful on LoG's three distinct channels
- test: `tools/filter_validate.cpp` (new, 12 checks) -- seven facility gates and five AOV gates, all analytic and synthetic, no
  golden images. The LoG gate asserts the closed form `4 t T0(s)(T1(s) - T0(s))` on 13 ladder rungs, that the argmax lands on
  `t == t0`, and that the peak approaches Marr & Hildreth's continuous `1/(4 pi t0)` with the derived `1/(2s)` discrete
  correction. `pyramid_matches_direct_diffusion` checks every level against a direct full-variance convolution of the original at
  its own samples, which validates the semi-group composition and the decimation in one assertion
- test: the affine-field invariant is exact at the plane level (`diffusion_reproduces_affine_fields` is bit-identical,
  `laplacian5_annihilates_affine_fields` is exactly zero) but only bounded at the AOV level, because the Rec.709 luminance
  reduction is not exact on an affine field. The AOV gate asserts that quantisation carried through the widest cascade step
  rather than pretending zero, and asserts its own interior is non-empty so it cannot pass vacuously
- docs: `PIPELINE.md` gains a Scale space section with the derivations the one-line comment budget cannot hold, DoG and LoG rows,
  and twelve references (Lindeberg 1990/1994/1998, Koenderink 1984, Witkin 1983, Burt & Adelson 1983, Marr & Hildreth 1980, Lowe
  2004, Wilson & Bergen 1979, De Valois et al. 1982, Abramowitz & Stegun 9.6.10, and the four IIR papers as surveyed-not-adopted).
  `arXiv:2601.16806`, cited for Sobel and unaccounted for, is replaced with Sobel & Feldman 1968, which `aov_filters.h` already
  carried; Gabor 1946 / Daugman 1985 added alongside
- docs: `ROADMAP.md` records the deferred designs so they are not re-derived -- CSF (blocked only on photometric calibration,
  with Barten 1999 chosen over Mannos & Sakrison / Daly / Watson & Ahumada because its parameters are physical rather than
  fitted, and Peli 1990 contrast over the shared pyramid rather than an FFT multiply), the Opponent/Retinex/CLAHE trio with
  Smith & Pokorny 1975 and MacLeod & Boynton 1979 justified against Stockman-Sharpe and DKL, and the Gabor bank's four remaining
  authored constants. Foveal/peripheral sampling is folded into the adaptive-sampling item, where it is a sample allocation
  rather than an AOV

## `README.md` becomes a landing page, the reference body moves to `docs/PIPELINE.md`

316 lines to 60. The README answered "what is this" and "how do I use the thing" in the same file as every
subsystem table, so neither read well: a first-time reader scrolled past 28 AOV rows to reach the build
command.

- docs: `docs/PIPELINE.md` (new, 236 lines) takes Pipeline, Components, Session settings, Benchmark tooling,
  Testing, Material Library, AOV and References verbatim in structure and rewritten for density -- the AOV
  table splits into the four HUD groups with a **Source** column naming the producer (10 traced, 14 raster,
  4 filter, gated by `aov.h`'s own `static_assert`)
- docs: the README keeps the abstract, sample image, build, run, controls and the Python quickstart, and a
  one-line pointer at `PIPELINE.md` and `ROADMAP.md` replaces the six-row documentation table
- docs: a Controls table replaces the prose that named `R` as the camera reset. `R` isolates the red channel;
  `0` resets the camera (`main.cpp`), and `PIPELINE.md`'s "Debug camera controls" row carried the same stale
  key over from the README
- refactor: `ROADMAP.md`'s three `../README.md#...` links repoint at `PIPELINE.md`, and the `aov.h` grouping
  comment drops its "Grouped by README.md" pointer, keeping the claim
- note: dropped rather than moved -- the build-target inventory, the static-analysis section (clang-tidy and
  the `cppcheck` target), the `third_party/README.md` pointer, the `bench_compare run` A/B invocation, and the
  Python prose on the C ABI, scene-referred values, `show_sky` and per-call producer cost. All still hold;
  none has a home in the tree now

## `docs/DERIVATIONS.md` removed: the load-bearing lines move to the code they govern

1706 lines across 31 sections, addressed from 48 source comments. Measured before deleting: **188 of its
234 measurements already appear in this file or the README**, so most of it was a second rendering of
what the chronology already held -- its "Running-mean forward error" section, for instance, is reproduced
recurrence and citations at the `driver_validate` entry below. Of the 46 unique figures, 29 sat in the
three validation sections, where they justify a tolerance.

- refactor: the contract material moves to the declaration it governs rather than to this file, which is
  chronological and so cannot be looked up by topic: the generation/requestedGeneration cancellation onto
  `renderPathTraced`, the one-renderer-per-thread and caller-allocates rules onto `pathtracer_c.h`, the
  bucket rule onto `PathTraceResult`, the binade bracket onto `kOverRangeEvRadius`
- refactor: the 7 measurements found nowhere in the tree land beside the constant they size -- the Karis
  revert's 7.8e-4 and the 2.03e-7 residual on `index_matched_coat`, 3.5e-5 at ior 1.33 on
  `coat_fresnel_average`, the roughness <= 0.022 / mu <= 4.4e-3 corner on `albedo_table_interpolation`,
  1.93e-2 on the sphere tessellation, and the 3.6e-3 and 3.9e-3 transmit-side residuals in `albedo_table`
- refactor: all 48 pointers lose the trailing reference and keep their claim. Six comments that ended
  "...are in docs/DERIVATIONS.md" were truncated mid-sentence by that strip and are rewritten to carry the
  fact instead of pointing at it; `.clang-tidy` now names the three false positives rather than counting
  them, the file having six disabled checks
- note: `tools/comment_lint.py` is the constraint that shaped this -- one line per comment, 140 columns,
  enforced by `style.comment_budget`. It is why the `renderPathTraced` parameter table could not survive as
  comments: `showSky`'s primary-miss-only gating, `instanceLightIndex`'s -1 sentinel, and the
  `scrambleSeed`/`sampleBase` pairing are now stated only where they are already documented elsewhere
- note: comment-only, and the evidence is byte-level -- `render_beauty` at the profile default is
  **byte-identical** across the change (CRC32 `94635855`), the diff contains **zero** non-comment changed
  lines in `.cpp`/`.h`, and `ctest` is **133/133**

## Headless display parity: one CPU display encode, and a background the caller controls

A Beauty render through the Python API looked nothing like the same scene in the viewer -- darker
throughout, and with the HDRI behind the Cornell box where the window shows black. Two independent
causes, both real divergences rather than sampling noise.

`Renderer.render` returns scene-referred linear radiance, by design, but nothing in the C ABI or the
Python package could turn it into a picture: `applyOcioDisplayTransform` had exactly one caller,
`tools/render_beauty.cpp`. `plt.imshow` on the raw floats clips to [0,1] and maps *linearly*, which
the notebook recorded itself -- `Clipping input data ... Got range [0.00034091552..14.97626]`. The
viewer meanwhile applies exposure and `Linear Rec.709 (sRGB)` -> `sRGB - Display` under the
`Un-tone-mapped` view, a curve with no tone mapping in it at all. Measured on one buffer, the median
texel reads 0.0419 clipped-linear against 0.2275 encoded -- **5.43x** -- and the lift runs from 5.85x
in the [0.01,0.05) band to 1.23x above 0.5, which is the sRGB inverse EOTF's shape and not a
brightness offset.

Second, `headless_renderer.cpp` passed `/*showSky=*/true` as a literal while the viewer defaults
`showSky` to `false`, and the ABI had no field to say otherwise -- nor for `envLightEnabled`, which
already existed as `std::optional<bool>` on the C++ Request but was pinned to `nullopt`.

- refactor: `encodeForDisplay` and `ditherOffset` move out of `render_beauty.cpp`'s anonymous
  namespace into `gfx/ocio_cpu_transform`, beside `applyOcioDisplayTransform`. One CPU definition now
  serves the CLI and the new ABI entry point, so a Python preview and the CLI's PNG cannot drift --
  the same argument `ocio_cpu_transform.h` already made for the transform itself. It takes
  `std::span<const float>` of packed RGB rather than an `HdrImage`, so the ABI hands over the caller's
  buffer with no copy of its own and the only copy left is the in-place working buffer OCIO needs
- feat: `pt_display_encode` and `pathtracer.display_encode(image, exposure_ev=, display_transform=)`,
  scene-referred linear to display-referred 8-bit sRGB. Verified against `render_beauty`'s PNG at
  256x144x32: **0/110592 channels differ** -- exact rather than within a tolerance, both calling one
  function whose dither is a pure function of `uv`
- feat: tri-state `show_sky` and `env_light_enabled` on `PtRenderRequest` and `Renderer.render`,
  backed by `std::optional<bool> showSky` on `HeadlessRenderer::Request`. `PT_DEFAULT` (-1) defers --
  to `true` for `show_sky`, to the scene's authored `environment.lightEnabled` for the other -- so no
  existing caller changes behaviour. Anything outside {-1, 0, 1} is rejected through `err` rather than
  coerced, a caller writing 2 having meant something the ABI cannot honour
- note: the two flags are not interchangeable and the difference is now asserted, not described.
  `show_sky` gates the primary ray's own miss, so the box interior is **bit-identical** with the sky
  on and off; `env_light_enabled` removes the environment from the light set and does change the
  image. Carried at both levels: `api.show_sky_changes_only_the_background` and four Python tests
- note: the renderer is untouched and the evidence is byte-level -- `render_beauty`'s PNG has the same
  SHA-256 before and after the hoist (0/110592 channels differ), and Python's Beauty is still
  bit-identical to the CLI's linear EXR. `std::lround` replaces `(value * 255.0F) + 0.5F`, silencing
  `bugprone-incorrect-roundings` now that the code sits in a clang-tidy-checked translation unit; the
  two agree over the clamped [0,1] domain and the identical golden is the proof rather than the claim
- note: `show_sky=False` measures 1.015x faster at 256x144x8 (274.6 -> 270.7 ms, best of 5). A primary
  miss returns early instead of sampling the environment, but misses are a small fraction of this
  frame, so it is reported as measured rather than claimed as a benefit
- note: still divergent, and deliberately left for their own commits -- the headless default remains
  sky-on where the viewer defaults sky-off, flipping it being a behavioural change to `render_beauty`
  and its goldens; env rotation and env exposure stops stay hardcoded at `headless_renderer.cpp`
  188-189 with no headless equivalent of the viewer's sliders; and `aperture`/`shutter_seconds`/`iso`
  remain inert headlessly, `ev100()` being read only by the GUI and only as a delta against the
  profile defaults. Full write-up and artifacts in `results/HEADLESS_DISPLAY_PARITY.md`

## Render resolution unified: `profile.json` authors it, the window is only a viewport

`profile.json`'s `window.width/height` meant two incompatible things. The GUI handed them to
`glfwCreateWindow` as *screen points*, and GLFW's macOS default doubled that into a 2048x1152 backing
store which `framebufferSize()` fed straight to the tracer -- 4x the authored pixel count. The headless
path read the same two numbers as *render pixels* and rendered 1024x576. The two readouts printed both
figures under the same label: the startup block `window 1024x576`, the dashboard `window 2048x1152`.
`raster_bench` carried the 2x as a hand-typed `2048` with a comment explaining Retina.

- refactor: **the resolution moves to `render.width`/`render.height`, in pixels, and the `window` block is deleted.** It sits beside `renderScale`, which multiplies it rather than the framebuffer. `HeadlessRenderer::defaultWidth/Height` now read it, so the GUI and the C/Python API size from one number by construction instead of by coincidence
- feat: **the window is an independent viewport.** It opens 1:1 -- `glfwGetWindowContentScale` on the window GLFW actually placed, not a guessed monitor, so no scale factor is assumed or hardcoded -- and is freely resizable after. A resize changes no trace input, so it cannot restart an accumulation; `GLFW_SCALE_FRAMEBUFFER` is now stated rather than left to a per-platform default. Under fractional scaling the opening framebuffer may land a pixel off, which costs nothing now that nothing reads it as the render size
- feat: **letterboxed and nearest-magnified.** `fitAspect` (`gfx/viewport.h`) returns the largest rect of the image's aspect that fits the viewport, centred, so a mismatched window shows bars rather than a distorted frame. `GL_TEXTURE_MAG_FILTER` becomes `GL_NEAREST`, superseding the `GL_LINEAR` upscale the `renderScale` entry below describes: one traced pixel reads as one visible block rather than an interpolated value that was never rendered, so a 320x180 render can be inspected full-screen and the probe agrees with what is on screen. Minification stays `GL_LINEAR`
- fix: **the readbacks no longer assume the image fills the window.** The pixel probe maps the cursor into the image rect and reports nothing over a bar; the histogram blits the rect alone, so bars never reach the bins. Both were previously normalised by the whole framebuffer
- refactor: the resolution leaves `ViewInputState`. It is fixed for the session and the traced size varies only through `renderScale`, which the trigger already carries, so comparing it every frame was work that could never fire
- refactor: readouts renamed to what they are. The startup block prints `resolution`, the dashboard prints `image` / `trace` / `window`, and the HUD's `Viewport` section becomes `Resolution`. `image` and `resolution` cannot disagree, being the same field
- test: `viewport_fit_preserves_aspect_and_centres` (`display_validate`), 8 cases x 3 assertions -- expected rects from exact halves and thirds, containment within the viewport, and aspect held to the `(0.5 + 0.5*aspect)/h` bound that half a pixel of rounding on each extent allows. `profile_config_integer_counts` retargets `render.width/height` and gains the missing-key case the window loop never had. `ctest` **132/132**
- note: **image-neutral where the resolution is unchanged, and the defaults are now consistent.** `render_beauty` at the profile default is byte-identical across the change (CRC32 `3396778085`, ray counts identical to the digit), as is `raster_bench` at an explicit 1024x576 (CRC32 `2784833415`). What changes is the GUI's default: it now traces the authored 1024x576 rather than 2048x1152, a 4x reduction in paths per pass. Historical entries below keep the resolutions they were measured at

## NEE shadow rays: exact target reconstruction from the offset origin

Reported as a dot lattice in the Shadow AOV across `cornell.json`'s spheres. Not a mesh fault -- the glTF
is 980 near-equal-area triangles, consistent winding, unit radial normals, no degenerates -- and not the
AOV's own gates: dropping `nearSide`'s `geoCos` term and hoisting the occlusion test out of the
`eval.pdf`/`bsdfValue` gate each changed the image by nothing. Instrumenting the blocker's `t/distance`
put it at `~1.0`: the blocker was the light itself.

- fix: **the NEE shadow ray's `tMax` was measured at `shading.position` while the ray leaves from the Chiang/Li/Burley offset origin.** On curved geometry that offset is a tangent-plane displacement of order `e^2/(2R)`, which for `cornell.json`'s spheres (`e = 0.026`, `R = 0.15`) is `~2e-3` against `kShadowDistanceEpsilon`'s `6e-4` back-off at that distance -- so the ray crossed the emitter's plane beyond `tMax` and the light's own front face, which `appendQuadLights` puts in the BVH, occluded it. The ray is now pbrt's `SpawnRayTo` (PBR 6.8.6): the sampled point is reconstructed as `shading.position + wi*distance` and the ray re-formed from the offset origin to it, so the back-off is relative to the distance actually travelled. A back-off along `wi` alone is not sufficient and was measured failing -- the offset origin's ray is a parallel shift, meeting the light's plane at `distance - dot(delta, n_light)/dot(wi, n_light)`, not at `distance - dot(delta, wi)`. The environment keeps the direction as drawn: its point is at infinity
- fix: **this was an energy loss, not an AOV-only defect.** Direct light from an area light was being discarded wherever the offset exceeded the back-off, which is every sufficiently curved emitter-facing surface. On a diffuse variant of `cornell.json` the spheres carried the same lattice in *beauty*: `relMSE 0.0149`, `linear RMSE 0.0463` at 640x360x64. The shipped scene hides it because chrome and glass take most of their light through the BSDF-sampled MIS half, leaving `relMSE 6.0e-06`
- test: `quad_light_visibility_on_curved_receiver`, a hard zero rather than a tolerance -- nothing lies between a convex receiver and a light it faces, so the Shadow AOV over a sphere under an unobstructed quad must read exactly 0. Two tessellations, `64x32` and `32x16`, because the offset scales with `e^2`; `makeSphereScene` takes slices/stacks, defaulted so its four existing callers are unchanged. The suite's other quad-light checks all use a FLAT receiver, where `shadowTerminatorOffset` is a no-op and the offset is `kRayEpsilon` alone -- one part in `1e4` of the distance, comfortably inside the back-off -- which is why 130/130 passed over this
- test: verified by breaking the code under test. The old construction fails both rows, at 0.874 (fine) and 0.987 (coarse); the `wi`-only back-off fails the coarse row at 0.535 while the fine row passes, which is the `e^2` scaling showing up as detection. `ctest` **131/131**
- note: work-neutral and perf-neutral. Ray counts identical to the digit at 640x360x64 (`primary 15272448, bounce 27615693, ao 7778812, shadow 21871231`) -- no sampler dimension is consumed and no ray is added. `pass_ms` B/A **1.0041 [0.9864, 1.0193]** at n = 16, not resolved

## Performance pass, wave 1: closure-constant hoisting, tile occupancy, threaded EXR decode

`ROADMAP.md`'s "Performance and Memory efficiency pass" item, which had monitoring but no reduction work.
Measured first: a `-bench` baseline at 2048x1152 put `pass_trace_ms` at **98.4%** of pass cost, leaving
`accumulate_ms` at 1.44% and the frame loop with 12.6 ms/frame of vsync slack -- so this wave is entirely
in the integrator, the interactive tile split and startup, and not in the display or accumulate paths the
plan had ranked higher before measuring. A `sample` profile then named the hotspots; three of the top five
turned out to be closure constants recomputed per evaluation. Full numbers in the local capture set
(`results/WAVE1.md`, gitignored like `results/wave7`).

- perf: **`evaluateContinuousLobes` evaluated `directionalAlbedo(wi.z, roughness)` twice** -- once for the specular energy compensation, once inside `diffuseKdAt` -- for one bilinear pair over two 256 KB tables. Resolved once in the caller and passed to both lobes
- perf: **the Kulla-Conty tint was rebuilt on every evaluation** from `fresnelAvg` and `albedoAvg`, both closure constants `computeLobeProbabilities` had already combined for `msReflectEnergy`. Now carried on `LobeProbabilities` as `multiScatterFms` and consumed by both
- perf: **EON's uniform-mix weight is a function of `wo.z` and `diffuseRoughness` alone**, so its `std::pow(r, 0.1)` was a per-evaluation cost on a per-vertex quantity. Hoisted to `lobes.eonUniformMix`; `sampleEon` and `pdfEon` now read it
- perf: **EON's clipped-LTC state is likewise wo-only** -- the fit coefficients, the LTC basis (a `sqrt`), the normalisation `s` (a second `sqrt` and a divide) and `detM` were all rebuilt inside `cltcPdf` on every call, where only `wh` and `lenSq` depend on `wi`. Stored as `eonLtcM`/`eonLtcBasisT`/`eonLtcS`; `cltcSample` transposes the stored basis back, which is exact and costs no arithmetic
- perf: **`rotateAboutY` took the angle and called `sin`/`cos` per ray**, up to three times per bounce, for a rotation that is constant across a whole pass. `YRotation` carries the resolved pair, built once in `LightSet`'s constructor. Its `inverse()` negates the sine rather than calling trig again -- exact, since `cos` is even and `sin` odd bit-for-bit, verified over 400k angles
- perf: **`sampleBilinear` filtered four taps that resolve to one texel.** Where the wrapped coordinates coincide -- every 1x1 default map, which is every material slot in `cornell.json` -- the three redundant fetches and the three `mix`es are skipped. The condition is on the resolved coordinates, not on a size, so it also covers a width-1 or height-1 map. Bilinear interpolation of a constant field *is* that constant, but glm's `mix` is `x*(1-a)+y*a` and does not realise that exactly: `mix(V,V,t)` differs from `V` by 1 ULP for a minority of `(V,t)`. The early return is therefore the exact value where the mix rounded -- strictly more correct, not merely equal. Byte-identical on every scene verified here; a different constant could shift a last bit, in the correct direction
- perf: **the path-trace tile size is derived from the render target** rather than pinned at 96 px. At `interactiveRenderScale 0.1` a 2048x1152 framebuffer renders 205x115, which 96 px tiles split into 6 across 8 workers. `kTilesPerThread = 2` shrinks the tile only where the grid cannot fill the pool, floored at `kMinPathTraceTileSize = 32` where the `(4t+4)/t^2` halo re-trace reaches 12.9%. **59.09% faster** at 205x115, `pass_ms` B/A 0.4091 [0.4013, 0.4226] at n = 24
- perf: **`Imf::setGlobalThreadCount` was never called**, so OpenEXR decompressed every scanline block of the HDRI on the calling thread. Process wall for a 64x64x1 render drops **111.91 -> 69.38 ms** median at n = 12, roughly halving the load phase
- fix: `samplePixelProbe` ran before the `showHud` check, so its synchronous 1x1 `glReadPixels` -- taken for the four post-filter AOVs, and documented in place as "the one place the thread blocks" -- was paid every frame with the HUD hidden. Now gated
- test: **the tile-size derivation had no coverage, and neither did the property it rests on.** `render_is_invariant_to_thread_count` looked like coverage but renders 8x8, a single tile at every thread count. `tile_size_covers_the_target_and_fills_the_pool` is 1215 assertions over extents including 0 and 1 and thread counts including 4096 -- the tile stays in range, the grid covers every pixel, and it shrinks only where the next size up could not fill the pool. `render_is_invariant_to_tile_size` renders 96x96 at an explicit 1 and 8 threads, which resolve to 48 px and 32 px tiles on any host, and first asserts the two differ so it cannot go vacuous. `pathTraceTileSize` is declared in the header so the pure function is testable directly
- test: both verified by breaking the code under test. `kFilterExtent` 1 -> 0 fails `render_is_invariant_to_tile_size` at 3348 of 36864 floats **while the pre-existing thread-count check still passes**, which is what proves the new one carries detection the suite lacked; removing the floor from the halving fails the pure-function check at 2048x2048 on 4096 threads, which shrinks to 24. `ctest` **130/130**
- note: **no image change from any of the above.** The beauty EXR is byte-identical to the pre-change binary at 640x360x32 (`linear RMSE 0, relMSE 0`, crc32 `639671753`) and at 205x115x32 (crc32 `1416599511`), and `ctest` is **128/128**
- note: the tile change is the one item that is **not work-neutral**: primary rays rise 7.4% at 205x115 because smaller tiles re-trace more halo, against a predicted ~8%. `bench_compare` correctly reports `work: DIFFERS` there. The image is unaffected because a pixel accumulates its 3x3 neighbourhood in raster order whatever rectangle encloses it, which the 1 px halo guarantees its owner traces. 2048x1152, 1024x576 and 640x360 keep 96 px tiles and are untouched in both image and timing
- note: **cumulative trace gain is 6.31%** at 640x360x32, `pass_ms` B/A 0.9369 [0.9204, 0.9489] at n = 18. Two candidate changes measured as not resolved and were reverted rather than kept: power-of-two masking in `wrapPixel` (1.0024 [0.9956, 1.0086]), and a `kTilesPerThread` of 4 that reshaped grids which did not need it
- note: the profile's remaining top entries are Embree's `BVHNIntersector1::intersect` at ~15%, which needs the packet/stream tracing that ROADMAP Large #3 owns, and `sampleBilinear` still at ~11% on 1x1 maps -- the structural fix there is resolving a constant texture slot to a constant at load, which is a `Material` change rather than a sampling one

## Repository hygiene: the header lint gate, the pathtracer namespace, licensing

`.clang-tidy`'s `HeaderFilterRegex` was `^(?!.*third_party).*(include|src)/.*`. clang-tidy matches with
`llvm::Regex` (POSIX ERE), which has no lookahead, so the pattern matched nothing and every header in
`include/` and `src/` went unchecked -- silently, because a filter that matches nothing and a filter that
finds nothing are indistinguishable in the output. Fixing it exposed 27 findings, 11 of them outside the
known over-length-function set. All 11 are fixed here rather than suppressed.

- fix: `HeaderFilterRegex` rewritten without lookahead, exclusion moved to `ExcludeHeaderFilterRegex` (raises the clang-tidy floor to 19). Verified by diagnostic presence, not by reading the pattern
- fix: `kGaborKernelSize`, a `std::size_t` extent replacing the `kGaborOrientations * kGaborTaps` int product at its four `std::array` sites. The factors stay `int`; `kGaborRadius` is used signed as a loop bound
- fix: `rasterizer.cpp` widens before the index arithmetic (`static_cast<std::size_t>(k) + 1`), not after
- fix: `HeadlessRenderer`'s constructor takes `defaultCamera` by const reference -- it is copied into the member, never moved from, so by-value cost one copy more than needed
- fix: `bench_compare`'s build grouping is an explicit if/else. The ternary it replaces evaluated `emplace_back()` in one branch and `*it` in the other, which is correct but reads as a use-after-invalidation and cppcheck scored it an error
- fix: `pathtracer_c.cpp` states `.envLightEnabled = std::nullopt` rather than leaving the field implicit, and includes `<optional>`/`<cstddef>` it was reaching transitively
- fix: stale `.clang-tidy` comment naming a `vendored_no_tidy` target that does not exist; the exemption is `vendored_cgltf`/`vendored_imgui` plus the new header exclusion
- refactor: `engine` -> `pathtracer` throughout -- `include/engine/` -> `include/pathtracer/`, the seven `engine::*` namespaces, `ENGINE_*` macros and CMake variables, `engine_check`/`engine_git_sha`/`engine_discover_checks`, `EngineChecks.cmake`. 1132 qualified uses across 117 files. HOST's `.gitmodules` section renamed to match its long-since-renamed path
- refactor: `pathtracer_target_defaults(target [NATIVE] [IPO] [TIDY])` replaces `-Wall -Wextra -Werror` repeated 19 times and the IPO block 9 times. `NATIVE`/`IPO` are opt-in so the codegen tools keep reproducing their committed output bit-for-bit; verified by diffing compile flags across all 75 translation units
- feat: `LICENSE` (Apache-2.0; the repo was public with none, so nobody could legally use it), `third_party/NOTICE` carrying the three MIT notices, and `third_party/README.md` recording each dependency's version, upstream and vendoring form
- docs: README gains the target inventory and the static-analysis section; CHANGELOG reordered newest-first with the Phase 0-5 build-out kept in sequence at the end; ROADMAP's over-length-function count corrected from 11 to the measured 16
- note: behaviour-neutral by construction and by measurement. Beauty and the Depth/Normal/Albedo AOVs are bit-identical before and after, 120/120 ctest entries pass, cppcheck is clean, and a 16-round interleaved A/B resolves no timing change (`pass_ms` B/A 1.0045 [0.9874, 1.0248], work identical every run)

## Lookahead AOV

`agent.md` §2 reads the Depth AOV per retina point to get distance-to-surface. Depth is unbounded planar camera-space Z in scene units, auto-ranged only at display time, so every consumer of it has to source a range of its own before the number means "near" or "far". Lookahead is that remap made into a lane: one `lookaheadDistance` float in `profile.json` declares a horizon, and the lane carries `clamp(1 - Z/lookaheadDistance, 0, 1)` -- 1 at the camera plane, falling linearly to 0 at the horizon. Inverse z-depth on a declared scale rather than an inferred one.

- feat: `AovId::Lookahead`, the 14th `RasterGBuffer` lane, written in `shadePixel` beside `depth` from the `viewZ` the depth pass already resolved -- a remap in the existing pass, not a second one. Available in the HUD dropdown, `render_beauty --aov lookahead`, the C ABI and Python like every other AOV, with no new plumbing: it rides `PathTraceSettings`, which the rasterizer is already handed for `ior` and the material fields
- feat: `pathTracer.lookaheadDistance` in `profile.json` (default 10.0, scene units), validated `> 0` at load beside `aoMaxDistance` -- it is a divisor at every covered pixel, and at or below zero the lane is inf/NaN rather than a bounded gradient
- note: geometry at or beyond the horizon reads 0, which is also what the per-row clear leaves on a primary miss. `alpha` is the discriminator, exactly as it already is for the other lanes. The polarity is deliberate: for a proximity signal, "past the horizon" and "nothing there" are the same answer
- note: the lane needs no display special case. It is already in [0,1], so `presentFrame`'s unscaled Raw passthrough is correct, and it deliberately does **not** get Depth's auto-ranging exposure branch or the full-image max scan that feeds it -- a declared scale is the point
- refactor: `main.cpp`'s `selectPathTracedImage` drops its 24-case switch for `aov_routing.h`'s `gbufferLane`/`pathTracedLane`. That switch was a third copy of the AovId-to-buffer mapping `aov.cpp` owns and `headless_renderer.cpp` already consumes, so a new AOV needed a row in two places or the viewer silently showed black. **50 lines shorter**, and the equivalence was checked mechanically -- both tables extracted and diffed, 24 entries each, no key or target differing -- before the switch was deleted. `api_validate`'s `aov_tables_are_total_and_consistent` already pins the contract the viewer now depends on, for every `AovId`
- test: `rasterizer_validate` gains a `lookahead` row against the independent Embree oracle, on all three poses. Its horizon is derived from the synthetic scene's own depth span rather than set to a constant: the midpoint puts roughly half the frame in the linear region and half clamped, so both branches carry pixels. Verified by mutation -- inverting the ramp, deleting the clamp, and dropping the divisor each fail it. The first draft used the scene's far extent and **did not** catch the missing clamp; the midpoint does
- test: `io_validate`'s `profile_config_scene_scale_distances`, 11 assertions over `aoMaxDistance` and `lookaheadDistance` -- zero, negative, non-numeric and missing must each be refused, and an accepted value must reach the struct unaltered and leave the other distance alone. Neither divisor had any load-boundary coverage before
- test: `python/tests` gains the end-to-end relation, `lookahead == clamp(1 - depth/D, 0, 1)` read off one render's own Depth lane against the shipped profile's horizon. Max observed error on cornell is **exactly 0**
- note: **no image change.** Beauty and the Depth/Normal/Alpha/ObjectID lanes are bit-identical to the pre-change binary -- linear RMSE 0, relMSE 0 on every one, measured before and again after the `selectPathTracedImage` refactor. `ctest` **120/120**
- perf: **`raster_bench`, randomized interleaved A/B against the pre-change binary, 20k triangles at 1920x1080.** `frame_ms` B/A **1.0248 [1.0142, 1.0322]** at n = 16 (n = 12 agreed: 1.0263 [1.0157, 1.0355]), `work: identical across every run`. A 14th lane costs **~2.5%** of the rasterizer -- one more per-row clear and one more texel write out of 14, against clipping, binning and the depth pass that do not scale with lane count. An earlier n = 12 round read 1.0605 but with an interval of [1.0041, 1.1128] that contains both, and its width is why it is not the quoted figure
- perf: `render_beauty` unchanged, as it must be -- `pass_ms` B/A **0.9954 [0.9529, 1.0321]** at n = 10, not resolved. The integrator is untouched
- fix: `render.defaultAOV` is a raw index into `kAovNames` that `main.cpp`'s startup spec block dereferences unchecked and the HUD casts to `AovId`, with no bound anywhere. Now validated at load like every other field, with `profile_config_default_aov_is_in_range` covering both ends, the first value past the top and a missing key. Pre-existing, but inserting an AOV mid-enum is what makes a stale index land somewhere new
- fix: `rasterizer.cpp`'s two per-shade write counts said 14 when `shadePixel` issued 13 `writeTexel` calls. Both now read 14 against the actual 14
- note: **`AovId` indices shift.** `Lookahead` is inserted after `Depth` for category grouping, so every AOV from `HSV` on moves up one. Nothing in the repo persists a numeric index except `profile.json`'s `defaultAOV`, which ships as 0 and is unaffected -- but a local profile carrying an index of 4 or more now names a different AOV, and one of 28 is now refused rather than read out of bounds
- note: the horizon is a property of the consumer's reach, not of the scene, and the lane is deliberately not auto-ranged -- so the shipped default only ever suits scenes of roughly its own scale. On cornell (depth extent 5.52-6.50) the default 10.0 puts the lane in [0.350, 0.448]; at 100.0 the same geometry reads [0.935, 0.945] and looks uniformly white, and at 6.0 three-quarters of it clamps to 0. Setting this to the agent's actual reach is the whole interface

## HOST integration: headless library, full AOV coverage, Python binding

`ROADMAP.md`'s HOST-integration items 1 and 3. The renderer had no library target at all -- every one of the ~18 executables re-listed the subset of `src/*.cpp` it needed -- so there was nothing for a foreign runtime to bind to, and `render_beauty`, the headless tool, linked glfw and all of imgui because `vendored_no_tidy` bundled imgui with `cgltf_impl.cpp`. Separately, only 10 of the 27 AOVs had any headless path: Depth and Normal are rasterizer-backed and `render_beauty` never ran the rasterizer, and HSV/Luminance/Sobel/Gabor existed only as GL fragment shaders. Item 2, the retina point field, is deliberately not here: the camera renders a full frame instead, which is the simplification HOST is built on today.

- refactor: `pathtracer_core`, one static library holding every source that does not include a GL, GLFW or imgui header, linked by every executable. `vendored_no_tidy` splits into `vendored_cgltf` (headless) and `vendored_imgui` (GUI), which is what took glfw and imgui off `render_beauty`. `cie.cpp` stays out, since `metal_fit_core` builds it without `-march=native`/IPO for reproducibility and having it in both would be a duplicate symbol. **CMakeLists.txt is a net 51 lines shorter** and the 9 validator targets collapse into one `foreach`
- feat: `engine::api::HeadlessRenderer` -- a scene loaded once and rendered many times, lifting the ~120 lines of setup that were inlined in `render_beauty`'s `main()` and reachable from nothing else. Neither copyable nor movable: `LightSet` stores a pointer to the `EnvironmentMap` beside it and `ThreadPool` is non-movable by design, so `open()` returns a `unique_ptr` and the object's address is part of its own invariant. Each producer runs **at most once per call** -- the 10 path-traced lanes share one sample set and one reconstruction filter, the 13 rasterizer lanes one scan-conversion, the 4 filters the one accumulated Beauty -- so `beauty+depth+normal+sobel` is one accumulation, one rasterizer pass and one filter, not four renders, and a request with no path-traced AOV skips the integrator entirely
- feat: CPU implementations of the four Beauty filters (`debug/aov_filters.cpp`), which previously existed only as `edge_filter.frag`/`hsv_display.frag`. **This is what makes `agent.md` §3's retina tiers 1-2 (Luminance, Sobel) reachable at all**, and what makes "any of 27 AOVs" true rather than aspirational. `buildGaborKernel` moved out of `main.cpp` and the shader now uploads that same function's output, so the CPU and GPU banks are one set of weights by construction, the way `gbuffer_shading.h` already keeps the path tracer and rasterizer in agreement. The CPU forms deliberately omit four things the shaders do -- the exposure multiply, channel isolation, invert, and the clamp to [0,1] -- all display state rather than properties of the AOV; a consumer training on these needs the unclamped scene-referred quantity. Luminance is materialised once per call rather than per tap, which Sobel reads 8 times and Gabor 25, so the intermediate plane removes up to 25x redundant dot products a fragment shader has nowhere to cache
- refactor: AOV classification is one table (`debug/aov.cpp`'s `aovSource`/`aovChannels`/`pathTracedLane`/`gbufferLane`/`aovIdFromName`), replacing four copies of the same knowledge -- `main.cpp`'s `selectPathTracedImage` and `aovNeedsLightTransport`, `render_beauty`'s private lane table, and the README. `aovNeedsLightTransport` is now *derived* from `aovSource` rather than tabulated beside it, so the two cannot disagree. `aovChannels` is per-AOV meaning, not storage: `HdrImage` is always 4 floats/texel and scalars are broadcast to RGB for display, but a consumer reading a depth map wants its one real channel, not three copies
- feat: `include/engine/api/pathtracer_c.h`, a flat C ABI over `HeadlessRenderer`, built as `libpathtracer_c`. A C ABI rather than pybind11 on purpose: no new third-party dependency, no Python headers at build time, one library for every interpreter, and `ctypes` drops the GIL around each call by construction. Output buffers are **caller-allocated**, so no memory ownership crosses the boundary and there is no `free` protocol to get wrong
- feat: `python/pathtracer` -- `Renderer(scene).render(aovs=..., width=, height=, samples=, seed=, camera=)` returning `{name: (H, W, C) float32}`. numpy allocates every buffer and the ABI writes into it, so `torch.from_numpy` shares memory (asserted by `data_ptr`) and leaves one explicit `.to(device)`. `Camera` is a frozen dataclass carrying the full photographic set; its docstring states that `aperture`/`shutter_seconds`/`iso` set exposure **only**, the camera being a pinhole with no thin-lens depth of field. `AOVS` is read from the library rather than restated, so Python cannot hold a stale copy of a list C++ owns
- feat: `render_beauty --aov` now takes any of the 27, not the 10 path-traced ones, and `--env-light` is carried through as `Request::envLightEnabled`. Per-pass wall clock and ray counts move to `HeadlessRenderer::RenderStats`, which also reports the rasterizer and filter timings separately -- averaging them into the path-traced `pass_ms` would describe neither
- fix: a rasterizer-backed `--aov` segfaulted. It runs no path-traced passes, so `milliseconds` was empty and the `minmax_element` reducing it dereferenced `end()`. The reporting now says what actually ran, and `image.gbuffer_aov` is the regression gate
- test: `tools/api_validate.cpp`, 9 checks / 162 assertions. The filters are asserted **analytically**, not against a GPU readback: a readback needs the GL context this path exists to avoid and would only show two implementations agree, not that either is right. A unit step edge has closed-form Sobel magnitude exactly 4 either side and 0 two pixels out; the odd-carrier Gabor bank sums to zero per orientation, so a constant field must give exactly zero response; Luminance of each Rec.709 primary returns that primary's weight; HSV inverts to RGB, including on the achromatic axis and above display white, where the shader's `1e-10` denominator guard only approximates the convention. Plus: every `AovId` is classified, sized and owned by exactly the producer its classification names; all 27 render finite; and a multi-AOV request is bit-identical to the same AOVs requested one at a time, which is the property the once-per-producer sharing has to preserve. `ctest` **118/118**
- test: `python/tests`, 16 tests. The headline one asserts Python's Beauty is **bit-identical** to `render_beauty --out-exr` at the same scene, resolution, seed and pass count -- exact equality, not a tolerance, because the renderer is deterministic by construction and any tolerance would hide the bugs worth catching. Also: a G-buffer-only request must be >10x faster than the same request with Beauty added, which is what proves the integrator is actually skipped
- note: **no image change.** `render_beauty`'s 256x256x32 Beauty EXR and PNG are byte-identical before and after the whole change, gated after the library extraction and again after `render_beauty` was rewritten onto `HeadlessRenderer`
- perf: **`render_beauty`, randomized interleaved A/B against the pre-refactor binary, cornell.** `pass_ms` B/A **1.0154 [0.9958, 1.0329]** at n = 40, 192x192x12 -- **not resolved**, the interval contains 1 (n = 12 at 256x256x16 agreed: 1.0356 [0.9457, 1.2083]). `work: identical across every run`, so the CRC and ray counts matched on every one of the 80 invocations. Moving the renderer into a static library and letting LTO cross that boundary changed neither the image nor the measured cost
- note: `docs/` was empty and is removed; `ROADMAP.md`'s links into `components.md` and `references.md` now point at the README sections that absorbed them

## Repository renamed ENGINE -> pathtracer

The repo was never an engine: it is a CPU path tracer, and the README has been titled `PBR Pathtracer` since it was written. `epochlab/ENGINE` is now `epochlab/PATHTRACER`, and the name is gone from everything a user or a build sees. The internals are deliberately untouched -- the `engine::` namespace, `include/engine/`, the `#include "engine/..."` lines and the `ENGINE_*` check macros are ~1600 of the 1709 occurrences and cost a whole-tree diff to change nothing observable.

- chore: `project(Engine)` -> `project(Pathtracer)`, `add_executable(engine)` -> `pathtracer`, and the 7 `target_*`/`set_target_properties` lines that name it. The binary is `./build/pathtracer`. `engine_git_sha`, `engine_check` and the `ENGINE_*` cache variables are build machinery, not branding, and keep their names
- chore: user-visible strings in `main.cpp`: the GLFW window title, the `usage:` line and the 7 `std::cerr` diagnostic prefixes
- chore: both of `perf_dashboard.cpp`'s branded lines -- the TTY frame header and `drawNonTty`'s `engine perf:` stdout prefix. `ENGINE` -> `PATHTRACER` is 4 columns wider, so the header's fill drops 50 `=` to 46 and the block stays on the 78-column grid it shares with `spec_report.cpp`
- note: **benchmark log schema change.** `-bench` records now carry `"tool": "pathtracer"`. Records logged before this say `"engine"`, so `bench_compare history` needs `--tool engine` to read them and will not group the two halves together. Nothing in `results/` is affected
- note: history above is left as written. Entries naming `engine::` symbols and `include/engine/` paths describe the API as it was at those commits, and it has not changed

## Wave 5, item: vsync toggle, display bit depth and the scene-texture data type in profile.json

The frame cap, the display texture's storage format and the storage precision of the scene's input images were all fixed in code. All three are now `profile.json` settings, validated at load like every other field, and measured before and after (`results/wave7`). The scene setting is the first step of a data type carried through the system: `engine::gfx::ScalarType` is its vocabulary.

- feat: `render.vsync` (bool). `false` skips `DisplayLink::waitForNextVblank()`; the one-frame-in-flight fence still bounds the GPU queue, and `DisplayLink` keeps running on its own thread, so `refreshHz` stays live. `pace_ms` is still timed and reads 0 uncapped. HUD `Cap vsync`/`off`, dashboard budget line `vsync`/`uncapped`
- feat: `render.displayBitDepth` (16 | 32) -> `engine::gfx::ScalarType` `Float16`/`Float32` (`scalar_type.h`), mapped to `GL_RGBA16F`/`GL_RGBA32F`. The GL internal format and bytes per texel both derive from the enum, replacing the hardcoded `GL_RGBA16F` and the `* 8` byte count. 8-bit is not offered: UNORM clamps scene-referred data to [0,1] before exposure. Non-integer values (16.5 would truncate to 16 through `get<int>`) and any other depth are rejected. Closes the README §5 Wave 5 "Texture bit depth (16/32) via JSON" item
- feat: `benchConfig` records `vsync`, `display_type` and `texture_type` (`render_beauty` records `texture_type`), since each changes the measured cost. Records logged before this have neither key, so `bench_compare` will not pair them with new ones
- feat: `render.textureBitDepth` (16 | 32, shipped 16) sets the CPU storage of the environment HDRI and every material texture. Shipped at 16 because every EXR this repo loads is half on disk, so it is lossless there and measured bit-identical; 32 is for a future full-precision source. `engine::gfx::ImageTexture` holds RGBA in `std::variant<std::vector<float>, std::vector<Half>>`, `Half` being the compiler's IEEE binary16 `_Float16`: AArch64 widens it with one `fcvt`/`fcvtl`, where Imath 3.2.2's `half` goes through a 256 KB lookup table on this target. `loadImageTexture` reads OpenEXR `HALF` or `FLOAT` slices straight into the chosen type, no float intermediate; a source that overflows binary16 is rejected at load rather than silently clamped (round-to-nearest overflows at the tie point one half-ulp above 65504, so the band just below it rounds to 65504 like any other value). `loadGltf` takes the type; `sampleBilinear(HdrImage)` had no callers left and is gone
- feat: `EnvironmentMap` builds its marginal/conditional CDFs through the same typed accessor it samples with, so the importance-sampling density stays proportional to the stored radiance at either depth (PBRT-v4 12.5)
- fix: the first cut dispatched each texture sample through `std::visit`, which libc++ lowers to an out-of-line function-pointer table (`__fmatrix`): `user_s` **1.027x [1.014, 1.032]** on cornell and 1.014x on the stump vs the pre-rework build, both resolved. A two-way branch on the variant's fixed index (`withTexels`) inlines the kernel; re-measured **0.994x [0.987, 1.002]** and **0.992x [0.973, 1.002]**, not resolved
- test: `io.profile_config_render_display_settings`, 19 rows over `vsync`, `displayBitDepth`, `textureBitDepth`, each varied against the shipped profile with the expectation derived from it (so the shipped depths can change without editing the test); 8, 24, 16.5, `"16"` and missing are rejected for each depth key. `io.image_texture_half_load`: binary16-exact data loads bit-identically at both types, arbitrary data at Float16 equals `static_cast<Half>` (round to nearest even), a 70000 texel is rejected at 16 only. `io.image_texture_bilinear_half_bound`: |s16 - s32| <= (2^-11 + 2 gamma_6) s32, derived from the convex bilinear weights and Higham's gamma_n, no fitted tolerance. `nee.environment_pdf_consistency` now runs at both types, and `nee.environment_pdf_tracks_stored_luminance` puts one texel at 1 + 2^-11 (a binary16 tie that rounds to 1.0) and checks the pdf ratio sits nearer the stored luminance than the source. Mutation-checked: a HALF slice read as FLOAT fails three load rows, a Float16-only off-by-one texel fails the bound at 7.4e9x. `ctest` **108/108**
- perf: **scene textures, `render_beauty`, 8 interleaved rounds, 2048x1152, 16 passes.** Every shipped EXR is half on disk, so Float16 storage is lossless: output CRC identical across pre-rework, 32 and 16 on both scenes. Peak RSS: stump **2102.5 -> 1280.8 MB**, measured saving **822.1 MB = analytic 822.1 MB**; cornell 481.4 -> 464.6 MB (the HDRI's 16.8 MB, also exact). `pass_ms` 16/32 **0.997 [0.934, 1.054]** stump, **0.993 [0.982, 1.015]** cornell, not resolved; stump wall per run **0.926x [0.899, 0.945]** from the lighter load
- perf: **scene textures in the engine, stump, 8 rounds of `-bench` at `maxSamples` 128.** Defaults vs pre-rework: no metric resolved (`pass_ms` 1.002 [0.976, 1.023]). Float16 vs Float32: `pass_ms` **0.978x [0.957, 0.988]**, resolved -- half the bytes per texel fetch; peak RSS 2952 -> 2129 MB, load 3.37 -> 2.76 s. Converged window captures byte-identical across all three. The HUD's RAM line is current resident memory, which macOS compression lowers for idle pages, so it understates the float32 footprint; peak RSS is the measure
- perf: 8-round randomized interleaved campaign, `maxSamples` 16, cornell 2048x1152, M1, 60 Hz. Paired Hodges-Lehmann ratio with exact Wilcoxon interval (96.1%). **Defaults vs `main`: no metric resolved** (`pass_ms` 1.0043 [0.9944, 1.0192], `upload_ms` 1.0007 [0.9662, 1.0361])
- perf: **RGBA32F vs RGBA16F: `upload_ms` 0.54x [0.50, 0.83], `present_gpu_ms` 1.44x [1.436, 1.438]**, frame time unchanged (vsync-bound). The driver's float-to-half conversion is about half the upload cost here (68% on the stump, below), which corrects README §5's assumption that a 16-bit display copy only saves transfer. GPU alloc 18.3 -> 36.3 MB (+2048x1152x8 B), RAM +15.5 MB (driver-side copy)
- perf: **uncapped vs vsync: 60 -> ~295 fps while converging, ~420 once idle (`frame_ms` 0.20x), `pass_ms` 1.58x [1.53, 1.61], `user_s` 1.29x.** The render thread spins on a core the 8 trace workers need, so an uncapped session converges ~58% slower. `fence_ms` 0.05 -> 0.68 ms: now partly GPU-bound, as expected
- note: **image change.** Defaults are byte-identical to `main` in the converged window capture, and uncapped is byte-identical to vsync. RGBA32F vs RGBA16F moves **3.61%** of pixels by exactly **1 code value** (max 1). Binary16 round-trip of the converged cornell beauty: max relative error **4.88e-4** (the 2^-11 bound), RMS 2.09e-4, no overflow, no flush to zero. The shipped HDRI is already half-precision at source (peak 11.1), so it round-trips exactly
- perf: **stump (local test asset, not in the repo: 20.5k triangles, six 4K half EXR maps), 8 rounds at `maxSamples` 128**: RGBA32F `upload_ms` **0.32x [0.31, 0.33]** (6.05 -> 1.92 ms), `present_gpu_ms` **1.44x**; uncapped `pass_ms` **1.47x [1.45, 1.48]**, 60 -> ~232 fps converging, 438 idle. Uncapped capture byte-identical; RGBA32F moves **0.22%** of pixels by 1 code value. No hang in 8 uncapped convergences
- note: the stump exposed two existing `-bench` gaps, not caused by this change, now README §5 items: at `maxSamples` 16 the interactive-scale accumulation converges inside the 0.25 s settle and is logged as the workload (width 205), and fast passes under vsync fail the one-record-per-frame contiguity check. The campaign ran at `maxSamples` 128, where every record is full-scale
- note: the GPU driver hang recorded under "B: First real draw" ("vsync re-enabled") came from GPU rasterization of the retired 5M-triangle stump with no frame fence; neither the asset nor that path exists today, so it does not constrain `vsync: false`

## Wave 2, item 1: CIE 1931 module and the measured-metal fit

`chrome.json`'s chromium came from Gulbrandsen eq 14/15 applied to Johnson & Christy's `(n, k)` at three hand-picked wavelengths (615/550/465 nm, not even `kRgbWavelengthsNm`'s 620/540/450). An RGB channel is a colour-matching-function integral, not a line, and eq 14/15 are nonlinear in `(n, k)`, so `(n, k)` cannot be averaged -- only reflectance projects linearly into RGB. Each channel's `(r, g)` is therefore a fit of Gulbrandsen's two-parameter family to the CIE-projected spectral Fresnel curve.

- feat: `engine/scene/cie.h`. CIE 1931 2° observer (CIE 018:2019 Table 6) and D65 (ISO/CIE 11664-2:2022 Table B.1) at 1 nm over 360-830 nm, in `src/scene/cie_1931.inc`, converted mechanically from CIE's own CSVs, whose SHA-256 match the checksums CIE publishes. `reflectanceToRec709` integrates under D65 per CIE 015:2018. `xyzToRec709` is derived from the BT.709 primaries and the *tabulated* D65 white by SMPTE RP 177, not pasted, so a perfect reflector maps to exactly (1, 1, 1). It is compiled into `metal_fit` and `colour_validate` only: no runtime consumer exists yet, and Kelvin lights will link it into `engine`
- feat: `tools/metal_fit --nk TABLE.csv`. `r` = CIE projection of `R(mu = 1, lambda)`, exact since the model's `R(1) = r` identically. `g` solves model average = CIE projection of the spectral cosine-weighted average `2 int R mu dmu`, the quantity the Kulla-Conty tint consumes; this is unique since `dR/dg >= 0` at every angle (eq 13), found by 53 halvings. It rejects rather than clamps a table that doesn't span 360-830 nm, is unsorted or unphysical, or a target outside the Gulbrandsen family. Prints floats at `max_digits10`, so a JSON paste is bit-exact
- refactor: `referenceConductorIor`/`referenceConductorFresnelAt`/`cosineAverageFresnel` move from `bsdf_validate.cpp` to `tools/conductor_reference.h`, shared by the oracle and the fit. It still never touches `src/scene/bsdf.*`
- fix: `chrome.json` `diffuseColour` `[0.552, 0.555, 0.558] -> [0.549619973, 0.55602181, 0.554227054]`, `edgeTint` `[0.555, 0.558, 0.672] -> [0.541698217, 0.569233477, 0.694220781]`. The old triples sat **2.4e-3 / 1.0e-3 / 3.8e-3** off the CIE normal-incidence target and **2.5e-3 / 1.3e-3 / 1.8e-3** off the average. The old "cool cast from the reflectivity leaning blue (+0.006)" was an artefact of the wavelength pick: integrated, blue reflectivity sits *below* green
- test: new `colour` suite (4 checks, fast/exact): CIE rows on their definitional normalisation points (`yBar` = 1 at its 555 nm peak, D65 = 100 at 560 nm), so an off-by-one row fails; RP 177 matrix reproduces all three BT.709 chromaticities and the white to 1e-12, which pins it completely; `chrome_matches_measured_chromium` asserts `chrome.json == float(fit)` bit-exactly *and* that the renderer's own float `fresnelAtViewAngle` reproduces the CIE targets within `2 FLT_EPSILON` (measured worst 7.5e-8), mutation-tested by reverting `chrome.json` to the old triples, which fails all 12 of its assertions; input rejection, including a spectral step that projects outside the family. `ctest` 104/104
- note: fit quality for chromium: two-parameter angular residual `max |R_fit(mu) - R_cie(mu)|` **3.5e-4 / 4.9e-4 / 2.4e-4**; interpolating `(n, k)` linearly in photon energy instead of wavelength moves `r` by <= 4.1e-5 and `g` by <= 3.0e-4
- note: **gold is out of gamut.** Across Johnson & Christy's metals, Cu/Ag/Ni/Fe/Ti fit (worst residual 9.6e-3, copper red) but Au's CIE red reflectivity is **1.038** in linear Rec.709, and `metal_fit` rejects it. New §5 Wave 2 item
- note: **image change.** `cornell.json` 96 passes, 640x360: linear RMSE **1.42e-3**, relMSE **5.61e-6**, confined to the chrome sphere. On the sphere the Fresnel AOV and Beauty move by the same per-channel ratios, **R x0.9956, G x1.0024, B x0.9965**. Structured change sits at the grazing rim, where the edge tint changed most; elsewhere it is sparse decorrelation speckle, RMS 1.06e-3 on a 0.30 mean
- note: stale prose fixed: `bsdf.cpp`'s and README §3's "idealised near-white mirror" (`chrome.json` has shipped the chromium triples since `8b47784`), README §5 Large #5a's "no colour-matching curves" and its line numbers. Closed "Albedo table accuracy" section removed from README §5 (recorded above)
- perf: none possible by construction. `render_beauty` A and B are byte-identical (compiled-source edits are comments only), so the 16-round interleaved A/B, **B/A 1.0108 [0.9779, 1.0488], not resolved**, measures machine noise. The asset change's only work effect is Russian roulette on a slightly darker sphere: bounce rays -1.3e-4, shadow rays -9.0e-5

## Albedo table accuracy: measure the interpolation, then warp and resize the axes

README §5's open item named the `sqrt(mu)` axis warp and a 7.1e-3 first `mu` bin as the next levers. Both numbers came from an ad-hoc measurement no code reproduced, so the first move was to build the instrument -- and the instrument corrected the premise before any table changed.

- test: `albedo_table_interpolation` reports **four worsts separately** -- each axis with the other held on exact nodes, the first `mu` cell split from the rest of its axis, and `Eavg`'s own 1-D lerp -- plus a **control row** on exact nodes where no interpolation happens at all. A single combined worst would hide which lever moved; the control exists so the instrument's own error is a printed number rather than an assumption. Needs `directionalAlbedoSplit`/`averageAlbedoSplit` and the grid accessors exported, under the rule that already exports the average-Fresnel pair for `checkAverageFresnel`
- fix: **the control immediately read 5.2e-3, and the fault was the reference, not the table.** A convergence probe (`results/wave6/probes/reference_convergence.cpp`) settled it: Simpson converges *toward* the committed value as panels rise, and Gauss-Legendre at 96 nodes matches it to 2.7e-9. `referenceDirectionalAlbedo`'s `psi` substitution flattens the NDF peak but compresses the other end -- `wi.z` falls from O(1) to 0 over an O(1) span of `theta_h`, which maps to a span of `psi` narrower by `~alpha*mu` -- so at `mu = 1/127` the horizon layer is ~1e-3 wide against a 96-panel width of 1.6e-2. Measured convergence order **~1.2**, with the **entire** error of a 96-panel Simpson in its last panel pair; 3072 panels still leave 3.6e-5. Gauss-Legendre puts its outermost node `O(1/n^2)` from the endpoint, inside the layer: **5.2e-3 -> 1.5e-7** at 192 nodes. No closed-form breakpoint substitutes -- the natural candidate, `wi.z = mu`, lands at an eighth of the layer's width
- fix: the reference also divided by `mu` to recover `E` from an integral proportional to it, amplifying relative error by `1/mu`. Unusable once the warped axis reaches 6.2e-5 and catastrophic at the 2.5e-6 the first cell is sampled at -- it read `E` as **37.4**. Folded into the closed `G2/cosO` form, which also removes a `max(cos^2, 1e-8)` clamp and makes `E(0) = 1` directly evaluable
- note: the recorded numbers were wrong twice over. The first `mu` cell was **4.2e-2**, not 7.1e-3; and the "3.2e-5 away from it" was the **roughness** axis, not `mu`, which is ~1e-5 there
- fix: reflect side **128x128 uniform -> 256x256**, `mu` uniform in `sqrt(mu)` as `escapeMu` already was one axis down. `kAlbedoRes` splits into `kAlbedoRoughnessRes`/`kAlbedoMuRes`/`kMsReflectMuRes`, the last deliberately its own constant so a later warp-resolution change cannot silently resize a sampling density
- fix: **a warp moves cells, it does not add them**, and the intermediate 256x128 bake proved it rather than the argument. Cell width in `mu` becomes `2 sqrt(mu)/(res-1)`, so at `mu = 0.4` cells are **1.27x** a uniform axis', and `coat_fresnel_average` -- whose worst rows all sit at `mu` 0.4 -- went **5.1e-5 -> 8.7e-5**, a regression on the instrument the table is read by. 256 `mu` nodes buy that band back at 0.63x the original width and quarter the grazing layer at once
- fix: **`mu = 0` is a real node**, not `gridMu`'s nudge off it. `E(0, alpha) = 1` exactly for every `alpha` -- at `mu = 0` the integrand collapses to `2 cos(phi) sin^2(psi)`, whose normalised integral is 1 -- and the bake **asserts** it on all 256 rows, so the column that used to be the table's worst is now its sharpest. `reflectAlbedo` takes the radical `smithG2OverCosO` form to reach it: mandatory rather than stylistic, because `smithLambda`'s `max(cos^2, 1e-8)` clamp bites once node 1 is `mu = 6.2e-5` and would substitute a different cosine on the grazing rows with no diagnostic
- fix: `buildMultipleScatteringShape` reads `E` through the warped axis via `reflectAtUniformMu`, the reflect twin of `escapeAtUniformMu`. Its own grid stays uniform in `mu`, where `invertPiecewiseLinearDensity` has one step width
- test: measured, before -> after: first `mu` cell **4.2e-2 -> 1.8e-3** (23x); `mu` axis cells 1..n **6.9e-3 -> 2.2e-3**; roughness axis **1.1e-3 -> 5.9e-4**; `Eavg` lerp **1.9e-5 -> 4.6e-6**. The control holds at **3.0e-5**, the stored values' own quadrature residual, agreeing with what `verifyReflect` independently prints
- test: `coat_fresnel_average`'s tolerance **2e-4 -> 6e-5**. Its worst row went 1.2e-4 -> **5.1e-5 on the reference fix alone, table untouched**, then -> **3.5e-5** on the table. So the ~7e-5 the old comment could not attribute to either the table or `dielectricFresnelAvg` was an instrument error, not a model one -- it leaked in at exactly the `mu = 0.4` rows the worst cases sit on
- test: `white_furnace_two_sided`'s off-grid rows re-placed for the third time, now at the exact floats landing on index `k+0.5` of the axis that indexes each -- roughness `(k+0.5)/255`, `mu` `((k+0.5)/255)^2`, so the lookup's blend weight is 0.5 on both. Nothing derives them from `albedoGridRes()` on purpose: a row that recomputed its own worst case would follow the grid wherever it went and could never be seen to go stale
- note: an alpha-dependent warp (Smith `G1` as the axis coordinate) was **measured and rejected**, not overlooked. It resolves each row's own layer exactly, but at `alpha` 0.25 it compresses `mu` in [0.3, 1] into 12% of the axis -- 5.3e-4 against a uniform axis' 1.2e-5 in exactly the band `coat_fresnel_average` reads, where `E` moves 0.844 -> 0.916 and is not flat. Two extra `sqrt` in a hot lookup to buy a 40x regression
- note: **what remains.** All three directional worsts sit on the lowest-alpha rows (roughness <= 0.022) at `mu <= 4.4e-3`, where the lobe is a near-mirror and the layer is narrower than a cell however the axis is warped -- sub-degree grazing, cos-weighted to ~1e-8. The quadrature residual, 3.0e-5, is the floor everywhere else regardless. The generator's `psi` rule converges algebraically rather than geometrically in those same columns; the printed reflect residual did not move (2.97e-5 -> 3.05e-5), so the panel split at `wi.z = mu` stays out of scope
- note: **image change.** `cornell.json` at 96 passes, 640x360: **linear RMSE 2.19751e-4, relMSE 1.67028e-7**
- perf: interleaved A/B, 24 rounds, `--metric pass_ms`: **B/A = 0.9998 [0.9934, 1.0041], not resolved at n = 24** -- the added `sqrtf` in `directionalAlbedo` and the 4x reflect table are both under the noise. A 12-round run of the same pair reported "B slower by 2.04% (resolved)" on a contended machine, with `nivcsw` 2.7x higher; the wider interval was saying so, and the extra rounds are what the repo's own rule exists for
- note: cost. `.inc` **9.07 MB -> 11.0 MB**, bake **4m25s**. `ctest` 100/100

## Coat Fresnel average: one quadrature rule for both interfaces

README §5's first roadmap item asked for a better `F_avg` function, not a better instrument -- `checkCoatFresnelAvg` already recovered the coat's `F_avg` to 6.6e-5, and spent its whole 0.0035 tolerance on `dielectricFresnelAvg`'s two-constant rational fit. The conductor half of the same file had already solved the identical problem with the right device: a 3-node quadrature rule `sum(w_i*F(mu_i))` over the same Fresnel its own single scatter evaluates. The question was only whether those nodes also serve the dielectric family. They do, unrefitted, so this adds no constants at all.

- fix: `dielectricFresnelAvg` is that rule over `fresnelDielectric`, sharing `kFresnelAvgNodes`/`kFresnelAvgWeights` with `conductorFresnelAvg`. Measured worst **5.5e-5 over ior in [1.05, 3.0]** (at 1.0575) and 4.5e-5 over [1.1, 3.0], against **6.5e-3** for the rational fit it replaces (at 1.17) and 2.5e-2 for Karis' Schlick mean (at ior 1.1) -- **118x tighter than the fit's own worst case**. At the iors that ship: **4.1e-6** (`glass.json` 1.5168) and **3.3e-6** (`clay.json` 1.55), against 2.0e-3 and 1.5e-3
- fix: uniformity, which is what actually unblocked the check. The rational fit was worse than the Schlick mean it replaced below ior ~1.42, so `checkCoatFresnelAvg` could only sweep 1.5-1.8 and still separate the two. The rule beats both everywhere except two windows: `ior <= 1.0046`, and a **4.4e-4-wide** sliver at 1.6518 where the fit's own error crosses zero and any fit is momentarily exact -- the rule is still within 3.0e-6 there
- fix: exactly `+0` at ior 1 **structurally**, not by an algebraic accident of a numerator. Every node evaluates `fresnelDielectric` at `etaI == etaT`, where `etaI/etaT` is exactly 1, `cos2Transmitted` returns `mu*mu`, `sqrt(mu*mu)` is exactly `mu`, and both polarisations are an exact `(c-c)/(c+c)`. The mean and `coatFresnelRatio` now collapse for one reason rather than two, so `checkIndexMatchedCoat`'s tolerance-exactly-zero contract cannot be lost to a refit of the mean. Weights sum to **exactly 1.0F** in float32, so each rule is a convex combination of `F` values and cannot leave [0, 1] at all
- fix: **an instrument bug the widened sweep exposed.** `checkCoatFresnelAvg` bisected the coupling on a fixed `[0, 0.6]` bracket, but `referenceCoupling` is unimodal in `fresnelAvg`, not monotone: `coatAlbedoAvg`'s `fresnelAvg/karisAvg` rescale eventually drives `1/(1-coatAlbedoAvg)` back down. The turning point is **inside** that bracket at every ior swept -- measured 0.104 at ior 1.1 rising to 0.419 at 3.0, always at roughness 1 and normal incidence -- so a root on the rising branch could still be answered with the bracket's ceiling. It only stayed hidden because the old 1.5-1.8 sweep happened to land where `C(0.6)` still exceeded the measurement. The bracket is now the model's own monotone branch, located by ternary search
- test: `average_fresnel`'s dielectric tolerance **0.0065 -> 1e-4**, over a sweep extended from 7 iors to 14. The `ior -> 1` boundary layer gets its own named band at 8e-4 rather than loosening the working one 8x over a region no material occupies: `F(0) = 1` for every `ior > 1` but collapses over a width `~sqrt(ior-1)`, which three fixed nodes cannot resolve, measured worst **5.9e-4 at ior 1.0057** where truth is itself 1.8e-3. It reaches `coatAlbedo` scaled by ~0.054, so under 3e-5 on the coupling
- test: `coat_fresnel_average`'s tolerance **0.0035 -> 2e-4**, and its sweep widens from 4 iors to 9 (1.1 to 3.0), 144 rows to **324**. Worst recovered error **1.2e-4** at ior 3.0, 7.0e-5 at 1.5. The check has changed job: `F_avg`'s own contribution is now 3.3e-6 to 2.2e-5, two orders under what it can resolve, so what it measures is `src/scene/albedo_table.inc` and the inversion. A table regeneration that loses accuracy trips here first
- test: the conditioning is a printed `dC/dF` column rather than a remembered constant in a comment. Measured **0.81 to 0.91** across the sweep. This is what caught the bracket bug -- ior 1.1 reported a *negative* slope at the pinned ceiling on its first widened run
- test: all three mutations re-measured, each failing at **every** ior and on value rather than on form (the residual fires on none of them). Karis' `schlickFresnelAvg(coatF0)` recovers 0.0856466 against that function's own 0.0857143 -- the instrument still names it, to 7e-5 -- and fails by **31x at ior 1.5, 123x at 1.1**, 3.5x at its weakest (ior 2.5, where Karis crosses truth); it used to fail by only 1.8x. `2*dielectricFresnelAvg` fails by **126x to 1381x**. The rational fit itself, which this check used to **pass**, now fails by **12x at ior 1.5 and 30x at 1.1** -- `average_fresnel` also forbids it now, by 64x, so the revert is no longer reachable through either instrument
- perf: `diffuseKdAt` recomputed `dielectricFresnelAvg(params.ior)` on every `wi`, duplicating the `dielectricAvg` `computeLobeProbabilities` already held. Hoisted onto `LobeProbabilities` as `coatFresnelAvg` -- distinct from `fresnelAvg`, which is the metallic-blended mean and wrong for the coat at any `metallic > 0` -- so the net cost of the change is **one** rule evaluation per `evaluateBsdfSplit`/`sampleBsdf` against two divides before. Landed and verified separately at **linear RMSE 0, relMSE 0**
- perf: interleaved A/B on `cornell.json`, 11 rounds, `--metric pass_ms`: **B/A = 1.0063 [0.9995, 1.0136] at 95.8% exact coverage, not resolved at n = 11**, median 150.8 -> 152.1 ms. The honest bound is the interval's, under 1.4%. `work: DIFFERS` is correct here and not nondeterminism -- the image genuinely changed, which is the point
- perf: `coat_fresnel_average` runs **11.225s -> 0.317s**, 35x. `referenceDirectionalAlbedo` is a 96x96 composite Simpson and `referenceAverageAlbedo` a 64-panel Simpson over it, and both depend only on `(mu, roughness)`; the check evaluated them once per row for 4 distinct roughnesses and 3 distinct cosines. Hoisted ahead of the ior loop, which is what made widening the sweep to 324 rows free. Printed output byte-identical, timing line excepted
- note: **image change.** `cornell.json` at 96 passes, 640x360: **linear RMSE 0.00138812, relMSE 5.79546e-06**. `clay.json`'s ior 1.55 is every wall, the floor and the ceiling, and `F_avg` there moves +1.5e-3, which at the measured coupling slope of 0.81 is ~0.1% on the diffuse channel -- about a sixth of the Karis revert's effect, and in the opposite direction at this ior. `glass.json`'s 1.5168 sphere moves with it
- note: the exact closed form of the integral (d'Eon & Irving 2011) was **measured against and rejected**, not overlooked. It would re-derive the interface in a second place, which is the desynchronisation `fresnel_dielectric.h` exists to prevent, one level up; its accuracy is unobservable against an instrument that is itself table-limited at ~1e-4; and its `log((ior-1)/(ior+1))` and `1/(ior^4-1)` terms cancel catastrophically in exactly the `ior -> 1` band the rule is weakest in, trading a measured 5.9e-4 for an unmeasured one plus a Taylor branch and a special case to recover the zero the rule gets for free. Two `log`s also cost more than three `sqrt`s
- docs: **README §5's coat `F_avg` item is closed** and replaced by what outlives it -- albedo table accuracy, with `checkCoatFresnelAvg` as its instrument. The first `mu` bin's 7.1e-3 bilinear error and the `sqrt(mu)` axis warp are the named next levers. §6's Gulbrandsen entry records that the same rule now serves both interfaces, with the dielectric's measured bound and the rejected closed form

## Roadmap: the transmissive sphere check's normal-incidence blind spot

- docs: README §5 Wave 3 gains the item. `transmissive_sphere_energy` scores through `renderCentre`'s centre 4x4 block, so a sphere is asserted at `mu` 1 and nowhere else -- curvature is the check's reason to exist and the centre is the one point on a sphere where it costs nothing.
- note: **measured, not inferred.** Widening the probe camera to bring the silhouette into frame (`kFocalLengthMm` 200 -> 40, `kImageSize` 16 -> 192) and binning the disc radially: roughness 0.2 reads **0.9450 at 90% of the silhouette radius and 0.9020 at 95%** while its centre reads 0.9989 and passes the 3% tolerance. Roughness 0.4 reads 0.9874 / 0.9813; 0.02 holds 1.0000 through 90% and 0.9935 at 95%, 0.7 holds 0.9985 / 0.9971. Non-monotonic in roughness -- worst at 0.2, absent by 0.7 -- so a single grazing point sample resolves nothing and the sweep is the test.
- note: no shipped material shows it. `glass.json` is roughness 0.02, below `kSmoothAlpha`, so it takes the delta branch; the item gates authoring rough glass.
- note: the dark grazing band on the Cornell glass sphere is **correct optics, not this defect**, and was ruled so by measurement rather than by reading: the uniform-environment invariant holds flat to 1.0000 across the inner 90% of the disc at roughness 0.02, the chrome sphere shows the expected bright rim on identical geometry, and the band tracks refractive deviation past 60 degrees against a Fresnel term still under 0.35. Raising `maxBounces` 8 -> 32 leaves it unchanged, and neutralising `transmissionOffsetEpsilon` deepens it, so neither is its cause.
- note: **no image change.** Documentation only.

## Wave 4 closed: AOV switching no longer restarts the image, and Fresnel became progressive

README §5 Wave 4 was one enabler and two consumers. The enabler was that `aovNeedsLightTransport` and the trigger-state comparison forced a full progressive-accumulation restart on every AOV switch -- across producers, between two light-transport AOVs, and even into the four GPU post-filters, which never read anything but `beauty`. Measured before the fix on a five-stage schedule at 512x288 / 32 samples, the three switches that touched the path tracer cost 12466, 13217 and 12350 ms; the one into a rasterizer AOV cost 16 ms.

- feat: `engine -bench-aovs A,B,C` walks a named `AovId` sequence, one stage per entry, timing each switch into a `stage_wall_ms` samples column that `bench_compare` reads as a mean per event with no change to that tool. Stage 0 is an unmeasured warm-up -- `restart()` still clears every column while it runs -- so each measured switch starts from a converged image and a run without the flag logs exactly the record it always did. `finishBench`'s pass contiguity check is now per generation rather than across the whole column, since a schedule spans several accumulations
- fix: the selected AOV is out of both producers' trigger keys. One pass writes every image `PathTraceResult` owns and one call writes every image `RasterGBuffer` owns, so which AOV is displayed cannot change what either must compute. `ViewInputState` factors out the camera/framebuffer geometry that is the rasterizer's entire dependency set and the leading part of the path tracer's, and each producer now compares its own key: an environment change no longer re-rasterizes a G-buffer that does not depend on it, and selecting a rasterizer AOV whose G-buffer is already current costs nothing
- fix: `setSuspended` no longer bumps the generation. The generation is both the restart signal and the sampler's scramble seed, so the bump -- there to cancel the pass in flight -- discarded the accumulation with it, and a converged image cost a full restart just to look at a G-buffer AOV and come back. The in-flight pass now runs to completion and is accumulated: at most one pass of latency, for work that is valid
- perf: the display texture is keyed on the image rather than the `AovId`, so `beauty` and the four post-filters over it share one upload. Cycling HSV/Luminance/Sobel/Gabor is now a shader uniform and a blit
- perf: randomized multiple interleaved trials over the schedule above, 11 rounds, `--metric stage_wall_ms`: **paired B/A = 0.0022 [0.0022, 0.0023] at 95.8% exact coverage, resolved** -- median switch cost 9787 ms -> 21.9 ms. Per stage, **13683 / 13834 / 17 / 13933 ms -> 16.6 / 16.6 / 16.7 / 41.8 ms**: every switch is now one frame, including the one into a rasterizer AOV, which was already. Accumulated passes 128 -> 32 and traced rays exactly 4x -> 1x, since three of the four switches no longer re-converge at all; median involuntary preemptions 330743 -> 88594
- note: `bench_compare` reports `work: DIFFERS` on this pair, and it should. Each build's CRC is identical across its own 11 runs, so neither is nondeterministic; they differ from each other because A's final image is its fourth accumulation and B's is its first, and the generation is the scramble seed. Same estimator and same sample count, a different noise realisation
- test: `driver.suspension_halts_and_resumes` asserted the discarded-accumulation contract and now asserts the kept one -- same generation, count continuing. `driver.suspension_preserves_the_accumulation_exactly` is new and pins a parked-and-resumed run bit-identical to an uninterrupted one, which is what makes the park free rather than merely cheap
- note: **no image change.** The relaxation changes when the renderer restarts, never what a pass computes. A before/after CRC of one accumulation will differ where a schedule is involved, because the build that restarted reaches its final image under a later generation and so a different scramble seed -- a different noise realisation at the same sample count, not a different estimator

The second consumer was the `Fresnel` AOV, which was rasterizer-only: one primary-hit sample at the pixel centre against the macro normal, not progressive.

- feat: `Fresnel` moves from `RasterGBuffer` to a tenth `PathTraceResult` accumulator lane, and from `F(n.wo)` to the expectation of Fresnel over the visible microfacet normal distribution, `E[F(wo.wh)]` for `wh ~ D_vis(wo)` -- one VNDF draw per sample (`fresnelAtMicrofacet`, `bsdf.h`) from the same `D_vis` and the same alpha `sampleBsdf` draws from, through the same `fresnelAtViewAngle` inversion. That is the angle the microfacet BSDF evaluates Fresnel at (Walter et al. 2007; Heitz 2018 for the distribution; Karis 2013's split-sum as the precedent for reporting the expectation as a term of its own)
- feat: it draws from a third independent sampler stream, `scrambleSeed ^ kFresnelSeedOffset`, exactly as the AO lane does. Taking dimensions from the path's own sampler would shift every later dimension and move all nine pre-existing images
- fix: `alphaForRoughness` replaces two copies of the perceptual-to-GGX mapping in `bsdf.cpp` and is now the single source `evaluateBsdfSplit`, `sampleBsdf` and `fresnelAtMicrofacet` share, so the AOV cannot report a Fresnel the lobe never evaluated
- test: `bsdf.microfacet_fresnel` asserts both halves of the claim: the smooth limit reproduces `fresnelAtViewAngle` to under 1e-3 (a strict generalisation, not a different quantity), and at roughness 0.6 grazing the expectation falls below the macro value. A roughness-blind implementation passes the first and fails the second
- note: **image change, this AOV only.** The lane is full RGB where the rasterizer packed `(F, 1-F, 0)`, so a dielectric now reads greyscale rather than a red/green ramp, and an edge-tinted conductor reads chromatic -- `edgeTint` inverts to a per-channel complex IOR (Gulbrandsen 2014), which the `.x`-only packing discarded. The values themselves are roughness-dependent where they were not: on an `ior` 1.5 dielectric at `n.wo` 0.05 the lane reads 0.7521 at the roughness floor, 0.4406 at roughness 0.3 and 0.1692 at 0.6, against a macro-normal 0.7521 throughout. Back-facing pixels follow `bsdf.cpp`'s own orientation convention instead of the rasterizer's `max(dot(n, wo), 1e-4)` clamp, and a primary ray hitting an emitter quad now reads 0 like the other path-traced lanes rather than a surface value
- perf: the lane costs one VNDF draw and one Fresnel evaluation per camera sample at bounce 0 -- no rays, no BVH query, nothing per bounce. Interleaved A/B, 11 rounds, `--metric pass_trace_ms`: **B/A = 1.0294 [0.9784, 1.0493], not resolved at n = 11**, median 408.8 -> 420.5 ms. So the cost is real but below what this many rounds can separate from run-to-run drift; the honest bound is the interval's, under 5%. `work: identical across every run` on the same pair, which is the ray counts and the beauty CRC agreeing exactly
- note: the other nine lanes are **bit-identical** across the change -- `render_beauty --compare-exr` reports linear RMSE 0, relMSE 0 -- which is the property the separate sampler stream exists to give
- docs: **README §5 Wave 4 is closed.** The contact sheet stays parked, and is now the one item that genuinely needs the rasterizer and path tracer running in the same frame: Wave 4 removed the restart cost of switching between them, not the mutual exclusion itself, which is why the driver is still parked while a rasterizer AOV is shown

## Wave 1 closed: the index-match item answered by measurement

README §5 Wave 1's last item was the transmit lobe's discontinuity across `eta = 1`: `transmissionIsRough` routes an index-matched interface to the delta branch while the tabulated escape deficit either side does not tend to zero (0.20865 at `eta` 0.999, 0.21014 at 1.001, roughness 1). Energy is 1 on both sides, so this was always a distribution question, and it is now answered rather than argued.

- docs: total-variation distance between exit distributions and a Heitz et al. 2016 stochastic Smith-microsurface walk, over `eta` 1.01 to 1.5 at roughness 0.3 / 0.6 / 1.0 and `mu` 1 / 0.7 / 0.3. Kulla-Conty is the best of the three models at `eta` 1.5 (0.052, against 0.061 for single scatter alone and 0.081 for Turquin 2019 albedo scaling), the three cross at 1.33, and below that the compensation's shape costs more than the energy it returns: at 1.01 it reads 0.076 against 0.040 and 0.021. The reference's multiply-scattered share is 13% of exit energy there on average and 72% at roughness 1 `mu` 0.3, collapsing toward `-wo`, which a `(1-Escape)cos` row cannot express at any scaling -- so the fix is a reference-accurate lobe, not another scaling, and switching models by `eta` would trade one arbitrary shape for another
- docs: the same sweep prices the Beta-form transmission G2 (Heitz 2014's B(1+Lambda_o, 1+Lambda_i), which the lobe does not use and which overstates first-order transmission by up to 80%): adopting it alone moves mean TV from 0.066 to 0.098, because the energy it correctly removes from single scattering lands in the approximate compensation shape. It waits for that shape
- docs: both findings move to README §5 as "Smith-exact transmissive multiple scattering", not sequenced by wave. §6 cites Turquin 2019 with its numbers, names the Heitz 2016 walk as the reference instrument, and records Walter 2007 sec. 5.3 as the source of the per-facet Fresnel split
- docs: **README §5 Wave 1 is closed.** The coat `F_avg` item that sat beside it keeps its own section

## Transmissive selection by energy, per-facet Fresnel, and a grazing-resolved escape axis

README §5 Wave 1: the transmissive path chose its strategy from macro Fresnel times the opaque E, clamped to [0.05, 0.95], and capped `msTransmit` at 0.75 of the transmit mass -- three numbers standing in for the energies the lobes actually carry. Per-vertex throughput sigma inside white rough glass was 0.2 to 2.1, and the BSDF-only slab walk's per-path sigma was 1.41 / 2.49 / 3.28 at roughness 0.4 / 0.7 / 1.0.

- fix: every strategy's selection mass is now its energy at wo over their total, in flux terms (no eta^2, as PBRT-v4's `DielectricBxDF` selects R/T, since the compression cancels over the round trip): VNDF single scatter `R_ss`, the opaque multiple-scattering share `(1-tw)*fms*(1-E)`, the transmissive reflected share `tw*(1-ts)*D`, diffuse, refraction `tw*T_ss` and `msTransmit` `tw*ts*D`. The clamp, the cap and the F*E heuristic are gone; a strategy gets zero mass only where its value is zero, which is the support condition the estimator needs
- fix: on a rough transmissive interface the VNDF strategy now refracts about the sampled facet as well as reflecting about it, split by that facet's own Fresnel (Walter et al. 2007 sec. 5.3; PBRT-v4), so each branch weighs `G2/G1` instead of the macro split's `F(wo.h)/R_ss`. The selection variate is reused for the split, so no sampler dimension is added. `evaluateSpecularLobe` and `evaluateTransmissionLobe` carry the matching factor in their pdfs
- feat: `msReflectTransmissive`, the reflected multiple-scattering share's own strategy, drawn from the forward-eta escape-deficit row its value already reads. That row is hoisted into `LobeProbabilities` beside the reciprocal one, so value, pdf and sampler share one interpolant and `evaluateSpecularLobe` stops rebuilding it per call
- note: selection ignores `transmissionTint`, so a tinted estimate stays exactly the white one times the tint (`transmission_tint`, `rough_transmission_tint` assert that exactly). A dark tint therefore spends draws on paths it then attenuates: measured as the one regression in the variance table, per-path sigma 0.124 -> 0.147 at roughness 0.15 on a (0.9, 0.5, 0.2) tint, against 0.37 -> 0.15 at 0.4 and 0.82 -> 0.20 at 1.0
- fix: the escape tables' mu axis is now uniform in `sqrt(mu)` (`escapeMu`), and the transmit integrand carries `G2/cosO` in the clamp-free radical form, so node 0 is the exact grazing limit `2/alpha` rather than a nudge to mu = 1e-3. E climbs from that limit over mu ~ alpha, which one uniform cell could not resolve. Measured against the generator's own exact quadrature at ior 1.5, worst error below mu 0.05: roughness 0.05 4.2e-2 -> 2.6e-2, 0.15 1.0e-2 -> 3.4e-3, 0.4 1.2e-3 -> 4.1e-4, 1.0 8.3e-3 -> 3.6e-4. The reflect side keeps its Lambda-form G2, so the opaque arrays are bit-identical
- note: what is left is the roughness axis, not mu: at a roughness that lands on a table node the same measurement reads 5.7e-4 at 0.16, against 3.4e-3 at 0.15 between nodes, because alpha changes by half between adjacent rows down there. Away from grazing the floor is the eta and roughness interpolation, 3e-4 to 6e-4
- test: `transmissive_slab_walk`'s band is now the Student-t half-width plus the table's interpolation floor times the walk's own measured vertices per path -- the null is "closure to the table's resolution", which is what the model states, and without it the check rejects that resolution as an error as soon as variance drops below it. Break: scaling the transmitted escape by 1.005 fails all three roughnesses (0.9925 / 0.9916 / 0.9933 against +/-0.0023 / 0.0028 / 0.0033)
- test: `transmissive_energy_balance` accumulates in double (a float sum of 200k throughputs near 1 has an ulp of 0.008 and rounds a clustered fraction down every time, a systematic 1e-3). It keeps its fixed 0.02 rather than a replicate-derived band, because what is under it is deterministic model error and not noise: worst 0.0036 at transmissionFactor 0.5 entering, the coat coupling's averaged rescale, and 0.0012 on the fully transmissive rows, the escape table between its nodes
- test: `transmissive_sphere_energy` keeps its 0.03 for the same reason and because its own residual is now small enough to measure: at 8192 spp the worst row reads +0.135% at roughness 0.7, against +1.04% before this change
- note: break tests, all at the final state. Dropping the new strategy's pdf term: `sampling_chi_square` p = 0 on the transmissive rows, `transmissive_energy_balance` to 1.88. Dropping the per-facet factor from the reflection pdf: chi-square fails, energy balance to 0.94. Weighting the rough-transmission miss at 1.0: `transmissive_slab_energy` +0.197 / +0.279 at roughness 0.7 / 1.0 against a band of +/-0.011 / 0.015, three times tighter than the same break measured before this change
- perf: per-vertex throughput sigma on white glass at ior 1.5, worst over mu on the exiting side, 2.10 -> 0.66; the slab walk's per-path sigma 1.41 / 2.49 / 3.28 -> 0.20 / 0.55 / 0.76 at roughness 0.4 / 0.7 / 1.0, which is 19x to 50x fewer paths for the same error, and 0.14 -> 0.027 at 0.15. Closure holds: the walk reads 1.00005 / 1.00035 / 1.00011 / 1.00003
- perf: render RMSE at 96 passes against each build's own 1024-pass reference, 640x360: cornell rough glass 0.5 0.0501 -> 0.0324 and 1.0 0.0451 -> 0.0311, which is 2.4x and 2.1x fewer passes for equal error; roughness 0.15 -1.0%, ior 1.05 -4.7%, delta glass unchanged. Per pass the renderer pays for it: interleaved A/B (`bench_compare`, 10 rounds, rough glass 0.5, 32 passes) resolves pass_ms B/A at 1.0259 [1.0149, 1.0705], so equal error arrives about 2.3x sooner
- perf: per call over 200k random direction pairs, min of four interleaved runs: evaluate+pdf -13% to +5%, `sampleBsdf` -1% to +7%
- note: **image change.** Against the previous commit at 96 passes, whole frame: opaque cornell byte-identical (the reflect-side table and code paths are untouched); rough glass 0.15 / 0.5 / 1.0 +0.011% / +0.045% / +0.069%, ior 1.05 +0.016%, delta glass -0.0005%. The rough-glass shifts are the grazing escape no longer over-reporting, so the compensation returns slightly more near silhouettes

## Transmit escape table converged; slab checks become instruments

README §5 Wave 1: "white slab 2.7% over" was Monte Carlo noise, not over-energy. `transmissive_slab_energy` averaged the centre 4x4 px at 512 spp from one seed, ~3% standard error against a hand-set 0.03; the same check read 0.984 at roughness 1 at 8192 spp. A BSDF-only plane-parallel walk (8M paths, SE ~0.001) put the real defect the other way: **energy loss**, 0.988 / 0.994 / 0.991 at roughness 0.4 / 0.7 / 1.0. Its cause was the transmit escape table: first-order midpoint quadrature (16x16 stratified per cell) and linear interpolation on a 32-node mu and 16-node eta axis, both of which bias the escape total and so the Kulla-Conty deficit it funds.

- fix: `tools/albedo_table.cpp` integrates the transmit side by Gauss-Legendre in the NDF measure (tan theta_h = alpha tan psi, D cos dw = sin psi cos psi dpsi dphi / pi), panelled at every kink of the integrand in closed form -- visibility, the reflection horizon and TIR onset in theta_h, the TIR tangency in phi -- so each panel is smooth and the rule converges spectrally. 48 nodes per panel; `verifyTransmit` rebuilds at 96 and prints the residual: 3.2e-4 worst, confined to the mu = 1e-3 column, 3e-6 mean. `--transmit-nodes` replaces `--samples`; the float VNDF sampler it drove is deleted
- fix: the table splits `kTransmitRes` into `kTransmitRoughnessRes` 32 and `kTransmitMuRes` 64, and `kEtaRes` goes 16 -> 64 (log-spaced). Per-vertex energy on a white ior-1.5 interface: 32 mu nodes lose 2% at mu 0.02, 32 eta nodes lose 9e-4 midway between nodes against 1e-4 on them; 64 x 64 closes to within 3e-4, and roughness keeps 32, where node and midpoint already agree to that level. `bsdf.cpp`'s escape, average and msTransmit row lookups index with the split constants
- test: `bsdf_validate` `transmissive_slab_walk` (new, Slow): `fixtures::slabWalkLo`'s BSDF-only walk at roughness 0.4 / 0.7 / 1.0, 16 replicates x 2^20 paths, Student-t band about 1, plus zero depth-cap truncations. After: 1.00023 / 0.99895 / 1.00086 vs +/-0.0025 / 0.0047 / 0.0103. Break (the old table): fails 0.98762 at 0.4 and 0.99291 at 0.7; 0.99225 at 1.0 is inside its +/-0.0099 -- per-vertex throughput variance, not the band's derivation, limits it there
- test: `integrator_validate` `transmissive_slab_energy` holds the Embree render to the same walk rather than to 1.0, so a table error cannot read as integrator bias: 16 scramble-seed replicates each side, whole-image mean (every pixel sees the infinite slab within 3.5 degrees of normal), `differenceBand`. Break (rough-transmission miss weighted 1.0, the double count it exists for): fails at 0.4 / 0.7 / 1.0 with +4.9% / +21% / +25%. The old check on the new table read 1.0395 at roughness 1 -- the noise it passed on before
- note: `transmissive_energy_balance` worst |1 - Lo| 0.0081 -> 0.0036 (roughness 0.4 exiting -0.9: 0.99268 -> 0.99957). Its fixed 0.02 and `transmissive_sphere_energy`'s 0.03 stay until the next change: at today's per-vertex throughput sigma (1.4-2.1) a replicate-derived band would be looser than the tolerance it replaces. Sphere at 8192 spp: 0.9975 / 0.9872 / 1.0043 -> 0.9988 / 0.9989 / 1.0104 at roughness 0.2 / 0.4 / 0.7
- note: **image change.** Against the previous commit, 640x360, whole frame: opaque cornell bit-identical; cornell (delta glass) 8596 px move by float rounding, -1.8e-8 mean; rough glass 0.15 / 0.5 / 1.0 brighter by +0.006% / +0.021% / +0.040% (z 6.1 / 4.7 / 14.5), the recovered loss; ior 1.05 roughness 1 darker by 0.20% (z -39). The last is the old eta axis *creating* energy near index match -- +0.6% to +1.65% per vertex, and 4.7% under at grazing exit at roughness 0.5 -- which the new table closes to +/-0.2%
- perf: `albedo_table.inc` 1.97 -> 9.08 MB; bake ~170 s, deterministic; `bsdf.cpp` compile 1.1 -> 2.1 s. Per call, rough glass: evaluate+pdf +1.4% to +1.7%, `sampleBsdf` +0.3% to +2.2%; opaque neutral. Render-level interleaved A/B (`bench_compare`, 10 rounds, cornell rough glass 0.5, 32 passes): pass_ms B/A 0.9956 [0.9815, 1.0092], not resolved
- docs: README §5 Wave 1 drops the item

## Clamp-free Smith terms at the silhouette

README §5 Wave 1: the specular value divided by `max(4*wo.z*wi.z, 1e-6)` while its VNDF pdf divided by `max(wo.z, 1e-6)` -- two thresholds on one geometry, darkening the lobe as `4c^2/1e-6` below cos 5e-4 (0.64x / 0.16x / 0.04x at cos 4e-4 / 2e-4 / 1e-4). The roadmap wanted one shared threshold; the exact rearrangement needs none.

- fix: `smithVisibility` = G2/(4 cosO cosI) = 0.5/(cosI*s(cosO) + cosO*s(cosI)) and `smithG1OverCos` = G1(c)/c = 2/(c + s(c)), with s(c) = sqrt(alpha^2 + (1-alpha^2)c^2) (Heitz 2014's height-correlated Smith term in Filament's `V_SmithGGXCorrelated` form). The cosines multiply rather than divide, so both are exact to the silhouette and vanish only where both cosines do. Used by the reflection value and pdf and by the Walter transmission value and pdf; `smithLambda`/`smithG1`/`smithG2` and their `1e-8`/`1e-6` floors deleted
- test: `dielectric_fresnel` sweeps on to cos 4e-4 / 2e-4 / 1e-4 / 1e-5 / 1e-7, through the old floor's band, against the unfloored double reference; its tolerance is now relative to F (float rounding is), 4.7e-7 over a measured worst 3.98e-7. Break: restoring the value floor fails 60 rows (0.637535 vs 0.996149 at ior 1.1 cos 4e-4 -- the 0.64x above). Cannot see the pdf's old `wo.z` floor, which engaged only below cos 1e-6 and is not isolable from the mixture pdf
- note: validators 96/96. Per-call A/B against `main` over 200k random direction pairs per material (clay, chrome, glass at roughness 0.15/0.5/1.0, half-transmissive): outside cos 5e-4 value and pdf agree to <= 5.0e-7 relative, 54-93% of pairs bit-identical; inside it the value moves by up to 79%, the removed darkening
- note: **image change, no systematic shift.** Every pixel of the six probe renders (cornell, opaque cornell, rough glass 0.15/0.5/1.0, ior 1.05) changes bit-wise, since reassociated arithmetic decorrelates the path's later decisions; paired mean luminance shift is within noise on all six (|z| <= 1.75, largest -0.075% at rough glass 1.0). The silhouette band itself is ~2e-4 px at 640x360
- perf: per call against `main`, min of 4 interleaved runs: evaluate+pdf -0.6% to +1.7%, `sampleBsdf` -0.1% to -2.7%
- docs: README §5 Wave 1 drops the item; §6's Heitz 2014 entry names the form

## Transmissive multiple scattering: one normaliser, no deficit gate

README §5 Wave 1: `msTransmit`'s selection was gated on `1 - escapeAvg > kMinDeficit` at the forward eta, the value it samples on the same test at the reciprocal eta. Escape is not symmetric in eta, so over roughness 0.128-0.168 on the entering side, at every ior from 1.01 to 2.5 (`glass.json`'s 1.5168 included), the transmitted lobe had value and no strategy drew it -- up to 9.9e-4 of the directional energy that neither estimator delivered: BSDF sampling never drew it, and NEE's `eval.pdf > 0` guard (`path_tracer.cpp`) discards it too. The exiting side mirrored it as selection mass on a zero lobe.

- fix: both transmissive multiple-scattering shares take value and normalisation from the row their density is read from: `f = energy * msTransmitPdf(mu_i) / mu_i`, over the reciprocal-eta row for the transmitted share and the forward-eta row for the reflected one. Each integrates to its energy exactly and the two sum to `1 - escapeWo` exactly; `f * cos / p_ms` is a constant, so the `msTransmit` strategy alone has zero variance (value, density and sampler read one interpolant, Dupuy & Jakob 2018)
- fix: `msTransmit` is selected exactly where its lobe has value (`transmitShape.scale > 0 && msEnergy > 0`), the MIS support condition (Veach 1997 §9.2). Weights stay bounded without a gate because the mass is energy-proportional: `K / p_ms = tint * transmitWeight * etaSq * (msE + ssE) / transmitProb`
- refactor: `kMinDeficit`, `multiScatterShape`, `LobeProbabilities::escapeAvg`/`escapeAvgRecip` and the split's `1e-6` guard deleted. `LobeProbabilities::transmitShape` holds the reciprocal row, built once per evaluation and shared by value, pdf and sampler; `msTransmitPdf`/`sampleMsTransmit` take it instead of rebuilding it
- test: `bsdf_validate`'s `strategy_coverage` (Fast, Exact): `evaluateBsdf > 0` implies `pdfBsdf > 0` over the whole sphere, swept at 8 steps per table node through the band. **Fails on `main`: 36624 of 127988 valued far-side directions with zero mixture density.** Passes after: 0 of 204288. Mutation: restoring the forward-eta selection gate alone fails it (76588). Cannot see density on a zero-valued lobe -- variance, not bias -- nor a dropped `scale > 0` term, since the row integral is never zero on the rough grid (measured)
- note: the gate existed because the value divided by a separately baked `1 - Eavg`, a trapezoid over unclamped grid values. Measured against the row's own exact integral it is 0.28-1.0x, and non-positive over roughness 0.032-0.068 (most negative -2.8e-5). The old lobe therefore over-delivered: at ior 1.5168 the reflected share by 4.9% at roughness 0.15 and 0.12% at 0.5, the transmitted share by 2.5% and 0.04%, falling to 1e-4 by roughness 1
- note: below the old gate both shares now return the energy they previously dropped. `transmission_reciprocity` isolated single scatter by relying on that switch-off; it now holds by magnitude -- worst mismatch unchanged with the term live (7.7e-5 at roughness 0.05, 3.0e-5 at 0.10, tolerance 1e-2). `rough_transmission_tint`'s roughness 0.15 row now carries 2.6e-5 `msTransmit` mass rather than exactly 0; its Exact linearity is unaffected since selection still never reads the tint
- note: validators: 96/96 (was 95/95). `transmissive_energy_balance` rows move by at most 2.55e-4, all in band; `transmissive_slab_energy` at roughness 1 1.02710 -> 1.02713; regenerated `albedo_table.inc` byte-identical (comment-only generator change)
- note: **image change.** Cornell (`glass.json` roughness 0.02, the delta branch for transmission): 111 of 230400 pixels, all brighter, on the glass sphere's continuous-reflection silhouette, +0.028% over those pixels -- the reflected share the forward gate used to drop. Bit-identical after the transmitted-share half alone (EXR `cmp`, against `main`'s render). Rough glass (probe scenes, sphere02 at roughness 0.15 / 0.5): +0.042% / -0.016% over changed pixels; at 0.15 a systematic bright ring at the silhouette, where the deficit is largest, is the unsampled energy arriving; at 0.5 the darkening is the removed over-delivery above
- perf: rough glass ior 1.5, per call against `main`: `sampleBsdf` 9-13% faster; evaluate+pdf 15% faster at roughness 0.5, 2.5% faster at 0.13 entering, 2.4% slower at 0.13 exiting, where `main`'s closed gates returned before any lookup. Render-level interleaved A/B (`bench_compare`, 10 rounds, rough-glass cornell, 32 passes): B/A 1.012 [0.995, 1.023], not resolved -- the BSDF is below the render's resolution
- docs: README §5 Wave 1 drops the item; the slab item records that its excess is not in the lobe's normalisation; §6 cites Veach's support condition and Dupuy & Jakob 2018

## Frame pacing: the swap tail attributed to WindowServer

README §5 Wave 0: `flushBuffer` blocked 14-26 ms at swap interval 0 on a few frames per convergence, cause unmeasured. System Trace needs Developer Mode, which is off on this host, so a scratch in-process probe (`results/wave0/swap_probe`, not committed) recorded every swap's wall and render-thread CPU time and sampled the render thread every 1 ms during swaps over a quarter period: Mach run state, CPU progress and a frame-pointer backtrace.

- note: **blocked on WindowServer, not waiting for a core and not working.** 808 of the 827 samples that found the render thread blocked were in a synchronous WindowServer query inside NSOpenGL's flush, `-[NSOpenGLContext flushBuffer]` -> `SLSFlushSurfaceWithOptionsAndIndex` -> `_CGSWindowIsOrderedIn` -> `mach_msg`; none found it runnable-waiting; swaps over half a period were 96-98% off-CPU. Not drawable back-pressure: no sample waits on a drawable
- note: the render thread already runs at user-interactive QoS (measured 0x21 on the main thread of a shell-launched process); the 8 trace workers and the driver thread are default (0x15). Dropping those nine threads to utility QoS, interleaved 3 against 3 convergences (~37.6k frames each), left the tail unchanged: 13 vs 9 swaps over half a period, exact conditional binomial P = 0.52; `swap_ms` p99.9 7.1-7.7 ms in both arms. The engine's own CPU load does not drive it
- note: long swaps are over-represented right after display-texture upload frames (7 of 41 against 1.3 expected, P = 3e-4), so the frame's GPU load raises the odds of a slow reply; 34 of 41 are not near an upload
- note: no in-process fix exists on the NSOpenGL present path; the query is inside Apple's flush. Removing it means presenting through `CAMetalLayer`
- docs: README §2 Frame pacing states the attributed tail and that a per-frame maximum of `swap_ms`/`frame_ms` measures WindowServer; `render_stats.h` `swapMs` and `waitForPreviousFrame`'s comment no longer claim the flush never blocks
- docs: README §5 Wave 0 closed: both items are done
- note: **no image change.** Comments and documentation only

## Driver: sample count published with its image

README §5 Wave 0: `driver.running_mean_matches_batch_mean` failed about one run in five. Measured cause: neither of the roadmap's two hypotheses. Accumulation order cannot vary with scheduling (`accumulateMean` is elementwise, each texel's recurrence runs serially on the driver thread, and the pass render is thread-count invariant), and the bound was not too tight. It was a **publish-order race in the driver**: `accumulatedSamples_` was bumped before the mean was formed and `result_` published, so a reader that saw `n` could fetch the `n - 1` image. Every failure had the identical error, 7.974e-02 against a 5.4e-07 bound. The HUD count had the same window, and ran ahead of the image again at every restart.

- fix: `PathTraceResult::samples`, stamped at publish beside `generation`, so image, count and generation are one immutable snapshot under the existing `shared_ptr` publish. `accumulatedSamples_` and `PathTraceDriver::accumulatedSamples()` are gone; the driver's count is a `driverLoop` local
- fix: the HUD and `-stats` dashboard read the count from the frame's displayed snapshot, so the readout always describes the image on screen
- test: `driver_validate` waits on the published snapshot's own `generation`/`samples` instead of a side counter; `new_request_restarts_accumulation` keys on the second request's generation, which the old counter-based wait passed vacuously
- test: the tolerance is now a derived bound. The old band was γ_n (recursive summation, Higham §4.2) times the image-wide peak of the *mean*, which is neither the running mean's error constant nor its magnitude. The new one is the running mean's exact error recurrence in Higham's θ/γ calculus, `E_k = (1 - 1/k)(1 + u)E_{k-1} + γ_4(|x_k - m_{k-1}| + E_{k-1})/k + u|m_k|`, evaluated per float from the exact data, plus `u|m_n|` for the oracle's own rounding (Welford 1962, West 1979, Chan, Golub & LeVeque 1983)
- test: all nine images are compared, not just beauty, and the check asserts every image varies across passes. The fixture gains a transmissive wall standing on the floor, so contact occlusion drives `ao`, floor-wall interreflection the indirect lanes, and its transmission lobe `refraction`; on the old single quad those four were constant and a missing accumulate entry for them was invisible
- test: `first_pass_is_bit_identical_to_oracle` compares all nine images
- note: **stress, `main` -> branch**, same seed, `--threads=2`: 1/50 -> **0/50** serial, 70/240 -> **0/240** at 12-way concurrency; the other eight driver checks 0/60 each at 12-way
- note: the bound on the honest driver: worst |running - batch| / bound = **0.363**, 0 of 2304 floats over. 13 mutations all fail it, and each dropped image is named: every one of the nine accumulate entries removed, `1/(n-1)`, `accumulateMean` skipped, the running sum published undivided, a stale 1/2 weight
- note: the fixture's first draft hard-coded tangent (1,0,0), parallel to the wall's normal, and rendered NaN in beauty and refraction; `pushQuad` now takes the tangent from the quad's first edge. A fixture defect, not a renderer one
- note: **no image change.** `-bench` CRC and ray counts identical to `main`; `render_beauty` for all nine path-traced AOVs, cornell 640x360/64 passes, PNG and EXR byte-identical
- docs: README §5 Wave 0 drops the driver item; README §6 cites the running mean and its error analysis

## Frame pacing: display-link vsync, per-event bench summaries

README §5 Wave 0: `-bench` per-frame columns depended on something outside the config (26653 vs 17383 frames for one config). Measured cause: two platform regimes, selected by window visibility. Visible, NSGL's swap interval let **two swaps through per refresh** (`frame_ms` alternating 3.6/13.0 ms, 2.00 frames per refresh in every 10 s bucket); occluded, GLFW's `swapBuffersNSGL` replaced it with a **fixed 60 Hz `usleep`** (1.00 per refresh). One run covered mid-way switched 2.00 -> 1.00 at the moment of occlusion, and `sample` put 1489 of the covered samples in `swapBuffersNSGL -> usleep`.

- fix: `DisplayLink` (`platform/display_link.mm`) paces the render loop to the vblank of the window's own display via `-[NSView displayLinkWithTarget:selector:]` on a user-interactive-QoS run-loop thread; `glfwSwapInterval(0)`, so neither platform regime runs. A minimised window, whose link was measured to stop ticking (0 ticks in 5.5 s; a covered one keeps ticking), free-runs on the display's period grid once a vblank goes a full period undelivered, so the loop never blocks on a paused link
- fix: one-frame GPU fence (`glFenceSync`/`glClientWaitSync`) after the vblank wait replaces the back-pressure the swap interval provided
- fix: refresh rate is the window's display's, `targetTimestamp - timestamp` of the latest vblank (seeded from `NSScreen.minimumRefreshInterval`), in the spec block, HUD, dashboard budget and bench config; `GpuInfo::refreshRateHz` read the primary monitor's integer mode
- fix: Homebrew's glew config links `OpenGL.framework` by absolute path into the SDK it was built with; `GLEW::GLEW` now links `OpenGL::GL`, so a foreign SDK no longer enters the `-F` path and shadows system framework headers as user headers
- feat: `FrameStageTimes` gains `paceMs` (vblank wait: slack) and `fenceMs` (GPU-bound time); `swapMs` is now the `flushBuffer` hand-off alone. The dashboard splits `swap (vsync)` into `gpu wait` / `vsync wait` / `swap`, and headroom excludes the vblank wait
- feat: `engine -bench` records `pace_ms` and `fence_ms`; `upload_ms` holds one entry per upload of the captured accumulation (exactly 128 at 128 passes), attributed by the generation `PathTraceDriver` now stamps on each published `PathTraceResult` -- the superseded request's in-flight pass had been landing a 129th upload. Schema 2
- feat: `bench_compare` summarises each samples column by its mean per event (Kalibera & Jones 2013's per-execution mean) instead of its total: every column is now indexed by its own event, and frame count scales with run duration, so a total of a per-frame column was never a fixed workload
- note: **pacing, before -> after** (cornell 1024x576, 128-pass `-bench`, Apple M1, 60.0024 Hz measured): 2.00 -> **1.000** frames per refresh visible, 1.00 -> **1.000** covered; `frame_ms` p10/p50/p90 **16.630/16.665/16.701 ms** visible, 16.631/16.665/16.701 covered; `sample` of the covered run shows the vblank wait, no `usleep`
- note: **the regime leaked into the per-pass columns** too, contrary to the roadmap item: halving render-thread frames returns CPU to the 8 trace threads. Median `pass_ms` 1728 ms before (visible) -> 1598 ms after, and `-stats` trace 1660 -> 1552 ms, 7.00 -> **7.49 Mray/s**. Unpaired single runs: the schema bump means the new `bench_compare` cannot pair against a schema-1 binary
- note: a default-QoS link thread measured `frame_ms` p10/p90 12.8/20.5 ms, the tick's delivery delayed by the trace threads; user-interactive QoS closed it to 16.63/16.70
- note: **no image change.** `-bench` CRC `3559296126` and ray counts identical in every run before and after; `render_beauty` Beauty and Refraction, cornell 640x360/64 passes, PNG and EXR byte-identical to `main`
- note: `driver.running_mean_matches_batch_mean` failed 6 of 30 on this branch and 7 of 30 on `main`: the pre-existing flake README §5 records, not a regression
- docs: README §5's Wave 0 pacing item is replaced by the residual it leaves: `flushBuffer` blocking 14-26 ms at swap interval 0 on 1-11 frames per convergence, cause not measured

## Roadmap: adaptive sampling parked, delighted view dropped

- docs: README §5 moves the render-mode selector from Wave 5 to Parked beside the adaptive per-pixel sample budget, so the two adaptive-sampling items sit together and the latter's "builds on tiling above" holds.
- docs: README §5 drops the delighted DirectDiffuse/DirectSpecular item and its references for now; Wave 4 is now two consumers, with the contact sheet noted as parked and its blocker stated directly.
- docs: Depth of field's adaptive-sampling pointer now says Parked rather than Wave 5.
- note: **no image change.** Documentation only.

## Rasterizer watertightness: the IOR-AOV diagonal crack

README §5 (Debug tooling #11): a dashed diagonal on the Cornell back wall in the IOR AOV, and in fact in every rasterizer AOV. It was measured before anything was changed. At 1024x576 there were **44** uncovered pixels whose Embree primary ray hits geometry, and they were all 44 of the frame's raster-vs-Embree coverage mismatches. At 640x360 there were **103**. Every one lay on a 45° line through pixel centres (x+y = 799 at 1024x576). At each one, the two triangles sharing the diagonal both rejected the pixel on that edge alone, with |w| below one float ulp of the edge function's product terms (e.g. -0x1.9db5d4p-8 against -0x1.9e7aep-10). The float edge function `(b-a) x (p-a)` rounds differently depending on which end of the edge it is referenced from, so a shared edge was not antisymmetric: a centre on it could be negative in both triangles. Forward and reversed evaluations of the *same* edge were both negative.

- fix: exact integer rasterization. Clipped vertices are snapped to a fixed-point grid, edge functions are int64, and coverage follows the top-left fill rule (D3D11.3 functional spec §3.4; Giesen 2013). A shared edge now evaluates to exactly opposite values in its two triangles, so every pixel centre on it is covered exactly once. `area == 0` on the snapped grid replaces the empirical `kMinTriangleArea = 1e-6`.
- fix: the sub-pixel precision is derived per frame rather than fixed: the most bits for which `max(W,H) << bits < 2^30`, so edge functions stay below 2^61 and exact. That is 19 bits at 1024x576. Hardware fixes 8 bits (D3D11 16.8, Vulkan `subPixelPrecisionBits`) for gate cost. Measured at 8 bits, the unclipped `rasterizer_validate` poses gained 18 and 9 silhouette mismatches against the Embree oracle over 20 seeds, from a ≤0.0028 px snap across centres 0.0008–0.0014 px inside a true edge. At the derived precision, all three poses reproduce the pre-fix counts exactly.
- fix: the Sutherland-Hodgman clip is now against the full view frustum (Blinn & Newell 1978 outcodes), not the near plane alone, which is what bounds snapped coordinates to the viewport. Each edge-plane intersection is computed from the lexicographically smaller endpoint, and a triangle needing any clip is clipped against all five planes in fixed order, so neighbours produce bitwise-identical clip vertices on a shared edge. The near-only clip computed them in each triangle's own winding.
- fix: the Wireframe AOV draws only original mesh edges. Clipping tracks which polygon edges lie on the source triangle, so clip-plane and fan-diagonal edges are no longer drawn, including the near-plane ones the old clip produced.
- perf: the depth pass steps edge functions incrementally along the row (Pineda 1988). This is exact in integers, so it is bit-identical to per-pixel evaluation (same `raster_bench` output CRC in every run) and 17.8% faster than it. Against `main`, `bench_compare run`, 12 rounds, 2048x1152:

  | workload | B/A (95% CI) |
  |---|---|
  | 20561 triangles | 0.977 [0.939, 1.011], not resolved |
  | 5M triangles | **0.780** [0.757, 0.802] |
  | 8 depth layers | **0.904** [0.859, 0.940] |

  The 5M gain comes from dropping sub-triangles whose snapped extent contains no pixel centre. Per-pixel from-scratch int64 evaluation had measured 9.2% *slower* at 8 layers, which is why the incremental form is part of this change.
- test: `rasterizer.watertight_closed_mesh`, `Fast`/**`Exact`**. From inside a closed, index-shared, 16x16-per-face cube, every pixel must be covered, with no oracle and no tolerance. The straight-on pose runs the shared diagonals exactly through pixel centres, as on Cornell. Eight seeded poses view a jittered cube, so shared edges are clipped against every plane. **Verified by breaking:** pre-fix `rasterizer.cpp` goes **red** (2 uncovered pixels, straight on), and so does making both sides of the fill rule exclusive (277 uncovered).
- test: `rasterizer.wireframe_ignores_clip_edges`, `Fast`/**`Exact`**. One screen-covering triangle, tilted to reach behind the camera and clipped against all five planes, must leave every pixel covered and none wireframe. **Verified by breaking:** flagging clip edges as mesh edges gives 380 wireframe pixels, **red**.
- note: **named gaps.** Two breaks leave the suite green. Making both sides of the fill rule *inclusive* double-covers shared edges, which a z-buffered G-buffer cannot show. Computing clip intersections in winding order rather than canonical order leaves a sub-ulp sliver no pixel centre lands in at these sizes.
- note: **image change, rasterizer AOVs only** (Cornell 1024x576, all 14 AOVs diffed):
  - IOR, Alpha, ObjectID and Roughness change at exactly the 44 filled gaps and nowhere else.
  - Elsewhere, interpolated AOVs move at float rounding: p99 ≈ 1e-7, Depth at most 1.2e-5, WorldPos at most 2.4e-7, from barycentrics now formed from exact integer weights.
  - 7 pixels on the 45° room-corner crease (x−y = 224) change surface. Their centres lie exactly on a boundary two faces share at equal depth: the old code gave them to whichever triangle came first in the list, and the fill rule now decides.
  - 66 Wireframe pixels flip, all on a line boundary.
  - Beauty is untouched: Cornell 640x360/64 passes, PNG and EXR byte-identical.

## Rough-transmission tint coverage at transmissionDepth 0

README §5.1: `on_surface_transmission_tint` authors roughness 0.02 on both its scenes (`alpha` 4e-4, below `kSmoothAlpha`), so every reading came from `sampleBsdf`'s delta branch. The rough transmission path's tint had **no** integrator-level coverage at all -- tinting only the delta branch and reverting both continuous sites failed `bsdf_validate` 54 times and left this binary green.

The roadmap proposed a rough row at `ior` 1.0, on the argument that Walter refraction is straight through there so the reading stays `transmissionColor^2`. Measured first, as it asked: the transmissive multiple-scattering lobe **does** activate at `eta = 1` (the deficit crosses `kMinDeficit` between roughness 0.10 and 0.15), and the single-scatter lobe is degenerate there besides -- so `ior` 1 now routes to the delta branch and is no longer a *rough* test point at all.

- test: `integrator.rough_transmission_tint`, `Slow`/**`Exact`**. One rough interface at `ior` 1.5, a black environment and a single one-sided light *behind* it, so a reflected path sees nothing and every photon in the reading crossed exactly once. `Lo` is then exactly linear in `transmissionColor` through the origin: `Lo(tint) == tint * Lo(1)` per channel, and `Lo(0) == 0` with no tolerance at all
- feat: `Exact`, not `Statistical`, and the scene is what earns it: `computeLobeProbabilities` never reads `transmissionTint` and Russian roulette is off, so the three renders of a row draw the identical sampler sequence and select the identical lobe at every vertex. The relation holds path by path, noise and all. Measured worst departure **1.37e-06** relative, identical at 1, 2, 4 and 8 threads -- float non-distributivity, not a reduction-order race -- against a 1e-5 band
- feat: two rows straddling the transmit-side multiple-scattering lobe's own switch-on (roughness 0.15, `msTransmit` exactly 0; roughness 0.6, `msTransmit` 0.0263 of the selection mass), so both far-hemisphere strategies are covered. Being MIS-eligible, the rough lobe also puts the far-side NEE shadow ray under test, which the delta rows cannot reach
- note: **verified by breaking**, against the exact faults it exists for. Tint dropped from `transmitMultiScatter`: **red**, `Lo(0)` 1.5e-3 at roughness 0.6 and linearity off 0.37% -- and **2.4e-10** at roughness 0.15, which is why the zero row carries no tolerance; any band would have hidden it. Tint dropped from `evaluateTransmissionLobe`: **red**, `Lo(0)` reads the full 1.279. Tint applied twice: **red**. `on_surface_transmission_tint` stays **green** through all three, which is the hole reproduced. Conversely, dropping the tint from the delta branch alone leaves the new check green and turns the old one red -- the two are complementary, not overlapping
- note: **no image change.** This commit adds a check and removes a roadmap item; no shipped code is touched

## Index-matched rough transmission: NaN at ior 1

`index_matched_transmission` drew 4096 rough samples per angle at `ior` 1 and asserted only their *count*, which is why this was invisible: at `ior` 1, `refractAbout` returns `-wo` about **every** microfacet normal, so `evaluateTransmissionLobe`'s half-vector `normalize(wo + etaR*wi)` normalised the zero vector and every guard below it became a NaN comparison. Measured **7783 of 7783** transmission draws non-finite at roughness 0.1 and **6666 of 7783** at roughness 1.0, the remainder being `msTransmit` draws, which do not form that half-vector.

- fix: `transmissionIsRough` (`bsdf.cpp`) gains `params.ior != 1.0F`, so an index-matched interface takes the Snell delta branch at **every** roughness. PBRT-v4's `DielectricBxDF` branches its value, pdf and sampler on the same `eta == 1 || EffectivelySmooth()`. Exact equality, no new constant: the predicate is already the single gate at all three sites (selection, evaluation, sampling), so the three cannot disagree
- fix: `computeLobeProbabilities` takes the exact index-matched boundary (`escape = escapeAvg = escapeAvgRecip = transmitShare = 1`) instead of the table. The escape table's eta axis is log-spaced and puts `eta` 1 exactly halfway between its 0.941 and 1.063 nodes, never on one, reading a deficit of **0.209 at roughness 1** where the truth is 0; left to interpolate, that deficit reached `evaluateSpecularLobe`'s `(1 - transmitShare)` term and deposited untinted reflected energy on an interface whose exact Fresnel is identically zero
- test: `index_matched_transmission` now asserts every rough draw rather than counting it -- `pdf == 0` (the delta branch), a finite throughput, `wi == -wo` and a scalar throughput/tint ratio -- plus three near-hemisphere rows per angle requiring the reflected value to be **exactly** zero, and an anti-vacuity guard, since every one of those assertions is per-draw and a collapse to no transmission at all would satisfy them by vacuum. **Verified by breaking:** reverting the `ior != 1` term fails 3891 of 3891 draws at every one of the 8 angles, and reverting the escape boundary alone fails the reflection rows at up to 0.0191; the pre-change check passed both
- note: this also answers README §5.1's open question, measured rather than assumed. The transmissive multiple-scattering lobe **does** activate at `eta = 1`: the deficit crosses `kMinDeficit` between roughness 0.10 (0.00023) and 0.15 (0.00140), reaching 0.209 at roughness 1.0, because it is dominated by Smith G2 shadowing and not by Fresnel. So `ior` 1 is no longer available as a *rough* transmission test point -- it now routes to the delta branch by design
- note: **no image change.** Every printed row of `bsdf_validate`, `integrator_validate` and `nee_validate` is identical before and after; Cornell 640x360/64 passes is byte-identical. No shipped material is index-matched (`glass.json` is 1.5168), and the change is inert for every `ior != 1`

## Benchmark log

README §5.5: timing lived on stdout only, so a single-change claim meant interleaving two builds by hand against drift larger than most effects. Blocked §5.3's display-texture upload and AOV-switch restart items.

- feat: `debug/bench_log.h` -- append-only JSON Lines, one record per timing run, written with a single `O_APPEND` `write(2)`: build (git SHA, Mach-O `LC_UUID`, type, compiler, march, IPO, Embree), host topology, argv, resolved config, raw per-iteration samples, rusage (user/sys CPU, context switches, max RSS), and the work done (ray counts, CRC-32 of the output). Arnold's `stats_file` append mode is the precedent
- feat: `render_beauty --bench-log`, `raster_bench --bench-log`, and `engine -bench PATH`. The app mode runs one accumulation to `maxSamples` and exits; the capture restarts on every dispatched request, so it holds the settled full-resolution convergence, not the interactive-scale warm-up whose length depends on build speed. It refuses to log a capture that closed early or missed a pass record
- feat: `bench_compare` -- `run` does randomized multiple interleaved trials of two binaries (Abedi & Brecht 2017) and reports the paired Hodges-Lehmann ratio with its exact Wilcoxon signed-rank interval; `compare`/`history` give the unpaired Mann-Whitney form over existing records. It says "not resolved" when the interval contains 1
- feat: `tools/stats.h` gains exact signed-rank and rank-sum nulls (in probability space, no normal approximation at any n) and both Hodges-Lehmann estimators
- feat: git SHA stamped at build time (`cmake/GitSha.cmake`, LLVM's `GenerateVersionFromVCS` pattern), so a commit made without re-running cmake is no longer attributed to its parent. `queryHostInfo`/`buildInfo` move to GL-free `host_info.cpp` so headless tools can report them
- feat: `PathTraceDriver::requestTrace` returns the generation it created, which the app needs to tell its accumulation's pass records from a superseded request's
- test: `io_validate` (log round trip, unwritable path, CRC known answers), `check_selftest` (nulls against brute-force enumeration and Hollander et al.'s tabulated values; interval coverage exact within a Wilson band), `driver_validate` (the returned generation labels the passes). **Verified by breaking:** no newline, open failure reported as success, CRC short by one byte, order-statistic index off by one, swapped rank-sum weights, and the pre-increment generation each turn their check red
- note: **verified end to end.** A vs A: B/A 0.993 [0.981, 1.003], not resolved. A vs a probe sleeping 3 ms per pass: 1.128 [1.116, 1.135], resolved at the right sign; the predicted 8 x 3.77 ms (sleep measured separately) = 30.2 ms against 31.5 measured. `user_s` sees no difference, as a sleep uses no CPU. Two builds of the same source give the same `LC_UUID`; a one-line edit changes it
- note: **no image change.** Cornell 640x360/96 passes: PNG and EXR byte-identical to `main`, ray counts identical
- note: `driver.running_mean_matches_batch_mean` fails about one run in five at `--threads=2`, on `main` (10/50) as well as here (11/50) at one fixed seed -- a pre-existing flake this branch's full-suite run surfaced, not a regression. Filed on the roadmap; the check is labelled `exact`
- note: first `engine -bench` at the default profile: 128 passes, 128 uploads at **3.6-7.4 ms, median 4.6 ms** (README previously said 4.0-7.0 from `-stats`). Frame pacing measured at median 10.9 ms against a 60 Hz display, cause not established; filed on the roadmap, since it makes per-frame totals depend on something outside the config. Two runs of the same config gave identical CRC and ray counts but 26653 vs 17383 frames, which is that dependence

## Validation harness: named checks, derived bands, full CPU coverage

README §5.6 recorded six validators wired into `ctest` as six opaque binaries, each rolling its own assertions, with hand-picked Monte Carlo bands and no image gate. This replaces the engineering around them; the physics each check asserts is unchanged except where a band was re-derived.

- feat: `tools/check.h` -- named, self-registering checks discovered into `ctest` from the binary itself (`--list-ctest`, POST_BUILD, same mechanism as `gtest_discover_tests`). 6 entries -> **85**, each `<suite>.<check>`, labelled `fast`/`slow` and `exact`/`statistical`. `ctest -L fast -L exact` is a **1.2 s** pre-commit gate (44 checks); the full suite runs **90 s -> 48 s** under `-j4`, with `PROCESSORS` set so concurrent tests do not oversubscribe
- feat: seeds are derived from each check's **name**, not its position, so adding or reordering a check leaves every other check's realization bit-identical. Negated-pass assertions keep NaN failing, guarded by `check_selftest`
- feat: `tools/stats.h` -- one family-wise error rate (`kFamilyAlpha`, Sidak across checks and assertions) replaces hardcoded levels. Student-t quantiles replace "6 sigma", Wilson replaces Wald (Brown, Cai & DasGupta 2001), Welch difference bands replace flat relative tolerances where both sides are noisy. Estimators validated against published values (t(0.05,15) = 2.1314, z(0.975) = 1.95996)
- feat: randomized-QMC replication (`kReplicates` = 16 independent scrambles) for Sampler-driven estimators, whose samples are one Owen-scrambled Sobol set and have no `sqrt(N)` error
- fix: `nee_validate`'s MIS bands (`5% of max(reference, 0.1)`) modelled the reference as exact and floored the denominator. Replaced by a difference band over replicates. **Verified by breaking:** a 1% bias now fails 3/18 rows and 2% fails 14/18; the old band admitted 5%
- fix: `sampler_validate`'s hardcoded chi-square critical value `54.0` becomes the distribution's own at the suite rate (50.49, tighter). `bsdf_validate`'s sampling chi-square moves from its own 1% rate to the suite rate, with draws raised 200000 -> 280000 by the noncentrality ratio (119.2 -> 165.6 at 128 dof) so detectable effect is unchanged
- feat: `driver_validate` -- `PathTraceDriver` had no coverage. Running mean vs batch mean under a Higham forward-error bound, first pass bit-identical to a direct render, `maxSamples` cap, restart on a new request, suspension, pinned results never overwritten, over-range stats exact against a serial scan, bit-identical output across thread counts. **Verified by breaking:** `sampleBase` stuck at 0 and an off-by-one running-mean divisor both fail
- feat: `io_validate` -- EXR round trip bit-exact (the path was linked into three validators and invoked by none), scene/profile/film-back config accept **and** reject. Rejection rows are built from a base that must itself load, after a first draft passed two rows for the wrong reason
- feat: `display_validate` -- the pinned OCIO view maps 0 -> 0 and 1 -> 1, is monotone and channel-independent, previously asserted only by a comment. `FrameStats::tick(time_point)` makes the ring buffer and percentiles testable without sleeping. `histogram.cpp` is FBO-bound and left uncovered, stated rather than faked
- feat: `render_beauty --assert-deterministic` (same seed, bit-identical floats -- the claim in the file's own header) and `--assert-converged` (two independent families of 8 sub-renders agree within a per-pixel Student-t band, Sidak over pixels; no golden image, no tuned threshold). Registered as `image.determinism` / `image.convergence`. **Verified by breaking:** a 0.5% systematic bias fails
- feat: `critical_angle_onset` and `tir_predicate_agreement` -- TIR at a real critical angle (ior > 1), which the index-matched checks cannot reach by construction. Both are `exact` and outside the band conversion
- refactor: `fresnelDielectric`/`cos2Transmitted` move to `fresnel_dielectric.h`, shared by `bsdf.cpp` and `albedo_table.cpp` in place of two verbatim copies a one-sided fix would silently desynchronise. `applyOcioDisplayTransform` moves from `render_beauty` to `src/gfx`
- chore: cppcheck **210 -> 0** (fixes plus a documented suppression list), clang-tidy on `engine` **30 -> 11**, all remaining `readability-function-size` on hot-path functions left for their own change
- note: **no image change.** Cornell 320x240/64 passes: linear RMSE 0, relMSE 0, 0/230400 channels differ, EXR and PNG byte-identical to `main`. Render time interleaved against a `main` build: 71.0 vs 72.5 ms/pass, within run-to-run spread

## Snell's law in cos^2 form: index-matched interfaces stop reflecting

README §5.1 recorded `fresnelDielectric(mu, 1, 1)` returning `1.0F` instead of 0 below `mu = 1.7263349e-4`, and proposed gating the total-internal-reflection early-out on `etaI != etaT`. That treats one symptom at one site. The cause is the transcription: `sin2ThetaT = r^2*(1 - cos2ThetaI)` forms `1.0F - mu*mu`, which rounds to exactly `1.0F` below `mu = 2^-12`, and **three** sites shared it -- `fresnelDielectric`, `refractAbout`, and the smooth transmission branch. The latter two returned no sample at all, which is energy deleted from the estimator rather than a wrong value, and no furnace, reciprocity or chi-square row can see an absence.

- refactor: `cos2Transmitted(cosThetaI, etaRatio)` returns `(1 - r^2) + r^2*cos2ThetaI`, algebraically `1 - r^2 sin2ThetaI` (PBRT-v4 `FrDielectric`/`Refract`, Walter 2007 eq. 40) but never forming `1 - cos2ThetaI`. One helper replaces all three transcriptions. TIR is `cos2ThetaT < 0`, strict: at the exact critical angle `cosThetaT` is 0, both polarisation terms are already exactly 1, and the general formula returns 1 without an early-out
- note: **the index-match case needs no branch.** At `r == 1` the constant term is exactly zero and the expression *is* `cos2ThetaI`, which cannot be negative -- so an interface with no critical angle cannot report one, by construction. At `r < 1` the constant term is positive, so entering a denser medium cannot either. Mitsuba 3's `fresnel()` covers the same degeneracy with an explicit index-match branch; this form makes that branch unnecessary
- fix: a **second** cancellation, not in the README item and found by the revert test. `cosThetaT` came from `1 - sinThetaT*sinThetaT`, which at `mu = 0.02` subtracts `0.99960004` from 1 in float32, so `cosThetaT` missed `mu` by ~1e-6 and the polarisation terms -- which must cancel to exactly zero at `r == 1` -- left `F ~ 6e-10`. `D/(4*mu*mu)` multiplies that by ~5e8, so the lobe read **0.337797** at roughness 0.02 where it must be 0. The cos^2 form returns `mu*mu` exactly, so `cosThetaT == mu` bit-for-bit and both terms are exactly zero. The defect was three orders of magnitude wider in angle than the item recorded
- perf: `fresnelDielectric` loses one of its two `std::sqrt` calls and both `std::max` guards, which the new form cannot need. It runs per specular evaluation, per transmission evaluation and once per sample in `computeLobeProbabilities`
- feat: `checkIndexMatchedTransmission` (`bsdf_validate`), the only check in the suite reaching `sampleBsdf`'s two refraction sites. Asserts the transmission **sample count** is constant across the angle sweep rather than bounding rejections: lobe selection is angle-independent at ior 1 and every row draws the identical sampler sequence, so the count is a constant of the sweep, and a false TIR is a deficit against the normal-incidence row with no tolerance and no noise
- note: the rejection count is the wrong quantity and was measured to be so. 6 of 4096 draws at `cos 0.1` return nothing, all of them **reflection** draws discarded on guards unrelated to refraction, carrying exactly zero energy at ior 1. Bounding rejections would have asserted against those instead. Chromaticity, not throughput, for the same reason: `sampleBsdf` returns `f/pdf`, so one draw carries `tint/P` and reads 0.842105 against a 0.8 tint at `P = 0.95` -- that factor is what makes the estimator unbiased, and asserting it away would assert a bias in
- feat: `checkDielectricFresnel` gains 24 index-matched rows asserting the specular lobe is **exactly** zero, no tolerance and no division by the geometry term: `F` is the only ior-dependent factor, so `F == +0` zeroes the product whatever `D` and `G2` are. Kept out of the main sweep deliberately -- its strict monotonicity assertion is stated for `n > 1`, and at ior 1 the true curve is identically zero
- feat: `checkIndexMatchedCoat`'s sweep extends from `mu >= 0.05` down to `1e-5`, 1512 -> 5082 comparisons. Its comment previously said "Do NOT widen mu below 1e-3" and recorded the artefact as a known limitation; the collapse is now exact there and the comment says why instead
- note: **verified by breaking.** Reverting `bsdf.cpp` alone turns all three checks red -- 18 index-matched Fresnel rows (6 of 8 cosines at every roughness, `cos 1.0` and `0.5` the only harmless ones), the coat sweep's value assertion at 0.99 relative, and the transmission count. The lobe reads **4.8e+11** at roughness 0.02 / `cos 1e-5` against a required exact zero
- note: `tools/albedo_table.cpp` carries verbatim copies of both functions and got the identical fix, so the table is baked against the same interface it is shaded against. Re-baked: `kAlbedoA`/`kAlbedoB`/`kAlbedoAvg*`/`kMsReflect*` are **bit-identical** -- they force Fresnel to 1 and never call it -- and only the escape/transmit tables move, worst **1.8e-6** absolute against the generator's own 2.97e-5 residual. The bake reproduces bit-for-bit on a second run

## Reflected multiple-scattering lobe gets its own sampling strategy

README §5.1's top item asked to gate the EON sampling *shape* on `diffuseKd`. That treats a symptom: `bsdf.cpp` said outright that the reflected Kulla-Conty share "has no sampling strategy of its own and is picked up by whichever of the two existing strategies draws that wi". The reflection hemisphere had two strategies for three lobes, so `lobes.diffuse` served both the EON lobe and the multiple-scattering lobe under one shape parameter that can be right for at most one of them. `metallic=1` is where they fully separate: the slot carries all of the multiple-scattering energy and none of the diffuse energy, and still drew CLTC.

- feat: `lobes.msReflect`, a cosine strategy over the near hemisphere, the exact mirror of the `msTransmit` the far hemisphere has had all along -- own slot in `LobeProbabilities`, own branch in the sampling ladder, own term in the mixture density. `sampleEon`/`pdfEon` are untouched and keep the material's true `diffuseRoughness`, which is correct for the lobe EON actually describes
- feat: the split is **purely energy-proportional, with no threshold anywhere**: `msEnergy / (msEnergy + diffuseEnergy)` of the existing `diffuseProb`. `msEnergy` is exact rather than a proxy -- `opaqueMs` integrates over the hemisphere to `fms*(1-E(mu_o))`, since `int (1-E(mu_i)) cos = pi*(1-Eavg)`. A conductor's `diffuseKd` is exactly 0, so its diffuse selection mass falls to exactly 0 on its own and **no gate on `diffuseKd` is needed at any site** -- which is why the roadmap's stated fix is not the one taken
- note: the `kMinDeficit` gate the transmissive split uses was **considered and rejected on measurement**. `chrome.json` is authored at roughness 0.045-0.05, where `1-Eavg` is order `1e-3` -- right on `kMinDeficit` -- so a deficit gate would switch `msReflect` off on precisely the asset this item is about. The energy ratio sends the share to zero continuously without one
- fix: a second, latent bug closed by the same change. Rough-metal multiple-scattering energy was bucketed `DirectDiffuse`/`IndirectDiffuse` because the diffuse *strategy* drew it, and a metal has no diffuse lobe. It is Specular now, in both senses: repeated scattering on the GGX microsurface, drawn by a strategy of its own. Measured on cornell at 256 passes -- `DirectDiffuse` mean signed **-0.0039**, `IndirectDiffuse` **-0.0201** against `DirectSpecular` **+0.0037**, `IndirectSpecular` **+0.0216**; the four move as one transfer, and `checkTransportPartition` still holds because `PathBucket` is exhaustive
- feat: `checkEonDiffuseFurnace` (`bsdf_validate`) gains conductor rows, and on them an assertion stronger than the energy band: a conductor's furnace reading must be **bit-identical across `diffuseRoughness`**, because `diffuseKd` is exactly 0 there so the parameter must reach nothing. Seeded by `(metallic, ndotV)` only, deliberately not by `diffuseRoughness`, or the comparison would measure the realization instead of the shape. The check was dielectric-only before, and said so -- which is exactly why the conductor case went unguarded
- feat: `checkSampleDensityConsistency` (`bsdf_validate`), 3.27M sampled directions re-evaluated at exact equality. `BsdfSample::pdf` is documented in `bsdf.h` as "exactly what `pdfBsdf` would return for it" and nothing asserted it; `checkPdfNormalization` reads `sample->pdf`, the density the sampler reports about itself, and never re-evaluates it
- note: **that check's coverage was measured, not assumed, and it is narrower than it looks.** It does NOT catch a mixture term missing from the density -- `sampleBsdf` reports `evaluateContinuousLobes`' own pdf, so both sides are wrong together: deleting the `msReflect` density term leaves **0 failures** here and fails `checkFurnace` at **Lo=1.28** instead. What it does own is the sign-mirroring round trip through the exiting-side rows, which nothing else closes, and any future strategy reporting a hand-computed density beside the mixture. Recorded rather than overclaimed
- note: **verified by breaking.** Against the pre-change BSDF the conductor invariance fails **12 of 12** rows while sample/pdf consistency passes cleanly -- the instrument is specific to this defect, not a tripwire. Suite cost 55.3 -> 59.8 s
- note: **the mechanism is measurably gone.** Re-authoring `chrome.json` to the `diffuseRoughness: 0.3` that `d3c9eac` removed as a workaround: before, 96 passes at 640x360 moved **8422/691200 channels, max 55/255** against the same render at 0 -- a pure sampling-shape change on a lobe of identically zero value. After, **0/691200, max 0/255, RMS 0**. Bit-identical, which is the whole claim: `diffuseRoughness` no longer reaches a conductor at all, so the asset-level workaround is no longer load-bearing
- note: the plan's "bit-identical Beauty on today's assets" prediction was **wrong and is recorded as such**. At `diffuseRoughness 0` CLTC does collapse to a cosine *distribution*, but `cltcSample` and `sampleCosineHemisphere` are different maps from the same `u` (CLTC negates x and rotates into `wo`'s azimuth), so the realization differs even where the density does not -- and `clay.json`, cornell's default material, is authored at `diffuseRoughness 0.3` with real diffuse energy, so its mass genuinely splits across two strategies now
- note: no bias, and no variance change on this scene. Converged 1024-pass before/after at a seed independent of the renders: linear **RMSE 9.77e-4, relMSE 4.20e-6, mean signed +2.3e-5** -- Monte Carlo residual between two independent realizations, not a shift. 96-pass relMSE against that reference over seeds 1/2/3: **0.009234/0.009338/0.009178 before, 0.009239/0.009345/0.009180 after**, a +0.03-0.08% move inside the 0.9% spread between seeds. Cornell cannot show a variance win here and was not expected to -- its only conductor is authored at `diffuseRoughness 0`, where both strategies were already cosine
- docs: README §5.1's item is replaced by the two residuals that outlive it -- the exact `(1-E)cos` sampler (bounded at `(1-E(0))/(1-Eavg)`, blocked on a per-roughness inverse CDF the 32x32 directional table cannot provide) and the exiting side's reflected share, still VNDF-only. §4 now states that the four Direct/Indirect buckets key on the sampling strategy rather than the material class

## Index-matched coat invariant

`coatAlbedo`'s `fresnelAvg` argument was the last unpinned half of `ae418ee`. `checkAverageFresnel` asserts `dielectricFresnelAvg` the function; nothing asserted the three call sites, so reverting them to Karis' mean of Schlick left all six validators green while moving the picture. This is the instrument that closes it, and it is test-only.

- feat: `checkIndexMatchedCoat` (`bsdf_validate`) asserts that at `ior = 1` the diffuse channel is **bit-identical across specular roughness**. An index-matched interface reflects nothing, so the coat is optically absent and its roughness cannot be observable: `dielectricFresnelAvg(1) = (1-1)/5.08638` and `fresnelDielectric(mu,1,1)` are both exactly `+0.0F`, so `coatAlbedo` is `+0.0F`, `diffuseCoupling` is `1.0F`, and `.diffuse` is `evaluateEon` multiplied by exactly one. Tolerance is therefore **exactly zero**, from `x * 1.0F == x` -- an algebraic guarantee, not a measured run
- feat: no new export. `coatAlbedo` stays in its anonymous namespace and the 32x32 albedo table keeps having no accessor, the convention `checkWhiteFurnaceTwoSided` states outright. `makeDiffuseParams` gains a trailing defaulted `roughness`, so `checkEonAlbedoInversion` is byte-identical and every pre-existing row is a strict subset of the swept coverage
- feat: a second assertion per row, complementary rather than redundant -- `.diffuse` against `referenceEon`, an independent double-precision transcription of the paper's eq. 16-19 with c1/c2 re-derived from literals. The invariance half cannot see a roughness-*independent* corruption (a pinned `diffuseKd`, a lost `1/(1-coatAvg)` normalisation, a channel swap); this half can. Worst residual **1.74e-7** against a 1e-6 bound, 5.7x under and thin on purpose
- note: **verified by breaking, which is the whole point.** Reverting all three call sites to `schlickFresnelAvg(coatF0)` fails **1454 of 1512** comparisons, worst **7.76e-4** relative at roughness 0.92 / `mu_o = mu_i = 1` -- about 8000 ULP at the diffuse value's magnitude. No other check in `bsdf_validate` fails, and `embree`/`nee`/`rasterizer`/`integrator`/`sampler_validate` all still exit 0. The instrument is specific, not a tripwire
- note: **the sweep is shaped by two facts that run against instinct.** The deviation peaks at *normal* incidence, not grazing, because `coat = msTint*(1-E(mu))` and `E` rises toward grazing at high roughness (0.31 at `mu=1`, 0.89 at `mu=1/31`) -- a grazing-first sweep is ~6x weaker. And it changes sign below `mu ~ 0.3` at roughness 1, so only `|delta| == 0` is a safe predicate; any signed bound breaks. Roughness 0.37 and 0.92 sit deliberately off the table's `k/31` grid, and 0.92 is where the revert is worst
- note: **both routes the README proposed were measured and rejected.** A direct assertion on `coatAlbedo` against an independently integrated reference cannot work at any `ior > 1`: every term carrying `F_avg` is `multiScatterTint(F_avg, Eavg) * (1 - E(mu, alpha))`, `E`/`Eavg` come only from the 32x32 table whose own quadrature error is ~1.5e-3, and the two candidates are only **2.0e-4** apart there -- the reference is already 7.5x coarser than the signal, and resolving it at 10:1 needs an `E` **75x** more accurate than the table being checked. A coat-specific energy identity fails the same way against 2e-2 furnace tolerances. `ior = 1` is the unique point where the table's coefficients are multiplied by exact zeros and drop out of the expression entirely
- note: a monotonic `ior -> 1` continuity column was designed, computed, and **dropped for having no discriminating power**: over `{1.5, 1.3, 1.2, 1.1, 1.05, 1.01, 1.0}` both candidates' roughness-sensitivity is monotone decreasing (shipped 3.2550e-2 -> 0, Karis 3.2466e-2 -> 7.74e-4), so Karis passes the predicate at every ior down to 1.001. Only the terminal exact-zero row discriminates, and the invariance assertion already owns it
- note: verified image-identical and instrument-only. Linear **RMSE 0, relMSE 0** against a pre-change 64-pass EXR at 640x360, `0/691200` channels differ, ray counts identical to the digit (primary 15272448, bounce 27663511, ao 7778812, shadow 21884435), 6/6 validators pass. `engine` does not link `bsdf_validate.cpp`, so there is no render-path surface to regress
- note: for the record, the Karis revert moves today's cornell **5612/691200 channels, max 49/255, mean signed +0.0086** (Karis brighter), linear RMSE 6.84e-4, relMSE 1.43e-6. Not comparable to `ae418ee`'s 1886/13-per-255: that scene had no emissive quad light and a different sampler, and at 64 passes part of this tail is noise reallocation rather than bias. The direction agrees, which is the claim being checked
- fix: found while building the check -- `fresnelDielectric(mu, 1, 1)` returns exactly `1.0F`, not 0, for `mu <= 1.7263349e-4`. At `etaI == etaT` the ratio is exactly 1 and `1.0F - mu*mu` rounds to `1.0F`, tripping the total-internal-reflection early-out at an interface that cannot have one. The sweep holds `mu >= 1e-3` and says why; the artefact itself is recorded rather than silently retuned
- docs: README §5.1's `F_avg` item is replaced by the two things that outlive it -- the index-matched grazing artefact, and the residual this check cannot close (the coat's `F_avg` *value* at working ior, blocked on a more accurate `E`). The Build section said "nine targets" and listed nine; `CMakeLists.txt` has **thirteen** `add_executable` calls, so `bluenoise_mask`, `gltf_tangent`, `sampler_validate` and `render_beauty` are added

## Stale validator count in README §7

- docs: §7's test-suite item said `ctest` wires 5 validators and named 5; it has wired 6 since `sampler_validate` landed. `CMakeLists.txt:274`'s own comment carried the same stale "five". Both line references in the entry had drifted too: the `add_test` loop is at 275-277, not 229-232, and `render_beauty`'s deliberate exclusion is at 245, not 199
- docs: the same entry offered "sampler, BSDF math in isolation" as logic with no isolated coverage, and **both halves were false** -- `sampler_validate` links only `sampler.cpp`, `bsdf_validate` only `bsdf.cpp` plus `sampler.cpp`, so each already tests its subject with no scene. The examples are dropped rather than replaced with an unverified substitute; the open work is what it always was, restated precisely: the six validators are standalone binaries rolling their own assertions, with no shared unit-test framework behind them and no automated regression-image gate

## Over-range statistics on the driver thread

`updateOverRangeStats` walked all 2.36 M texels of the beauty buffer single-threaded on the render thread, one frame in four, to produce the HUD's two over-range scalars: **8.950 ms** per firing at 2048x1152, ~2 ms amortised into every frame, the largest render-thread cost after the vsync wait. The work now happens once per pass on the driver's own pool, and the render thread reads two published numbers.

- feat: `OverRangeStats` on `PathTraceResult` -- an exposure-free `rawPeak` plus `aboveBin`, the complementary CDF of per-texel `max(R,G,B)`. On `PathTraceResult` rather than `PassRecord` so it rides the existing `shared_ptr` publish under `resultMutex_` with no new synchronisation and is guaranteed to describe the same pass as the pixels on screen, which `PassRecord` cannot promise because it is republished for cancelled passes too
- feat: `reduceOverRange` (`path_trace_driver.cpp`) folds beauty on the driver pool between `accumulateMean` and the publish, per-chunk private histograms concatenated serially after the dispatch -- the reduction shape `buildSubTriangles` already uses, since a histogram bin is far too contended for one atomic increment per texel. Scratch is a driver member, so a steady-state pass allocates nothing, the same convention as `passStats_` and the buffer pool
- note: **the README's suggested cheap fix was rejected on architecture, not taste.** `ThreadPool::parallelFor` over rows from the render thread would have been the first time 2N workers were runnable on N cores: the driver owns a private pool with no accessor, the render thread can only reach `rasterThreadPool`, both are sized `hardware_concurrency()`, and today they are never both busy because selecting a rasterizer AOV suspends the driver. It buys sub-linear speedup by slowing the path trace
- feat: **exposure is factored out rather than baked in**, which is the whole difficulty. Exposure is display-stage only -- not in `PathTraceInputState`, so moving the slider triggers no retrace and `driverLoop` idles once converged -- so a driver-side count at a fixed threshold would freeze on a converged image while the user drags. The peak factors out exactly (`exposure * rawPeak`); the fraction is `aboveBin` read at `1/exposure`. Both stay live under a drag for O(1) render-thread work
- feat: bins are the IEEE-754 binades of `max(R,G,B)` subdivided by the top mantissa bits. Positive floats are monotone under integer comparison of their bit patterns, so a right shift IS a monotone binning with exactly representable edges, at one shift and one subtract per texel rather than a `log2`. Clamped into a +/-16 EV bracket, which covers every threshold the exposure controls can reach (+/-13.95 EV) and is what makes the count exact inside it: a value past either end folds into that end's bin, unambiguously above or below every interior threshold
- note: **the resolution was set by measurement, and the first choice was wrong by 60x.** Against an exact sorted reference over a +/-14 EV sweep on the live image, worst fraction error falls 2.6 -> 0.65 -> 0.18 -> 0.047 percentage points at 4, 6, 8 and 10 mantissa bits kept. The shipped 10 bits (1024 bins per stop, 32768 bins, 128 KiB) lands at **0.026-0.040 pp** at 2048x1152 -- below half the 0.1 pp the HUD's `%.1f%%` can display, so the digit on screen is the digit an exact per-texel count would print. `rawPeak` is bit-identical to the old serial fold, and `aboveBin[0]` equals the texel count on every pass
- note: linear interpolation across the threshold bin was tried and **rejected on measurement**: it plateaus at 0.23 pp no matter how fine the bins get, always at the same threshold. That is the signature of a true delta in the data -- the emissive ceiling panel renders a region of exactly equal radiance -- which interpolation smears and the step estimator resolves once bins are narrow enough to isolate it. Uniform-within-bin is precisely the assumption a rendered image violates
- feat: the complementary CDF is published rather than the histogram it is folded from, so the reader indexes instead of summing. That is what removes the last O(bins) loop from the render thread and, with it, the 4-frame gate the old scan needed: the readout now updates every frame
- feat: `PassRecord::overRangeMs` and an `over-range` row in the dashboard's driver pane -- the work moved threads, it did not become free, and an unmeasured stage would have hidden that. `over-range` on the render-thread side stops being a bursty row and becomes a normal averaged one
- note: **8.950 ms -> 0.000 ms on the render thread** (`-stats`, cornell, 2048x1152, Apple M1, 60 Hz vsync), firing every frame now rather than one in four, against **2.91 ms per pass** on the driver -- 0.2% of a 1666 ms trace. As with the HUD split, the recovered time is **slack, not budget**: `swap (vsync)` goes 4.401 -> 6.729 ms (53.3% -> 81.4%) while `cpu total` barely moves (8.251 -> 8.265 ms) against a 16.67 ms budget. Report it as headroom; there is no fps win to claim
- note: verified image-identical and instrument-only. Linear **RMSE 0, relMSE 0** against a pre-change 16-pass EXR at 512x288 with ray counts identical to the digit (primary 2438784, bounce 4401740, ao 1239162, shadow 3484288), 6/6 validators pass, clean under `-Wall -Wextra -Werror` and clang-tidy. `render_beauty` calls `renderPathTraced` directly and never constructs a driver, so it never runs the reduction at all
- docs: README §5's "Move the over-range scan off the render thread" is replaced by the sibling it leaves behind -- Depth's auto-range maximum (`main.cpp:848-854`), the same serial per-texel render-thread scan, but confined to the display-texture cache-miss path of one AOV and needing the range published from `RasterGBuffer` rather than `PathTraceResult`

## Spectral lambda-draw sizing correction

- docs: README §5 Large #5(a) no longer calls the stochastic per-path wavelength draw a quick, transport-neutral swap. "No transport change" holds only when lambda is drawn exactly proportional to each channel's spectral sensitivity, so that the weight `S_c/p` is a constant absorbed by normalisation; any other density leaves an `S_c/p` factor in the throughput, which is an estimator change and puts `checkTransmissiveSphere`'s per-channel `|Lo - 1| <= 0.03` gate at risk. The repo holds no spectral data to draw from -- six per-wavelength floats in total (`kRgbWavelengthsNm` and the three Fraunhofer lines), no colour-matching curves, no XYZ-to-RGB matrix -- so the entry now names today's model as `S_c = delta(lambda - lambda_c)`, states the CIE 1931 + primaries prerequisite, records that the Rec.709 `r` sensitivity's negative lobe is not a density, and points at the two other roadmap items (§5.1 chromium, §5.2 Kelvin) that need the same CIE module

## HUD stage timing split

The `hud` stage was one number covering both halves of the HUD's cost, which is one number too few to act on: `HudOverlay::render` is unconditional by design (ImGui frame pairing), so a cost sitting there would be paid on every frame with the HUD hidden, and a cost sitting in `draw` would not. The roadmap item asked for the split before any optimisation. This is that split, and the measurement it existed to produce.

- feat: `FrameStageTimes::hudRenderMs` and one `ScopedCpuTimer` around `HudOverlay::render`, reported as `hud build` / `hud render` rows. The new timer is **nested inside** the existing `hudMs` rather than replacing it with two disjoint fields, so the `cpu total` reconciliation sum is untouched and the identity that makes `unaccounted` trustworthy holds by construction instead of by careful editing. `hud build` is then the difference -- the same arithmetic-not-a-second-instrument treatment `present blit` already gets from `presentMs` minus `uploadMs`, and the same `std::max(.., 0.0)` float-rounding guard
- note: **the cost is entirely widget construction, and the concern the split was built to test is disproved.** `hud build` 1.520 ms / `hud render` 0.107 ms with the HUD shown; 0.000 ms / 0.021 ms with it hidden (cornell, 2048x1152, Apple M1, 60 Hz vsync, Release `-march=native` + IPO). `render` is unconditional but not fixed-cost: with no widgets built there is no draw list to submit and it falls to 0.021 ms, so hiding the HUD recovers essentially the whole stage rather than the fraction the roadmap entry feared
- note: the recovered 1.5 ms is **slack, not budget**, which is why the follow-up item is deprioritised rather than opened. Hiding the HUD moves `swap (vsync)` 48.6% -> 68.8% while `cpu total` barely moves (8.576 -> 8.463 ms) against a 16.67 ms budget: the render thread is vsync-bound, so the time comes back as headroom, not frame rate. The HUD's build cost becomes worth attacking only once something else makes the thread CPU-bound
- note: the slider write-back to `DebugCameraController` (`main.cpp`) sits between the two halves and is billed to `hud build`. Deliberate: all five setters are single-field stores, and applying HUD-edited values is the interaction half, not GPU submission. Leaving it between two disjoint scopes instead would have made it unbilled and fed a false residual
- note: verified image-identical and instrument-only. Linear **RMSE 0, relMSE 0** against a pre-change 16-pass EXR, ray counts identical to the digit (primary 2438784, bounce 4401740, ao 1239162, shadow 3484288), 6/6 validators pass. `drawStageRows` lands at 58 lines against clang-tidy's 60-line threshold, so the layout gained a row without the split point having to move
- fix: `kBodyLines` 29 -> 30 for the added row. It reserves the block on the first draw only (every later redraw uses `lastLineCount_`), so a stale value costs one misaligned frame rather than a walking block -- confirmed correct by the captured redraws landing exactly 30 lines apart
- docs: README §5's "Separate the HUD's build cost from its render cost" becomes "Reduce the HUD's build cost", carrying the measured split, the disproved hypothesis, and the vsync-bound reason it is not urgent

## Blue-noise dithered sampling

Sobol+Owen (previous commit) fixed *how fast* the sampler converges. This fixes *where the remaining error lands on screen*, which is a separate axis: at equal RMSE, error correlated between neighbouring pixels reads as far cleaner than error that is independent per pixel. Georgiev & Fajardo 2016. Not a convergence claim, and the measurements below are reported as such.

- feat: `Sampler` applies a per-pixel Cranley-Patterson toroidal shift read from a blue-noise mask tiled over the image, and `hashSeed` loses its pixel arguments entirely. Every pixel now draws the SAME Owen-scrambled Sobol sequence and differs only by that shift -- the paper's construction, in which a random per-pixel offset (equivalently the per-pixel scramble this replaces) is the white-noise special case. The shift is applied in the sampler's own 24-bit output space, `((scrambled >> 8) + dither) & 0xFFFFFF`, where unsigned wraparound IS the toroidal wrap: one add, one mask, exact
- feat: `tools/bluenoise_mask.cpp` -- Ulichney 1993 void-and-cluster, transcribed from the author's paper: wrap-around Gaussian at sigma 1.5 (no truncation radius, so no free parameter), the initial-binary-pattern generator, and all three ranking phases. Emits `src/scene/blue_noise_mask.inc`, 16384 `uint16` ranks, 101 KB, 2.1 s to regenerate. In-repo and deterministic rather than a lifted binary tile, so the provenance is an algorithm that can be re-run and audited; `std::shuffle`/`uniform_int_distribution` are avoided because only mt19937's raw output is specified exactly by the standard
- feat: `power_spectrum.{h,cpp}` -- octave-band radially averaged power spectrum, plus `whiteNoiseBandShare`, the analytic flat-spectrum null. Used by `sampler_validate` to gate the mask and by `render_beauty --error-spectrum` to characterise a render's error. Octaves are the dyadic partition of the frequency axis, so "low-frequency power" is read off a fixed band rather than a cutoff picked after seeing the answer
- feat: `render_beauty --seed` (default 1, so every existing invocation and its bit-identity gate is unchanged) and a ray-count line. The seed matters for correctness of measurement, not just convenience: a reference sharing its seed with the render measured against it also shares that render's exact sample subset, cancelling part of the error being measured -- biasing RMSE low by ~0.8% at 64 of 4096 passes. The reference here is rendered at an independent seed
- fix: **the first implementation of this was wrong, and measurement is what caught it.** Giving every dimension set one shared shift puts a pixel's whole d-dimensional sample vector on the diagonal of the d-torus. A neighbourhood of pixels then integrates the path integrand along a *line* rather than over the torus, so its local mean does not converge to the integral, and the residual -- varying slowly across the image -- is itself low-frequency error. Measured at **199x** white noise in the lowest octave at 16 spp, against 2.7x for the sampler being replaced: the exact opposite of the intent, and plainly visible as blotching. The energy-function argument that justified the diagonal is about the mask's own distances and simply does not cover this. Each dither channel now reads the same mask under its own translation, which decorrelates the components (a blue-noise mask's autocorrelation is near-delta) while each stays exactly the same blue-noise field in screen space
- fix: those translations are placed by the R2 low-discrepancy sequence (Roberts 2018; `2^32/phi2`, `2^32/phi2^2`, phi2 the plastic number), not by a hash. A hash was tried first and is insufficient -- hashed offsets are free to land close together, and two channels inside the mask's correlation radius are correlated, measured at |r| = 0.14 between channels 8 and 23. R2 spreads them by construction: minimum toroidal separation over the 146 channels a 12-bounce path reaches is 8.1 px, where the sigma = 1.5 filter has fallen to ~1e-6. Worst |r| over all 146 is now 0.032
- feat: five new `sampler_validate` checks, four of them exact -- the mask is a permutation of [0, 16384); the shift is rigid across sample index; dither channels decorrelate (the direct regression test for the diagonal defect above, gated at |r| < 0.1, which is ~13 sigma for independent fields and exactly 1.0 for a repeated translation); and the mask's spectrum against the analytic white-noise null, measuring **20401x** low-band suppression. The pre-existing net checks survive unchanged and still assert integer counts with zero tolerance, by undoing the shift exactly: `(rank + 0.5)/2^14` is `rank*1024 + 512` in 24-bit fixed point, so both operands are multiples of 2^-24 below 1 and the subtraction is exact in float32
- note: the white-noise null is analytic rather than a shuffled control, and that was also a measured correction. A sampled null read **eightfold high** because shuffling 16384 elements with `mt19937(1)` reproduces the exact permutation `bluenoise_mask.cpp` uses to place its own initial binary pattern -- the "control" was a rearrangement of the mask by the mask's own generator sequence. A control has to be independent of what it controls for, and white noise's flat spectrum needs no sampling at all
- note: **RMSE is neutral, as predicted.** Against an independent 4096-pass reference at 512x512, linear RMSE moves 0.633 -> 0.605 (1 spp), 0.301 -> 0.291 (4), 0.147 -> 0.146 (16), 0.0727 -> 0.0725 (64) -- the gap closing toward zero with sample count, which is what "same convergence, different arrangement" predicts. At 16 spp over five independent seeds: 0.1477 +/- 0.0012 before, 0.1465 +/- 0.0026 after, a 0.87% difference at t ~ 1.0, statistically indistinguishable. A clear improvement here would have meant the change altered sampling rather than only its screen-space arrangement
- note: the visual result is much stronger than the spectral one, which is the paper's own claim ("appear less noisy, even though their numerical error is roughly the same"). At 1 spp the frame is markedly cleaner -- walls read as solid colour rather than heavy speckle -- for a 4.4% RMSE move. The spectrum at 1 spp shifts power out of the low and mid bands into the top octave (0.79 -> 0.85x white) in 7 of 8 bands. At 16 spp there is no spectral improvement: per-pixel averaging has already washed the arrangement out, and the technique is a low-sample-count one
- note: the low bands read ~2x white even for the sampler being replaced, because a spatially varying variance envelope (the glass spheres, the light, the bright walls) puts low-frequency power in the error periodogram regardless of sampler. That envelope is common to both and dominates the low bands, which is why the sampler's own contribution reads small in this metric
- note: **not free, and faster.** 1711 ms/pass against 1764 (`-stats`, cornell at 1024x576, engines run alone with the kill verified): `hashSeed` sheds two 64-bit multiplies per call and gains a table lookup plus two small multiplies per draw
- note: the Phase-1 claim that ray counts must be identical across a sampler change is **false**, verified rather than assumed. Only `primary` is invariant (270400 in both), because it is purely geometric -- pixels x passes plus the tile halo. `bounce` moves 869262 -> 868307, `shadow` 686381 -> 686277, `ao` 243886 -> 243891: Russian roulette and NEE both consume sampler draws, so different sample values terminate different paths and cross different silhouettes. Total -0.05%
- docs: README §5 drops both completed sampler items and gains the open one (the annealed d-dimensional sample matrix, the principled replacement for translating one scalar mask); references gain Sobol/Joe-Kuo/Bratley-Fox, Georgiev & Fajardo, and Ulichney, and lose the stale PCG32 entry (that fallback no longer exists). `light.cpp`'s comment no longer calls the sampler randomised Halton

## Render telemetry: `-stats` spec block and live dashboard

Four instruments existed and none of them measured the render cycle. `FrameStats` timed the frame boundary to boundary, one `GpuTimer` bracketed the post-process blit, `PathTraceDriver` published a single pass total, and glTF load and BVH build each printed one line. Everything between was dark -- including the two stages that actually stall the render thread, the synchronous raster G-buffer and the full-image display-texture upload. That was load-bearing: README §3's "Reduce AOV-switch restarts" is explicitly blocked on having numbers, and this changelog's own benchmark entry records the reason ("every change in this section is a performance claim and the previous workstream's figures are unreproducible without rebuilding its benchmark from prose"). Three instruments now exist, each the minimum for its concurrency class, and two surfaces read them.

- feat: `render_stats.h` -- `RayCounts`/`PassStats`/`PassRecord`/`FrameStageTimes`/`ScopedCpuTimer`, header-only. Render-thread stages are plain floats written by an RAII scope timer, since writer and reader are the same serial thread and no synchronisation is meaningful; the driver's per-pass phases are a POD copied under a mutex at ~2 Hz; the tile workers' ray counts are **stack-local** per tile, folded into relaxed atomics once per tile. A `fetch_add` per ray was the case where the instrument destroys the measurement -- ~15 M contended RMWs across 8 cores onto 4 cache lines, each requiring exclusive line ownership, is ~450 ms against a ~412 ms pass. pbrt's `thread_local` `StatsAccumulator` was considered and rejected: on Mach-O it costs a `__tlv_get_addr` call on the hottest loop, and its lifetime is the *worker thread*, which outlives the pass, so the reset boundary and the pass boundary differ
- feat: startup **Computer & Engine Spec** block (`spec_report.{h,cpp}`), printed unconditionally after the GL context exists and the scene has loaded, so every field is real rather than a default: GPU/driver/refresh rate, host CPU topology and cache line from `sysctl` (`hw.perflevel{0,1}.*`), compiler/build type/`-march`/IPO/git SHA, library versions, scene load and BVH costs, and the render settings the numbers were produced under. Plain text, no ANSI, no redraw -- it has to survive being piped to a log. Arnold prints the equivalent header at the top of every render for the same reason
- feat: `-stats` live dashboard (`perf_dashboard.{h,cpp}`) -- a 78-column, 29-line base block redrawn in place at 3 Hz by cursor-up (`\x1b[<N>A`) plus per-line erase (`\x1b[2K`), never `\x1b[2J`/`\x1b[H`, which would destroy scrollback and pin the block to the screen origin. Everything including the escapes is `snprintf`'d into a fixed member buffer and emitted as a **single** `::write(STDOUT_FILENO, ...)`, not `std::cout`, whose locale/`num_put` path can allocate and whose partial flushes tear visibly
- feat: the dashboard's cursor-up distance is the newline count the **previous draw actually emitted**, not a layout constant. A constant that drifts from the layout makes the block walk up the screen on every redraw; deriving it means the block can also change height, which is what lets the `?` key map appear and disappear as a section of it
- feat: `-stats` piped to a non-TTY emits one plain summary line every 5 s instead. In-place redraw is meaningless in a pipe and the escapes would corrupt a log file, but a CI or piped run still leaves a progress trace
- feat: Embree device memory monitor (`rtcSetDeviceMemoryMonitorFunction`) accumulating into a relaxed atomic, so `bvh 0.2 MiB` is Embree's own accounting -- the same mechanism its RTCore stats use -- rather than an estimate from triangle count. Device-global, which is exact today (one device, one scene) and stops being exact the moment a second scene is attached; the comment says so
- feat: `FrameStats::percentileMs` reusing the existing 120-entry ring rather than adding a second, so the dashboard leads with p50/p95/max and trails with the mean. A mean hides the hitch by construction -- one 50 ms frame in 120 moves a 60 fps average by 0.4 fps and is exactly what a viewer feels
- feat: `?` prints the key map. Below the spec block at startup and on each press when the dashboard is off; under `-stats` it is a toggled section of the dashboard block beneath its closing rule, because the redraw rewrites every line it owns and a direct print into that region is erased before the eye can catch it. One row list (`hotkeyRows`), two framings, no restatement
- refactor: CLI moves to single-dash throughout, after Arnold's `kick` -- `--scene` is now `-scene`, joined by `-stats`. The tools (`render_beauty`, `raster_bench`) deliberately keep double-dash: they are separate binaries not otherwise touched here, and sweeping them would be reformatting untouched files
- refactor: `PathTraceDriver::lastPassSeconds_` and its accessor deleted; `PassRecord::passMs` is the single source of truth and `PathTracedStatus` reads it. A pass total measured twice is a pass total that can disagree with itself
- note: ray counters ride a `RayCounts& __restrict` out-parameter through `tracePath`, and the direction was **measured, not reasoned**. The plan specified the opposite -- return by value on `TraceResult`, on the argument that under IPO a reference must be assumed to alias `out` -- and that form cost a reproducible **+4.6%** by growing the per-sample sret aggregate 32 bytes; `__restrict` retires the aliasing concern and measured **+0.3%**. The comment records this so a future tidy-up does not reverse it back
- note: every `PassStats` operation is `memory_order_relaxed`, and this is provably sufficient rather than hopefully sufficient. The counters carry no other data, and each worker's stores precede `--workersRemaining_` under `ThreadPool::mutex_`, which `parallelFor`'s `doneCv_.wait` acquires -- that release/acquire pair publishes every tile's contribution before `renderPathTraced` returns, strictly before the driver reads them. The same argument `nextIndex_`/`generation_` already rest on
- note: bursty stages (raster G-buffer, texture upload, over-range scan) print their **last real cost** and a duty cycle, never a per-frame mean, and get no bar. Averaging a 150 ms stall that fires 1 frame in 88 reports 1.7 ms for something that drops a frame every time it runs -- the single most common way a perf HUD lies -- and a share-of-this-frame is either ~100% or zero
- note: the `cpu total` / `frame measured` / `unaccounted` reconciliation is Unreal's `stat unit` device: a ground-truth number the parts must sum to, with the residual **printed** rather than left for the reader to subtract, so a missing instrument becomes visible as a growing gap instead of hiding in rounding. The dashboard's own draw cost is charged to exactly one frame rather than all ~20 in the window, since a redraw happens on roughly 1 frame in 20 and spreading it would inflate `cpu total` ~20x its real share and push the residual negative
- note: `unaccounted` is drawn as a diverging bar centred on zero, because it is the one row that goes both ways and the sign is the information. Its bar runs at a 10% display range rather than 100% -- a healthy residual is ~0.5%, which is one eighth of a cell at full scale and never deflects. A display range, not a threshold: the number beside it is unscaled
- note: `rays/sample` and `mean depth` are arithmetic on the existing counters, not new instruments -- one primary ray per sample gives `total/primary` and `bounce/primary + 1`. No depth accumulator was added; `PathTraceResult::bounceHeatmap` already carries the per-pixel version
- note: postFX is one fullscreen draw, so per-effect GPU cost is not separately measurable without splitting it into real passes. The dashboard reports the whole blit and the spec block lists the active in-shader chain. Inventing per-effect numbers would be a fabrication
- note: `stages` is zeroed at the top of `renderFrame` and that is mandatory, not tidiness. Skip it and `tex upload` reports the last pass's 6.2 ms on every cache-hit frame and `raster gbuffer` reports 150 ms forever after one rasterization -- the most plausible way this system silently starts lying
- note: the ray counts were checked against an identity, not eyeballed. At 1 spp `primary` must equal `renderWidth * renderHeight` plus the tile halo: 2.454 M against 2048x1152 = 2.359 M is +4.0%, matching the 96 px tiling's own overdraw, and `ao` equals `primary` exactly
- note: verified image-identical and cost-free. 5/5 validators pass, `render_beauty --compare` is byte-identical against a pre-change render (0/1769472 channels differ), and an interleaved A/B on user CPU time measured **-0.28%** mean -- inside run-to-run spread, in the direction of noise rather than a regression. If ray counting had been measurable the design would have been wrong
- note: the dashboard perturbs what it measures and this is bounded, not denied. A `::write` to a terminal can block ~50-200 us; at 3 Hz that is ~1 frame in 20, after `swapBuffers`, and its cost is **shown** as the `dashboard` row one refresh late -- the only point at which it can be known -- rather than hidden. `kRefreshHz` above ~5 starts to distort what it reports
- docs: README §2 gains the spec-block and telemetry rows (the old "GPU/system readout" row described the startup print this replaces), §Run documents the single-dash flags and `?`, and §3/§6 gain three roadmap items the instrumentation surfaced and one it closes: the 37.7 MB display-texture upload at 4.0-7.0 ms per published pass, the single-threaded over-range scan at 7.4-9.0 ms one frame in four for two scalars, and the HUD's build and render costs being indistinguishable at 1.3-1.6 ms combined

## GGX distribution magnitude (dielectric Fresnel assertion)

The glass sphere's reflection read far too low beside the chrome sphere. Half of that is correct physics and must not be "fixed": glass at ior 1.5168 has `F0 = 0.0422` against chrome's ~0.55, and on a sphere `F >= 0.4` occupies only the outer 1.44% of the radius, so the bright grazing rim is roughly one pixel at 640x360 before the Blackman-Harris 1.5px filter spreads it. The other half was a real defect, and it was not in the Fresnel term: `distributionGGX` carried two independent errors in its denominator, and no test in the suite could see either. `sampleBsdf` returns `f/pdf` and both carry `D`, so every furnace, reciprocity and round-trip check cancels a constant factor on it; `checkConductorFresnel`'s normalised ratio cancels the geometry term by construction; `checkPdfNormalization` asserted an upper bound only, which a suppressed pdf passes vacuously. `D` is wrong only where it is read absolutely -- NEE's `f` and the MIS pdf weights -- which is exactly the direct highlight the report was about.

- fix: the denominator floor `std::max(kPi*d*d, 1e-8F)` engaged whenever `kPi*alpha^4 < 1e-8`, i.e. for **every roughness below 0.0867**. Measured against a double reference: `D` under-reported by **124340x** at roughness 0.02 (`glass.json`) and **81.5x** at 0.05 (`chrome.json`), unaffected at 0.1 and above. It protected nothing -- callers enforce `alpha >= kMinAlpha = 4e-4`, so `kPi*d*d >= 8e-14`, twenty-four orders above float32 underflow. Defensive noise that was inactive as protection and active as corruption, on exactly the two materials in question
- fix: underneath it, `d = ndotH^2*(alpha^2-1)+1` is catastrophic cancellation -- two near-equal numbers subtracted wherever the half-vector is near the normal, which at low roughness is the entire lobe. Worth a further **20%** of `D` at the peak at `alpha=4e-4` and 3.3e-4 at `alpha=1e-2`, so removing the floor alone would not have made the lobe correct. Now `d = alpha^2*cos^2 + sin^2` (Filament 4.4.2), algebraically identical for a unit half-vector and exact at the peak, taking `sin^2` from the half-vector's own tangential components since `nh` is in the local shading frame. `distributionGGX` takes the half-vector rather than `ndotH`; both call sites already had it
- test: `checkDielectricFresnel` (`bsdf_validate`) reads the lobe's **absolute** magnitude, which nothing else in the suite does. At the coplanar mirrored pair `nh` is exactly `+z`, so `D` collapses to `1/(pi*alpha^2)` and `G2` to `1/(1+2*Lambda(cos))`; `specularGeometry` derives `K = D*G2/(4*cos^2)` in double as an independent reference, and `evaluateBsdfSplit(...).specular / K` is absolute Fresnel reflectance, compared against the existing `referenceDielectricFresnel`. 108 rows over ior `{1.1, 1.5, 1.5168, 2.5}` x roughness `{0.02, 0.05, 0.1}` x cos `{1.0 .. 0.02}`. Worst error **2.57e-7** against a 3e-7 bound, and 2.54e-8 at normal incidence -- where `F(0) = ((n-1)/(n+1))^2` is an exact identity -- against 3e-8. Deliberately couples the check to `D` and `G2` as well as to Fresnel: absolute magnitude is the axis nothing else tests
- test: `checkPdfNormalization`'s estimator re-derived. Restoring `D` broke it for the opposite reason -- at roughness 0.05 the lobe spans ~2e-5 sr, which 200k uniform-hemisphere draws hit a handful of times at O(100) weight each, reading **1.67** against a true bound of 1.0 and still 1.17 at a hundred times the samples. Now multiple importance sampling with the balance heuristic (Veach 1997 sec. 9.2) over uniform and `sampleBsdf`'s own density; since that density **is** `pdfBsdf`, the combined weight collapses to `p/(N1/(2*pi) + N2*p)` -- one pdf evaluation per sample, bounded below by the uniform term, well conditioned at every roughness, and unbiased despite the second density being sub-normalised. Worst integral **1.00069**, so the check is now tight rather than vacuous. Restoring the clamp to make it pass would have been the dishonest fix and was rejected
- note: the picture, `cornell.json` at 640x360, 96 passes. Direct specular on the glass sphere **x4.36** (1.0017 -> 4.3622); on the chrome sphere, the back wall and the floor **x1.000, exactly unchanged**. The asymmetry is the MIS weights, not the lobe: chrome's pdf stayed large enough that BSDF sampling still dominated its direct highlight, where `D` cancels in `f/pdf`, while glass's pdf fell 124340x to the same order as the light pdf and the estimator lost the light-sampling half. Beauty moved 8911/691200 channels, direct specular 1543, indirect specular 2105
- note: the plan predicted the refraction lane would be unchanged, since glass's transmission is the Snell delta branch at `alpha 4e-4 < kSmoothAlpha 1e-3` and never calls `distributionGGX`. It moved: 969/691200 channels, x1.002 over the glass disc. The prediction was too narrow -- `PathBucket` is sticky, so once a path refracts its **whole** remaining contribution lands in the refraction lane, including later specular events on the chrome sphere and the glass's own internal reflections. Same reason the clay walls move x1.008 in indirect specular while their direct specular is bit-identical: inherited illumination, not a direct change
- note: break tests run, not written. Restoring the floor fails the roughness 0.02 and 0.05 rows; restoring the cancelling denominator fails all 108 at 0.8006x, which is the 20% deficit read back directly; Schlick fails 96 of 108, the 12 survivors being exactly the `cos=1` rows where Schlick is exact by construction, over-predicting by **+156%** at ior 1.1 / cos 0.5 and under-predicting by **-21.6%** at ior 1.5168 / cos 0.5 (so the parent plan's "structural under-prediction near grazing" is not what fires); pinning Fresnel to its normal-incidence value fails 96 value rows and 42 monotonicity rows; pinning the sweep to one mid-angle keeps the value assertion but **silently retires** the monotonicity one, which is why the sweep is the test rather than a point sample
- note: asserting on `total()` instead of `.specular` produces **65 spurious failures**, against the plan's prediction that it would pass. At a 3e-7 bound the diffuse substrate is not negligible, so the `BsdfEval` split is load-bearing here rather than cosmetic
- note: measured and deliberately not fixed -- `singleScatter` divides by `std::max(4*wo.z*wi.z, 1e-6F)` while the pdf beside it divides by `std::max(wo.z, 1e-6F)`. The clamp engages at exactly `cos = 5e-4` and darkens one-sidedly as `4c^2/1e-6` below it: 0.64x at `cos 4e-4`, 0.16x at `2e-4`, 0.04x at `1e-4`. Reachable, unlike the `D` floor, but only within 0.029 degrees of tangency -- a ~2e-4 px band on a 40px sphere at 640x360. Recorded in README §5.1 rather than retuned in the dark
- docs: README §5.1 gains the silhouette-clamp item; the reference list gains Filament 4.4.2 as the source of the cancellation-free denominator, beside Heitz 2018 and Heitz 2014

## Ray-traced ambient occlusion (§5 Large item 2)

The AO AOV read solid white. Not a regression: `cornell_v001.gltf` ships 0 textures, so `gltf_loader.cpp` substituted the neutral 1x1 white AO fallback and `rasterizer.cpp` sampled it to 1.0 at every pixel. The AOV was a baked-texture passthrough by design, and a baked map cannot express contact darkening against geometry it was not baked from. AO is now a path-traced accumulator lane, structurally identical to Shadow; the AOV partition moves 12/15 to 13/14.

- feat: cosine-weighted ambient occlusion (Miller 1994; Landis 2002) as `PathTraceResult::ao`, one bounded hemisphere ray per sample at the primary hit. Sampling at `pdf = cos/pi` cancels both the `1/pi` and the cosine, so the estimator is the mean of the visibility term alone -- no weights, no division. **1.0 = unoccluded**, the opposite polarity to `shadow` (1.0 = shadowed), keeping the white-is-open convention of the baked texture it replaces; background (no primary hit) is 1.0
- feat: `PathTracerConfig::aoMaxDistance` (`profile.json`, validated `> 0` at load) bounds the occlusion ray. A hard cutoff, no falloff curve. `0.25` for the Cornell interior is scene scale, not taste: the box is the unit cube `[-0.5,0.5]^3` (glTF accessor bounds), so a quarter-box radius darkens contacts and corners while leaving open wall faces near 1.0
- feat: AO draws from its **own** `Sampler`, seeded `runSeed ^ 0x9E3779B9` and decorrelated by `hashSeed`'s SplitMix64, and `tracePath` receives the drawn `vec2` rather than the sampler -- so AO provably consumes exactly two dimensions and cannot drift. Drawing from the existing sampler would have shifted every later dimension and moved all eight pre-existing images; instead Beauty is bit-identical (`--compare`, 0/691200 channels differ) across every step of this work
- refactor: `sampleCosineHemisphere` promoted out of `bsdf.cpp`'s anonymous namespace rather than written a second time, following the precedent `fresnelAtViewAngle` set when an AOV needed it
- refactor: baked AO removed from the rasterizer -- `RasterGBuffer::ao`, its `sampleBilinear(material.aoTexture, ...)` and its `writeTexel` are gone, `aovImages`' compile-time size drops 15 to 14 in both the return type and the local, and `main.cpp` routes `AovId::AO` through `fromSnapshot` like Shadow. `Material::aoTexture` and its glTF read stay, now flagged display-dead in `material.h`: the asset pipeline stays lossless and a future baked/traced blend has the map to hand
- feat: `render_beauty --aov <name>` captures any of the nine `PathTraceResult` lanes headlessly -- the AO lane had no reader outside the viewer otherwise. Names resolve through `AovId`/`kAovNames` rather than a private list, case- and separator-insensitively, so the CLI and the HUD dropdown share one vocabulary; rasterizer-backed and post-filter AOVs are rejected with the reason. `encodeForDisplay` now mirrors `presentFrame`'s `isBeauty ? userLut : Raw`, since a display curve over a data AOV would distort values that are already display-ready
- test: `checkAmbientOcclusionAnalytic` (`integrator_validate`) against a closed form with no Monte Carlo in it. Malley's method (PBR 4th ed. 13.6.3) makes the sampled direction's tangential projection uniform on the unit disk, and a ray reaches an infinite perpendicular wall at `t = d/wx`, so it is occluded exactly when `wx >= c = d/D` -- a circular segment of area `acos(c) - c*sqrt(1-c^2)`, giving `AO(c) = 1 - (acos(c) - c*sqrt(1-c^2))/pi`, independent of how `buildShadingFrame` rotates the tangent frame. An unoccluded plane reads exactly 1.0; the corner scene is swept over `c = 0.05/0.25/0.50/0.75`, measured 0.5341/0.6597/0.8062/0.9285 against analytic 0.5318/0.6575/0.8045/0.9279
- test: `checkAmbientOcclusionDistanceBound` brackets `aoMaxDistance` from both sides on one fixed geometry, so the only thing changing between rows is the bound: `c = 0.90` reads 0.9814 against analytic 0.9813, and `c = 1.10` reads exactly 1.0
- note: tolerances come from the estimator's own statistics, not a hand-picked constant -- binomial standard error on the visibility fraction at 6 sigma, `6*sqrt(p(1-p)/N)`, the same device `nee_validate`'s quad-solid-angle check uses. At `p = 1` this is exactly zero, which is correct rather than degenerate: every sample is then deterministically unoccluded, so the AO lane and the filter-weight lane accumulate the identical weight sequence and the write-out quotient is bit-exactly 1.0
- note: the plan's assumption that `makeCornerScene`'s wall at `x = 1` is spatially near-uniform across the measured region was wrong, and the checks are built on the correction. `AO(c)` is nonlinear, so the region's spread in `d` biases its mean by `AO''(c)*Var(d/D)/2` -- ~2e-3 at the `c = 0.9` row, which would have dominated its tolerance -- and decisively, the `c = 1.1` row's nearest sample sat at `c = 0.976 < 1`, so it would not have read exactly 1.0. A defaulted `wallX` pushes the AO checks' wall to 10, dropping the bias to 1.6e-4 and putting the frame-edge sample at `c = 1.067`
- note: measuring the whole frame rather than `centreMean`'s 4x4 block cut the traced paths 16x for the same N -- AO is a visibility query about a constant surface normal and does not depend on the view direction, so the spread `centreMean` exists to limit costs nothing here. Spending part of that on 4x the samples halved every tolerance
- note: `c = 0.05` is honest ballast. It demonstrates the `c -> 0` half-space limit 0.5, but separates the cosine-weighted curve from uniform-hemisphere sampling's `(1+c)/2` by only 0.0068, so it discriminates at no practical N. That is why the sweep carries four rows: at `c = 0.25` the gap is 0.033, 5.8x the tolerance
- note: break tests run rather than written. Inverting the polarity fails all 7 rows; ignoring `aoMaxDistance` fails all 6 corner rows at 0.5078, the half-space limit up to the wall's finite extent; uniform-hemisphere sampling under the same estimator fails 4 of 6, measuring 0.6267/0.7513/0.8756/0.9501 against the predicted `(1+c)/2`. Substituting `frame.normal` for `geoNormal` in the ray-origin offset **fails nothing** -- named as a gap: flat test quads make `shadowTerminatorOffset` a no-op, so the suite structurally cannot see that substitution
- note: the AO render shows what the estimator predicts -- contact darkening in the box corners and where each sphere meets the floor, ceiling and open wall faces near 1.0, the glass and chrome spheres reading as ordinary occluders (AO is pure visibility, blind to material), the emitter panel dark where it meets the ceiling, and no light-panel influence anywhere. One Bernoulli draw per sample leaves visible grain at 96 passes, converging as `1/sqrt(N)`
- docs: README §1/§4/§5/§6 -- the 15/14 rasterizer counts, the rewritten AO row, the roadmap's ray-traced-AO clause dropped from Large item 2, and Miller 1994 promoted out of "not yet implemented" with Landis 2002 added beside it

## Area-light hardening (§5 Large item 2)

A review pass over the area-light feature above, against Ureña/Fajardo/King 2013 and PBR 4th ed. Sec 12.5.3. The sampler math, MIS weighting and emitter-hit ordering held up; the scene-authoring boundary did not.

- fix: `loadSceneConfig` accepted any `lights` entry with no semantic check at all. The spherical-rectangle sampler is exact only for a rectangle, and both `light.h` and `scene_config.h` documented `edge0` perpendicular to `edge1` as a precondition while nothing enforced it -- a skewed quad builds a non-orthonormal frame, so `sample()`'s reconstruction `p + xu*x + yv*y + z0*z` lands off the parallelogram the BVH actually holds and direction, distance and pdf all describe a surface that is not there. Wrong energy, no error, no test failure. Now rejected at load, with `"type"` (previously parsed by nobody, so `"type": "sphere"` silently built a quad), zero-length edges, and negative intensity/colour
- note: the perpendicularity bound is the sampler's own worst-case positional error rather than an authoring tolerance -- the far corner deviates by `|edge1|*|cos|`, so `1e-4` keeps that under `kRayEpsilon` (1e-4 world units) for any sub-unit light, i.e. a sampled point never lands further off the surface than the ray offset that must clear it; `1e-3` would put it past that epsilon and let the light self-occlude. What it costs authors was measured, not assumed: over 20k random 3D rotations of a 0.3 x 0.2 rectangle the worst `|cos|` by authored decimal places is 6.1e-3 at 3dp, **5.7e-4 at 4dp**, 6.0e-6 at 6dp, 9.9e-8 at 8dp -- so an off-axis quad needs 6+ decimals (what any DCC exporter writes) and a hand-rounded 4dp one is rejected by design
- fix: the emitter-hit MIS weight evaluated the light pdf at `ray.origin`, the epsilon-offset continuation origin, while NEE sampled from `shading.position`. Solid angle measured from two points ~1e-4 apart differs, so the two strategies' weights did not sum to exactly 1 -- a small real bias, O(eps/d). `tracePath` now carries `lastShadingPosition` alongside `lastBsdfPdf`/`lastSampleWasDelta`, the three moving as a unit; pbrt carries the previous interaction for the same reason
- fix: `LightSet::sample` returned from the degenerate-quad case after drawing the 1D light selection but before the 2D, shifting every later Halton dimension on that path relative to its neighbours -- randomized Halton's low-discrepancy property (`sampler.h`) rests on paths agreeing which dimension is which. The 2D is now drawn unconditionally. Not theoretical: isolated to 6 changed pixels on `cornell.json`, so the degenerate branch is genuinely reachable there
- refactor: `buildQuadLights` (`material_binding.h`/`.cpp`) replaces the config-to-world quad transform duplicated verbatim in `main.cpp` and `render_beauty.cpp`, following the precedent `resolvePerInstanceSettings` already set in that file -- a headless comparison render must place lights identically to what ships, and two copies of a transform convention drift
- refactor: `QuadLight` gained a constructor caching `normal`, which `quadRadianceToward` recomputed as `normalize(cross(edge0, edge1))` on every NEE sample and every emitter hit, and `appendQuadLights` computed again. All six construction sites are 4- or 5-arg brace-init and bind to it unchanged
- test: `checkQuadLightInverseSquare` (`integrator_validate`) asserts what nothing did. Quadratic falloff is *implicit* here -- NEE divides by `pdf = selectionPdf/solidAngle`, so the subtended solid angle enters as a multiplier and shrinks as `A*cos_l/d^2`; there is deliberately no explicit `1/d^2` term, which is needed only under uniform-*area* sampling where the geometry term converts the measure. Measured off the rendered irradiance, `E*d^2/(L*A*cos_r*cos_l)` reads 1.020 / 0.993 / 0.996 / 0.9987 at d = 0.71 / 1.12 / 2.06 / 4.03
- note: that check's near-field row is a conditioning guard, not decoration: a light 30x wider at half the nearest distance reads 7.72 on the same ratio, so a renderer that had hardcoded a `1/d^2` point light would fail there while passing every far-field row. The check also asserts the rendered value against the independent Lambert polygon irradiance at *every* distance, near field included, so the two regimes cannot be confused
- test: `appendLightGeometry` took the quad index instead of hardcoding 0, which silently made it correct only for single-light test scenes
- note: break tests run rather than written. An extra explicit `1/d^2` in NEE fails `checkQuadLightInverseSquare` 8 times, the ratio decaying to 0.234 then 0.061 as `1/d^4`; a 10-degree-skewed quad, a negative intensity, a zero edge and `"type": "sphere"` are each rejected at load with a named error (the skew reporting `cos 0.17369`, exactly sin(10 deg)). **Reverting the MIS reference-point fix fails nothing** -- named as a gap rather than passed over: that fix rests on construction (pbrt's convention), not on coverage, and the suite structurally cannot resolve an O(eps/d) bias
- note: `cornell.json` moves 64 of 691200 channels by 1/255, scattered as isolated pixels with no structured change to lighting, shadows or colour bleed -- the MIS and sampler-dimension fixes, both below visual threshold. `tree.json` is bit-identical to HEAD at matching parameters. The 15 remaining clang-tidy warnings were checked against HEAD rather than assumed pre-existing; splitting `parseQuadLights` out removed one
- docs: README reconciled against the roadmap triage that landed in parallel -- the Large chain's renumbering (Cornell box item dropped) propagated to four cross-references, and the new Light temperature item rewritten, since it was authored against a codebase with "no synthetic light source" that now has one

## Global illumination: area lights (§5 Large item 2)

A rectangular area light: its own geometry in the BVH (occluding, camera- and BSDF-hittable), solid-angle NEE sampling, MIS-combined with BSDF sampling exactly like the environment. The first light type beyond the environment map, and the abstraction (`LightSet`) the rest of that roadmap item -- ReSTIR, ray-traced AO, mesh lights -- builds on.

- refactor: `LightSet` (`include/engine/scene/light.h`) generalises NEE from a hard-wired `EnvironmentMap` reference to a uniformly-selected set of zero or more lights. Every `renderPathTraced`/`tracePath` call site threads `LightSet` + `instanceLightIndex` (parallel to `instances`, -1 for ordinary geometry) instead of `EnvironmentMap&`/`envRotationRadians`/`envExposure` directly. Selection draws no sampler dimension when exactly one light is present (a degenerate one-element categorical needs no random variate), which is what makes this step a true no-op: verified bit-identical (MD5 match) on `cornell.json` (0/691200 channels differ, 96 passes) and `tree.json` (0/172800, 24 passes)
- feat: `QuadLight` + `SphericalRectangle`, Ureña, Fajardo & King 2013's constant-solid-angle-density spherical-rectangle sampler (as given in Pharr/Jakob/Humphreys, *PBR* 4th ed. Sec 12.5.3) -- the same sampler Arnold's own `quad_light` uses. Landed and unit-tested (`nee_validate`'s `checkQuadLightSolidAngle` against an independent Monte Carlo ray/rectangle-intersection oracle, `checkQuadLightMisAgreement` against a brute-force hemisphere reference) one commit before any scene authored one, matching the review's own "test coverage precedes visible feature" discipline
- feat: `appendQuadLights` (`gltf_loader.h`/`.cpp`) injects a light's own two emitting triangles + one `MeshInstance` into a `LoadedModel`, using a new shared `makeDefaultMaterial()` (`material.h`/`.cpp`, promoted out of `gltf_loader.cpp`'s six inline default-texture functions, now called from both sites) as the emitter's inert fallback material -- never read by the path tracer (the emitter-hit branch returns before `resolveBsdfParams`) but needed by the CPU rasterizer, which walks every triangle unconditionally for the primary-hit AOVs
- feat: `SceneConfig::lights` (`QuadLightConfig`: origin/edge0/edge1/color/intensity/twoSided) and `EnvironmentConfig::lightEnabled`, both optional keys defaulting to today's behaviour (no lights, environment always a light) so every existing `scene.json` loads unchanged
- feat: HUD "Environment Light" checkbox (`AppResources::envLightEnabled`, threaded through `PathTraceInputState`/`PathTraceDriver::Request`) removes the environment from `LightSet` entirely -- NEE, MIS, and every miss's radiance, background included -- distinct from the pre-existing "Show/Hide Background", which only ever gates the camera ray's own miss. `render_beauty --env-light 0|1` is the headless equivalent, overriding the scene's own authored default
- feat: `assets/scenes/cornell.json` authors a ceiling panel (warm white, `color [1.0, 0.85, 0.65]`, `intensity 15.0`), landing the still-blocked classic Goral et al. 1984 Cornell box -- reachable by toggling the environment light off
- fix: the emitter-hit branch was originally placed *after* `tracePath`'s depth-cap break, so it never ran on the terminal "extra" bounce the miss branch is specifically traced one iteration past `maxBounces` to collect (see the existing depth-cap comment) -- silently dropping the BSDF-sampled half of the MIS estimator for any path whose last allowed bounce hits a light rather than missing entirely. Moved before the depth cap, symmetric with the miss branch. Caught while writing `checkQuadLightIrradianceOneSidedOcclusion` at `maxBounces 0`, where it zeroed the estimator outright rather than just biasing it
- fix: `kShadowDistanceEpsilon`, a `1e-3` relative back-off on a finite light's NEE shadow-ray `tMax` (pbrt's `ShadowEpsilon` convention) -- needed now that a light's own triangles sit in the BVH, where an unshortened `tMax` let the light's front face register as its own occluder at `t == distance`. A no-op for the environment (`distance == FLT_MAX`; `FLT_MAX * (1 - 1e-3)` is still a normal, effectively-unbounded float), so one formula covers both light kinds with no branch
- test: `checkQuadLightIrradianceOneSidedOcclusion` (`integrator_validate`), an exact analytic reference with no Monte Carlo in it at all: a Lambertian receiver (`ior 1.0` zeroes the dielectric specular lobe exactly, `diffuseRoughness 0` is EON's own Lambertian limit) under a one-sided rectangular emitter must read exactly `Lo = (baseColor/pi) * E`, `E` Lambert's 1760 / Baum-Rushmeier-Winget 1989 closed-form polygon irradiance -- a genuinely different formula from `SphericalRectangle`'s own solid angle (Girard's theorem on internal vertex angles vs. summing the projected solid angle over edges), so the two cannot share a transcription bug. Three light positions/sizes agree to Monte Carlo noise (~0.1-0.3% of reference); a one-sided emitter's back face and an opaque wall between light and receiver both read exactly 0
- note: the test camera looks straight down at the receiver, and every light position in the check above had to be offset off that exact sightline -- a one-sided, downward-facing light centred on it puts its own non-emitting back face directly in the camera's primary ray, which is correct renderer behaviour (a real one-sided panel viewed from behind is black) but reads as "rendered 0" until diagnosed as a test-construction issue rather than a transport bug
- note: three break tests confirm real (not incidental) coverage. Flipping `quadRadianceToward`'s front-face sign fails 4 of `checkQuadLightIrradianceOneSidedOcclusion`'s 5 sub-checks (the fifth, occlusion, is orthogonal to front-face orientation and correctly stays green). Neutralising `kShadowDistanceEpsilon` collapses all three irradiance cases toward 0 (measured 0.4-6% of the unbroken reading), confirming self-occlusion is what the fix prevents rather than a defensive no-op. A sign error in `SphericalRectangle`'s Girard's-theorem solid angle, introduced and reverted while developing it, was caught by `checkQuadLightSolidAngle` at an 8x discrepancy (6.28 vs 0.80 sr) against its independent Monte Carlo oracle
- docs: README §1/§2/§4/§5/§6 updated -- `LightSet` in the pipeline description, an Area lights + Environment-light-toggle component row, the Shadow AOV's "sampled light" wording, §5 Large item 3 (now item 2 after the roadmap triage renumber) marked landed (ReSTIR reworded: it needs *multiple* lights to resample across, which one area light plus the environment isn't yet), Ureña/Fajardo/King 2013 and Lambert 1760/Baum-Rushmeier-Winget 1989 added to §6
- note: still deferred -- ReSTIR (needs many lights), ray-traced AO (still a baked-texture AOV), emissive-mesh lights (the light most naturally slots in as a second `LightSet` variant), disk/sphere lights, power-weighted light selection, photometric lumen/candela units, a separate Emission transport-AOV lane (would touch the fixed 20-lane per-tile accumulator in 4 places). The interactive viewer's `0`-key reset (already a filed Quick roadmap gap) now also omits `envLightEnabled`, extending rather than introducing that gap

## Shading pipeline: the transmission-tint convention (F13)

Closes the last finding of the review above. `baseColor` multiplied the transmitted value at three sites in `bsdf.cpp` while `transmissionColor` was Beer-Lambert extinction in `path_tracer.cpp`, so two independently authored knobs tinted the same transmitted light under two different conventions. Settled from the specs, which disagree only because of what parameters each one has: glTF `KHR_materials_transmission` tints with `baseColor` for want of a transmission colour of its own, being scoped to "infinitely thin surfaces" whose absorption "is constant and equal to `1.0 - baseColor`", while OpenPBR and Arnold both make `transmission_color` the sole tint and define `base_color` as "the observed reflection color (viewed at normal incidence under uniform illumination)". This pipeline follows OpenPBR -- the standard it already takes `edgeTint`, EON, `base_color`'s meaning (F11) and its RGB wavelength triple (F12) from -- so using an observed *reflection* colour to tint *transmission* was a category error against a definition the engine already shipped.

- fix: `BsdfParams::transmissionTint` replaces `params.baseColor` at all three transmission sites (`transmitMultiScatter`, `evaluateTransmissionLobe`, and `sampleBsdf`'s smooth delta branch). It has to live on `BsdfParams` rather than in `path_tracer.cpp`: far-side NEE evaluates the transmission lobe through `evaluateBsdfSplit`, so a tint applied only at transmission *events* would miss it. `baseColor` keeps every reflection-side role -- the diffuse lobe through `diffuseRho`, the conductor's `f0`, and the rasterizer's albedo AOV -- and loses only this one
- fix: `transmissionDepth == 0` is a live defect fixed, not a regime left undefined. It is Arnold's and OpenPBR's own default and means "no interior medium, tint at the surface"; `sigmaAFromTransmission` floored it to `1e-4` and returned `-ln(c)/1e-4`, rendering any coloured depth-0 material black. Measured on cornell: `transmissionColor [0.5, 0.8, 1.0]` at depth 0 annihilated the sphere's transmitted red and green (interior minimum `0/0`) where the tint should have been a light cyan. `resolveBsdfParams` now selects the regime -- `transmissionDepth > 0` gives white here and Beer-Lambert in the medium, `0` gives the on-surface tint and no absorption -- which is the spec's own definition of the parameter's domain, not a special case. The two cannot be unified branch-free: the `depth -> 0` limit of the volumetric form is total blackness, which is why OpenPBR branches too
- fix: `sigmaAFromTransmission` returns exactly zero at `depth <= 0` rather than skipping the medium, so `tracePath`'s enter/exit toggle stays symmetric across a depth-0 interface -- skipping the enter would desynchronise the medium stack on the exit, which is the F3-root failure mode
- feat: `transmissionDepth` defaults to `0.0` (was `1.0`), matching Arnold and OpenPBR. A no-op for every material in the tree, and it stops the engine's default diverging from the convention it now documents
- test: `checkTransmissionTint` (`bsdf_validate`), the only instrument that can see any of this. Two complementary assertions over 12 rows: independence of `baseColor`, asserted **exactly** -- at `metallic 0` it reaches `f0` not at all and the diffuse lobe is identically zero on the far side, so no term may move by one bit -- and exact linearity in the tint, which carries a 1e-6 relative band only because the rough lobe sums a single-scattering and a multiple-scattering term and FP multiplication does not distribute over a sum. Rows cover both code paths a tint travels: the rough continuous lobe and the smooth delta branch
- test: `checkOnSurfaceTransmissionTint` (`integrator_validate`) asserts the contract with no formula restated, in `checkBeerLambert`'s style: an index-matched (`ior 1.0`) slab and sphere at `transmissionDepth 0` under uniform `L0 = 1` must read exactly `transmissionColor^2`, one factor per interface crossed. The two shapes are the point -- their traversal distances differ by construction, so reading the **same** square on both is what distinguishes an on-surface tint from any absorption. Measured `[0.249918, 0.0624794, 0.562316]` against `[0.25, 0.0625, 0.5625]`, identical on both, a 3.3e-4 residual with none of the path-deficit bias that makes `checkBeerLambert`'s sphere row read high
- test: `checkBeerLambert` is the guard on the *other* regime and the pair is complementary, not redundant: it authors a colour at `depth > 0` and requires exactly `transmissionColor`, so a tint leaking into the volumetric regime reads `colour^3` there. Verified by breaking -- dropping the depth ternary fails it 12 times (all four rows x three channels) while the on-surface check stays green, and restoring the old `1e-4` floor fails the on-surface check 6 times while Beer-Lambert stays green
- note: **the pre-existing suite is completely blind to the convention**, measured rather than assumed. `bsdf_validate`'s `makeParams` hardcodes `baseColor 1` and every transmissive case in `integrator_validate` leaves `transmissionColor` white, so both tints are exactly the identity everywhere the old suite looked. Reverting all three sites leaves the new checks failing 72 and 6 assertions and every other row byte-identical; the two new checks carry the entire proof
- note: cornell is **bit-identical** (0/691200, md5 unchanged) -- `glass.json` authors `diffuseColour` white and `transmissionDepth 0.4 > 0`, so both tints are the identity. Demonstrated instead on two throwaway `glass.json` probes. A coloured `diffuseColour [1.0, 0.4, 0.4]` tinted 48.4% of the frame before and **nothing at all** after: the probe render is now bit-identical to the shipped cornell, which is the stronger statement that on a fully transmissive material `baseColor` reaches nothing. Its red channel moved **0 of 230400 pixels** in both directions, confirming that no lobe-selection probability consumes a colour and so this change cannot decorrelate the sampler -- unlike F12's
- note: the depth-0 probe (`transmissionColor [0.5, 0.8, 1.0]`, `transmissionDepth 0`) recovers the sphere from black to a clean cyan, and the picture agrees with the validator quantitatively: through the display transform the interior reads 0.55 and 0.79 of untinted in red and green against `0.25^(1/2.2) = 0.53` and `0.64^(1/2.2) = 0.81`, which one crossing rather than two (0.73 in red) would not give. Its blue channel is an internal control at `transmissionColor 1.0`, the identity in both regimes, and moved 0 pixels
- note: **one pre-existing row moved that was predicted not to**, and it is FP association rather than a value change. `checkTransportPartition`'s `slab dispersive glass` residual reads 2.22979e-06 where it read 2.35366e-06. That scene's `diffuseColour`, `baseColorTexture` and `transmissionColor` are all exactly `1.0f`, so the multiplicand is bitwise identical and only the load site differs; it is a cancellation *residual*, the most FP-sensitive statistic in the suite, and both values sit 45x under its 1e-4 bound. Isolated by probe rather than argued: adding the struct field while leaving the three reads on `baseColor` restores the baseline byte-for-byte
- note: two coverage gaps found by breaking, named rather than passed over. Tinting **only** the delta branch fails `bsdf_validate` 54 times and leaves `integrator_validate` green, because `checkOnSurfaceTransmissionTint`'s slab and sphere are authored at roughness 0.02 (`alpha 4e-4`, below the smooth threshold) and so exercise the delta path alone; the rough path's coverage is entirely `bsdf_validate`'s. Reverting **only** `transmitMultiScatter` fails 44 assertions, catching 8 of the 9 rough rows but not `roughness 0.3` at `ndotV 0.95`, where the Kulla-Conty deficit feeding that lobe is smallest
- docs: F13 closes the review's filed backlog, so everything still open moves to the README §5 roadmap rather than living only in this file -- the two coverage gaps above, an instrument that can see the coat half of `F_avg` (reverting it leaves all five validators green), and a CIE-integrated fit for the chromium triples, joining the EON `diffuseKd` shape-gating, the occlusion-sensitive curved transmissive test and the spectral-transport items already recorded there

## Shading pipeline: dispersion (F12)

Continues the review above, and closes the next item its `note:` deferred. Glass authored a single scalar `ior` shared by all three channels, so no prism, fringe or coloured caustic edge was representable at all. Hero-channel dispersion: a path commits to one RGB channel at its first dispersive interface and stays on it, which is the standard RGB-renderer construction and the only one that does not require a spectral integrator.

- feat: `cauchyIor` -- Cauchy's `n(lambda) = A + B/lambda^2` with `(A, B)` inverted from an authored `(ior, abbe)`, per Khronos `KHR_materials_dispersion`. `B` follows from applying the Abbe definition `V_d = (n_d-1)/(n_F-n_C)` to the Cauchy form, `A` from pinning `n(lambda_d) = n_d`. Written in that general form rather than the spec's composite one, which pre-multiplies `1/(lambda_F^-2 - lambda_C^-2)` into a literal `523655` and hides both Fraunhofer lines inside it. Two terms is the right order: the material supplies exactly two numbers, so Sellmeier would have to invent its remaining coefficients
- feat: `MaterialConfig::abbe`, optional-with-default `0.0` = no dispersion, matching Arnold's `transmission_dispersion_abbe` and OpenPBR's dispersion scale. `abbe <= 0` returns the authored `ior` bit-for-bit, which is what keeps every existing material and every existing render unchanged. No range clamp on the result: `n_d = 1` already gives `B = 0` exactly, so an index-matched medium is non-dispersive out of the algebra rather than by special case
- feat: `kRgbWavelengthsNm` = 620/540/450 nm, from `OpenPBR_BaseRgbWavelengths_nm` in Adobe's OpenPBR BSDF reference implementation -- the same standard this pipeline already takes `edgeTint` and EON from, rather than a fresh convention. Using the Fraunhofer d/F/C indices directly as the R/G/B indices was considered and rejected: it needs no wavelength constants at all, but 656/588/486 nm are measurement lines rather than channel centres, understating the B-R spread by ~18% and foreclosing the stochastic-wavelength upgrade
- feat: the hero channel is chosen in `tracePath` at the first vertex whose material is dispersive and transmissive, *before* any BSDF work there, because the whole interaction is wavelength dependent -- Fresnel and the lobe probabilities as much as the refraction direction. Dispersion then enters the shading through the existing scalar `BsdfParams::ior`, resolved per channel in `resolveBsdfParams`, so every downstream `ior` consumer (Fresnel, lobe probabilities, escape-albedo tables, refraction) is spectrally consistent through one value and `bsdf.cpp` needs no change at all. Committing at the interface rather than at path start, as a spectral hero-wavelength renderer must, is strictly cheaper: a path that never meets dispersive glass keeps full RGB
- feat: one-sample channel estimator with probabilities proportional to the throughput carried so far (OpenPBR implementation paper, arXiv:2512.23696), not the uniform `1/3` this was filed as. The surviving channel takes `T_c/p_c`, which is `sum(T)` for **every** `c`, so the path's magnitude -- and hence Russian roulette's continuation probability -- no longer depends on which channel was drawn
- feat: `assets/materials/glass.json` is now Schott **N-BK7** (`ior 1.5168`, `abbe 64.17`), a real catalogue glass rather than an invented pair; `principled.json` gains `abbe: 0.0` as the full parameter surface
- test: `checkCauchyDispersion` asserts the contract rather than restating the formula -- `n(lambda_d) == n_d`, `n_F - n_C == (ior-1)/abbe` (the Abbe definition itself, recomputed in the test from the authored inputs), normal-dispersion ordering across the shipped wavelength triple, and the `abbe = 0` no-op bit-for-bit. Swept over N-BK7, SF10, water and diamond. Verified by breaking: dropping `B/lambda^2` fails all three assertions on all four materials (12 failures), swapping `lambda_F`/`lambda_C` fails Abbe difference and ordering but **not** the d line, since `A` compensates -- so the assertions are complementary, not redundant
- test: `checkTransmissiveSphere` gains dispersive rows, and they are the gate on the estimator rather than on the optics: a white non-absorbing sphere is invisible whatever its index, and a wavelength-varying index is still an index at each wavelength. Read per channel rather than through `renderCentre`'s max, which would hide a per-channel loss behind whichever channel read highest. N-BK7 reads `[0.995, 0.994, 0.993]` and SF10 `[0.995, 0.995, 0.995]`, both inside the non-dispersive rows' own 0.988-0.996. Reverting the `1/p` rescale to a bare mask reads `0.331` in every channel
- test: `checkBeerLambert` gains a dispersive row that isolates the channel estimator from the refraction geometry. At `ior 1.0` the Cauchy `B` is exactly 0, so `n(lambda) = 1` at every wavelength and the interface stays exactly index-matched -- ray path, traversal distance and expected reading all unchanged -- while hero-channel selection still fires, since it keys on `abbe` and `transmissionFactor` alone. What it asserts is that the hero channel and the already-per-channel `sigmaA` do **not** interact: the surviving channel is attenuated by its own `sigma_a` and the masked channels stay exactly zero, so the reading is still the authored `transmissionColor`. Reverting the rescale reads exactly one third of it
- test: `checkTransportPartition` gains a dispersive slab row. The hero channel is the one mechanism that writes different values into different channels of the same throughput -- it zeroes two of them -- so a mask applied to beauty but not to the bucket accumulators breaks the per-channel identity in exactly the two zeroed channels, which no furnace can see: a furnace measures the total that arrives, and this asks where it was filed. Gap 2.4e-06 against a 1e-4 bound
- note: **the throughput-weighted choice is not measurable by this suite**, and that is stated rather than implied. Substituting uniform `1/3` selection leaves `integrator_validate` **byte-identical**, because every scene in it presents an exactly grey throughput at the first dispersive vertex, where the two rules coincide. The choice rests on the literature and on tinted glass, not on a measurement made here
- note: hero-channel **stickiness is a property of the estimator, not of the guard that appears to enforce it**, measured rather than assumed. Removing the `heroChannel` guard so the channel is re-drawn at every dispersive vertex leaves the whole suite green: masking leaves two channels exactly zero, so a throughput-weighted redraw can only return the same channel -- 400k redraws instrumented, **zero** changed. The guard buys one fewer sampler dimension per later crossing; under uniform selection it would be load-bearing
- note: the extra `Sampler::next1D()` shifts every downstream Halton dimension on paths that take it, so authoring an `abbe` decorrelates the sample sequence as well as bending the light. Measured separately rather than assumed away: at `abbe 10000` (`dn(B-R)` 6e-5, optically inert) cornell moves 48.4% of pixels at RMS ~3.6-4.2 with a near-zero signed mean, which is the whole of the decorrelation term. Against that reference the shipped N-BK7 moves 14.7% of pixels with the predicted per-channel ordering -- B rms 1.12 / 22346 channels, G 0.53 / 13714, R 0.41 / 9256 -- blue's index deviating furthest from `n_d` (+0.0086) and red's barely at all (-0.0012). An SF10 probe at 3.2x the spread scales the same signature the same way
- note: committing a path to one channel costs 3x the colour variance on every path through dispersive glass, plainly visible as chromatic noise in cornell's sphere at 96 passes. That is the technique's real cost, not a defect, and it is what the stochastic-wavelength upgrade in §5's spectral item also improves
- note: still deferred -- the `baseColor`/`transmissionColor` transmission-tint overlap (now *decidable*, since this settles how glass is authored), stochastic per-path wavelengths, full spectral transport, EON shape-gating on `diffuseKd`, an occlusion-sensitive curved transmissive test, and a CIE-integrated fit for the chromium values

## Shading pipeline: EON albedo inversion (F11)

Continues the review above, and closes the first of the items its `note:` deferred. `evaluateDiffuseLobe` handed the authored `baseColor` to `evaluateEon` as the single-scattering albedo `rho` directly -- the paper's own "simpler alternative", as the code comment said. EON's multiple-scattering term then saturates, so the albedo the renderer is *observed* to have drifts below the albedo authored: measured **-12.2%** at `diffuseColour 0.5`, `diffuseRoughness 1`, and **-20.6%** on the darkest channel of `[0.8, 0.3, 0.1]`. Exactly zero at `rho=1` and at `r=0`, which is why nothing in the pipeline could see it.

- feat: `eonAlbedoInversion` ports Appendix A of the revised paper (JCGT 14(1), **revised 2026-02-04**, which added this appendix). Eq. 29's normal-incidence FON albedos `E_F(N)` and `<E_F>` reduce eq. 28 to the quadratic `a*rho^2 + b*rho - C = 0` with eq. 31's coefficients; solving it gives the `rho` whose observed albedo at normal incidence *is* the authored colour. That is OpenPBR's reading of `base_color` -- "the observed reflection color (viewed at normal incidence under uniform illumination) in areas where the Fresnel reflection is negligible" -- which OpenPBR **declares but does not enforce**, setting `rho = C` directly. The authors ship no inversion code (`portsmouth/EON-diffuse` has none), so this is a transcription of the equations, not of a listing
- fix: eq. 30 states the **unstable** root `(-b + sqrt(b*b+4ac))/(2a)`: `a` is proportional to `r`, so as `r -> 0` a vanishing denominator divides a difference of near-equal quantities, and the paper's remedy is a Taylor form switched in below some roughness. The conjugate-multiplied root `2C/(b + sqrt(b*b+4ac))` is algebraically identical, cancels nothing since `b > 0` throughout, and needs **no threshold constant** (Press et al., *Numerical Recipes* 5.6). The whole domain is then one branch-free expression: `r=0` gives `a=0, b=1` and hence `rho=C`, the Lambertian identity out of the algebra rather than special-cased, and `C=1` gives `rho=1` exactly, leaving the white furnace untouched
- feat: `BsdfParams::diffuseRho`, resolved once per hit in `resolveBsdfParams` rather than per lobe evaluation -- `evaluateBsdfSplit` runs repeatedly per vertex under NEE and `sampleBsdf`. Kept separate rather than overwriting `baseColor`, which the transmission lobe and the rasterizer's albedo AOV both still consume as the authored colour
- test: `checkEonAlbedoInversion`, the only instrument that can see any of this. Two assertions per row: an analytic round-trip against `referenceEonAlbedo`, an independent transcription of Listing 1's forward `E_EON` that re-derives `c1`/`c2` from literals; and a Monte Carlo integral of the **shipped** lobe, which additionally catches a `diffuseRho` computed and never consumed. `ior=1` is what makes the second exact -- exact dielectric Fresnel is identically zero there while Schlick's `(1-c)^5` tail is not, so `coatFresnelRatio` and `dielectricFresnelAvg` both collapse and `diffuseKdAt` becomes exactly 1. Tolerances are derived, not tuned: the estimator's own standard error accumulated from its second moment, plus the paper's stated <0.1% bound on the quartic albedo fit the renderer evaluates
- note: **the pre-existing suite is completely blind to this**, measured rather than asserted -- every case in every validator runs at `baseColor 1` or `diffuseRoughness 0`, where the inversion is the identity, so reverting the fix leaves all five validators **byte-identical**. Verified by breaking: reverting the consumer fails 8 measured rows and **0 analytic** (the algebra is still right, only the plumbing broken); dropping the `4ac` term fails 12 and 12, an exact no-op on the three `r=0` rows; substituting the appendix's *other* inversion (eq. 33, which constrains the average albedo instead) fails 8 and 8 by **3.07%** at `C=0.5, r=1`, so the check pins *which* convention shipped rather than merely that some inversion happened
- note: cornell is **bit-identical** (0/691200). Its glTF declares no materials and no images, so `baseColorTexture` is the white fallback and every wall resolves to `baseColor [1,1,1]`, while `sphere01` is `metallic=1` and `sphere02` `transmissionFactor=1`, both of which zero `diffuseKd`. Demonstrated instead on a throwaway coloured `clay.json` at `diffuseRoughness 1`: 335622/691200 channels, **every one brighter**, and over the changed pixels R +3.87% / G +10.57% / B +10.89% -- the darkest channel lifted most, which is the per-channel saturation the inversion undoes
- note: still deferred -- dispersion, the `baseColor`/`transmissionColor` transmission-tint overlap, EON shape-gating on `diffuseKd`, an occlusion-sensitive curved transmissive test, and a CIE-integrated fit for the chromium values

## Camera film-back preset drop-down

- feat: `Camera::FilmBackPreset` (`{name, FilmBack}`), and a new asset `assets/config/camera.json` -- a JSON array of real, sourced camera/format film-back dimensions (ARRI Alexa 65, Hasselblad X2D 100C, RED V-RAPTOR, Canon 5D Mark IV, Leica M11, IMAX 15/70, VistaVision, Super 35/16/8, plus a 4x5 view camera and two reference-only entries, Human Eye and GoPro), replacing the single hardcoded `{23.76, 13.37}` in `profile.json`
- feat: `loadFilmBackPresets` (`profile_config.cpp`), same fail-closed contract as `loadProfileConfig` -- validates every entry's `widthMm`/`heightMm` > 0. `profile.json`'s `camera.filmBack` numeric pair is replaced by `camera.filmBackPreset`, a name resolved against the catalogue at startup (`main.cpp`, `tools/render_beauty.cpp`); an unresolvable name aborts startup with a clear stderr message, same as an unrecognised `defaultLUT`
- feat: `DebugCameraController::filmBack_` is no longer `const` -- `filmBack()`/`setFilmBack()` added, mirroring the existing aperture/shutter/iso live-edit pattern. HUD's Camera section replaces the read-only film-back text with an `ImGui::Combo` (same shape as the AOV dropdown) plus a read-only `width x height mm (aspect:1)` readout
- fix: `PathTraceInputState` (`main.cpp`) tracked `focalLengthMm` but not film back, so switching presets updated the camera's stored value with no visible effect -- neither the path tracer nor the rasterizer ever re-triggered. Added `filmBackHeightMm` (the only component that feeds `verticalFovRadians()`; `widthMm` stays display-only, so tracking it here would trigger retraces for a change with no visible effect on the image)
- docs: roadmap's now-shipped "Camera film-back preset drop-down" item removed (§5)

## Chrome preset returned to an idealised mirror

- chore: `chrome.json` goes back to a near-white mirror -- `diffuseColour` `[0.95, 0.95, 0.97]`, the value it held before the F10 reshading, with `edgeTint` now authored explicitly as white rather than left to the loader default. White edge tint is the one setting at which the conductor reproduces Schlick's grazing behaviour, i.e. **no reflectance dip**, so the preset exercises transport rather than the parameterisation
- docs: the measured chromium the preset used to carry -- reflectivity `[0.552, 0.555, 0.558]`, edgeTint `[0.555, 0.558, 0.672]` from Johnson & Christy 1974 via the paper's eq 14/15 -- is recorded at `conductorIorFromReflectivity` with its provenance, its 0.0048 round-trip error against the source n,k, and the wavelength-triple sensitivity. JSON takes no comments, so the shader is where an authoring reference can live; pasting the two triples back into `chrome.json` restores the look
- docs: README's material-library row for `chrome.json` said "Polished **chromium**, not an idealised mirror", which this inverts; it now describes the mirror and points at the shader comment for the measured alternative

## Shading pipeline: F_avg on each interface's own Fresnel

Continues the review above. F10 made the conductor's single-scatter Fresnel exact complex-IOR Gulbrandsen, but the Kulla-Conty multiple-scattering lobe attenuates each repeated bounce by a cosine mean `F_avg`, and both interfaces still took that from Karis' closed-form mean of Schlick -- the exact mean of a function neither one evaluates any more. The deferred `note:` in the previous section is this section.

- fix: `conductorFresnelAvg` replaces `schlickFresnelAvg` on the metal path. A 3-node quadrature rule `sum(w_i*F(mu_i))` over the same `fresnelConductorChannel` the single scatter calls, nodes and weights fitted by equality-constrained least squares against 128-point Gauss-Legendre over the whole clamped domain (`r` in `[1e-4, 0.9999]` x `g` in `[0, 1]`). Max absolute error **4.0e-4**, RMS 1.2e-4 in float32, against Karis' 0.086 -- **216x**. The weights sum to 1, so a perfect mirror stays exact to 4e-8 and the value cannot leave `[0, 1]` by more than that. Karis' error changed **sign** with `edgeTint` because it is a function of `f0` alone: -0.086 at `edgeTint=1` near `r=0.25`, +0.030 at the shipped chromium
- fix: `coatAlbedo` hand-inlined the same Karis expression while its own single scatter evaluates exact `fresnelDielectric`. It now takes `dielectricFresnelAvg(ior)`, 2.7x closer at `ior 1.5` (-0.0023 against -0.0061) and, unlike Karis, collapsing to exactly 0 at `ior 1` where an index-matched interface reflects nothing. This half, not the conductor half, is what moves the cornell picture: **1882 of 1886** changed channels
- test: `checkAverageFresnel`, the instrument the suite never had. Composite Simpson quadrature of this file's *own* independent complex-arithmetic reference -- not the shipped Fresnel, so a transcription error surfaces instead of cancelling -- asserted against both shipped averages, plus the `ior=1` collapse. **No energy test can see an error here**: `checkWhiteFurnaceTwoSided` runs at `f0=1` where every candidate mean agrees to 1e-4, and `checkFurnace`'s coloured-metal rows are upper-bound-only and so blind to a loss. Verified by breaking: perturbing one fit weight while holding `sum(w)=1` gives **75 failures here and zero in any furnace**
- refactor: `checkConductorFresnel` rested on the multiple-scattering term M cancelling out of a difference across `edgeTint`, because "M's Favg comes from `f0` alone". That premise is now false, and its comment said so in as many words. The exact ratio identity is restricted to roughness 0.05, where `E ~ 1` leaves M under the tolerance -- **90 -> 45 ratio points**, and `F` does not depend on roughness, so the curve is still pinned everywhere. At roughness 0.6 it instead asserts the normal-incidence value rises **monotonically** with `edgeTint`, the property a Schlick `F_avg` structurally cannot have. Reverting the conductor call site fails through that assertion's all-rows-skipped backstop, again with no furnace failure
- refactor: `dielectricFresnelAvg` and `conductorFresnelAvg` given external linkage (the anonymous namespace closes and reopens around them) and declared in `bsdf.h`, so the validator can measure them. The `constexpr` node/weight arrays keep internal linkage, and within-TU inlining is unaffected
- note: a rational fit over `(r, g)` was measured first and **rejected**, not skipped. At `r=0.99` the inverted `n` collapses from 39.8 to 0.005 across the last tenth of `g` -- a boundary layer no low-order form in that chart holds, and 49 terms reached only 9e-3. Sampling the function's own values sidesteps the chart, since `(n, k)` is what `F_avg` actually depends on
- note: cornell moves 1886/691200 channels, **every one darker**, max 13/255; all five validators exit 0. The conductor half accounts for only 4 of those channels -- chrome's roughness 0.05 leaves a `(1-E)` deficit of ~3e-4, so it is correctly near-invisible on this scene, and the prediction that it would dominate was wrong
- note: reverting the coat half alone leaves the suite **green** -- a real gap, stated rather than papered over. Its effect is at most 1.9e-4 on any furnace reading, below every tolerance in the file
- note: still deferred -- EON's Appendix A albedo inversion, dispersion, the `baseColor`/`transmissionColor` transmission-tint overlap, EON shape-gating on `diffuseKd`, an occlusion-sensitive curved transmissive test, and a CIE-integrated fit for the chromium values

## Ground-truth Beauty pixel probe, Cornell default scene

- fix: the HUD pixel probe read Beauty back from framebuffer 0 (`glReadPixels`, 8-bit, `[0,1]`-clamped, already OCIO-transformed) instead of the raw `HdrImage` texel every other AOV samples directly -- the same clamp the histogram's over-range stats had, fixed the same way. Now routes through that same raw-texel path (`selectPathTracedImage`'s existing `Beauty` case), then reapplies the viewer's exposure/LUT/invert pipeline to that texel on the CPU via a cached `OCIO::CPUProcessor::applyRGB` call, so the probe matches on-screen colour without the clamp or 8-bit quantize. Chromatic aberration and dither are intentionally not reproduced -- display artifacts, not scene data
- refactor: `kOcioRec709Display` extracted as a header constant (`ocio_display_transform.h`) alongside the existing `kOcioSrgbDisplay` -- was a bare string literal only inside the GPU shader build; now the one source of truth the new CPU processor also reads
- chore: `Options::scenePath` defaults to `scenes/cornell.json` instead of `scenes/tree.json` -- cornell.json is fully wired (materials, sphere overrides, HDRI environment); `tree.json`'s stump model has no ground plane yet (§5 roadmap) and stays reachable via `--scene`
- docs: README's Run section and material-library table updated for the new default scene; the two now-shipped roadmap items (ground-truth pixel probe, Cornell default) removed, and the Screen-capture-to-PNG item's stale line references corrected
- fix: the pixel-probe panel had no visible border -- `style.WindowBorderSize = 0.0F` disables ImGui's own border on every window (`hud_overlay.cpp:437`), and a rect added to the window's own draw list at its exact bounds gets clipped by that window's clip rect to near-invisibility. `drawPixelProbePanel` now captures the window's pos/size before `End()` and draws the histogram panel's existing border style (`IM_COL32(60,60,60,180)`, `hud_overlay.cpp:259`) on the foreground draw list instead, matching `drawChannelViewCorner`'s existing pattern for HUD elements drawn outside window clipping
- fix: `WindowBg` alpha lowered from 0.85 to 0.80 (`hud_overlay.cpp:438`) -- at 0.85 the shared HUD panel background read as opaque black regardless of backdrop, most visible on the small pixel-probe panel where there's no high-contrast content to fake translucency through the fill
- docs: roadmap's now-shipped "Probe panel border" item removed

## Shading pipeline: test coverage and conductor Fresnel (F7-F10)

Continues the review above. Phase 2 closed the coverage gaps that let curvature-driven bugs hide; Phase 3 replaced the last approximated BSDF term. Every finding measured, and every new assertion verified by breaking the code under test -- a green run proves nothing when the check prints only on failure.

- test: `checkBeerLambert` -- absorption was the newest feature in the pipeline and entirely unexercised, both existing slabs being white and non-absorbing. Asserts the documented contract instead of restating its formula: `transmissionDepth` is the distance at which transmittance reaches `transmissionColor`, so probing at exactly that distance expects `transmissionColor` with no `exp()` in the test, and a second row halves the depth and expects the square, which is what separates a true exponential from anything linear in distance. `ior 1.0` makes it exact -- an index-matched interface neither bends nor reflects, so absorption is the only thing left that can move the reading
- test: `makeSphereScene`, the suite's first curved geometry. Every transmissive test used flat quads, whose coplanar vertex normals make `transmissionOffsetEpsilon`'s curvature factor exactly zero and `shadowTerminatorOffset` a no-op -- which is why two real curvature-driven bugs (F2, F3-root) were invisible to all four suites while plainly visible in a cornell render. `checkTransmissiveSphere` applies the flat slab's own invariant to a sphere: a white non-absorbing dielectric under uniform L0=1 is invisible whatever its shape. Reads 0.988-0.996, inside the flat slab's own 0.985-1.000, so the residual is the known albedo-table error and nothing curvature-specific
- fix: inverted pole guards in `makeSphereScene` dropped the valid cap triangle of each pole row and emitted the degenerate one -- 128 zero-area triangles and an annular hole at each pole, measured as surface area 12.4808 against 4*pi = 12.5664, now 0 degenerate and 12.5412. Found by pre-commit review while the five-validator suite was green throughout
- test: `checkPdfNormalization` and `checkReciprocity` now sweep `diffuseRoughness` {0, 0.5, 1.0}. At 0 -- the only value previously tested, and the one `default.json` ships -- `eonUniformMixWeight` is `pow(0, 0.1)`, exactly 0, so `pdfEon`'s uniform/CLTC mixture collapsed to `cltcPdf` alone and CLTC to plain cosine: neither the LTC fit's normalisation nor EON's reciprocity was ever reached. Verified against breaks chosen to be exact no-ops at r=0 -- a 5x uniform mixture term gives 44 pdf failures, an asymmetric `sOverT` gives 96 reciprocity failures, and neither produces a single failure at `diffuseRoughness = 0`
- test: per-instance material binding, both sides. `checkPerInstanceMaterials` renders two coplanar quads with identical `Material`s distinguished only by per-instance `diffuseColour`; `checkMaterialBinding` compares `resolvePerInstanceSettings` field by field against `loadMaterialConfig`'s own reading of the shipped `glass.json`. A render cannot supply the latter, since a render only exercises keys that already match -- which is why F6's bit-identical cornell could not gate it
- fix: that binding test first checked only 5 of its 11 field copies, because `makeSettings`' defaults happen to agree with `glass.json` on six. Base is now sentinels differing in all 11, asserted first: deleting the `ior` copy reads 10 of 11 against the sentinels and a clean 11 of 11 against the old base
- feat: exact complex-IOR conductor Fresnel (Gulbrandsen 2014, ported from the paper's Appendix A), replacing Schlick. Metals gain an authored `edgeTint` alongside reflectivity `r` (the existing `f0`), inverted to a complex IOR through eq 12 and eq 2. Schlick is `f0+(1-f0)(1-c)^5`, monotone in `cos` by construction, so it forced every metal to exactly white at grazing and could not express a reflectance dip at any `f0`. `edgeTint` defaults to white -- the no-dip edge Schlick produced -- so existing material files load unchanged
- fix: two departures from the paper's listing, both measured. `k^2` is evaluated as `(nMax-n)(n-nLow)`, the same expression through eq 2's own roots, because the literal form subtracts two large near-equal numbers and in float32 at `r=0.9999, g=0` returns `-1.28e6` where the true value is exactly 0 -- that ill-conditioning is what forces the listing's own `clamp(r, 0, 0.99)`, and 0.99 would cost 1% of normal-incidence reflectance at `f0=1`, enough to move the white furnace onto its tolerance edge. And the reflectance is the exact unpolarized form rather than the listing's `rs`/`rp`, which is the large-|eta| approximation and deviates by up to 0.094 absolute at mid reflectivity, where real metals sit
- fix: computing `a^2` as `(a2b2+t0)/2` cancels catastrophically when `t0 < 0`. At `f0=1`, `n` is ~5e-5 and it must resolve 2.5e-9 out of two numbers near 1.0, which collapses `a` to zero in float32 and pins `F` at exactly 1.0 -- Schlick's own answer there, so the white furnace's conductor rows came back byte-identical and the fault read as an inert change rather than a bug. Found only because the prediction that those rows would move was stated before measuring. That branch now divides by a sum, and the `max()` that was masking it is gone so a future fault fails loud
- fix: the Fresnel G-buffer AOV applied Schlick to the metallic-blended `f0` across the whole surface, reading 0.192 at `metallic=0.2` where shading evaluates `0.8*0.04 + 0.2*0.192 = 0.0704`, a 2.7x over-report -- wrong before this work and wrong in a second way after it, since Schlick can never show the dip an authored `edgeTint` produces. `fresnelAtViewAngle` now evaluates the same term shading does; `fresnelSchlick` had no other consumer and is deleted
- feat: `chrome.json` shaded as measured chromium rather than an idealised white mirror. At `metallic=1` `diffuseColour` *is* `f0`, i.e. the reflectivity, so authoring a measured edge tint onto an invented `r=0.95` would have been incoherent -- both now come from Johnson & Christy 1974 n,k through the paper's own inverse mapping (eq 14/15). Round-tripping the authored values back through the shipped forward inversion recovers the source n,k to 0.0048, inside its three-significant-figure precision. 0.58x darker at normal incidence, a -6.4/-6.6/-3.9 percent dip at cos ~0.30, and chromium's characteristic cool cast
- note: `lobes.fresnelAvg` still uses Karis' Schlick cosine mean while the single-scatter Fresnel is now exact, and its **sign depends on `edgeTint`** -- Karis is a function of `f0` alone while the true mean falls as the edge tint darkens, so it under-predicts by up to 0.086 at `edgeTint=1` and over-predicts by 0.030 at the new chromium. Scales with the `(1-E)` deficit: 3e-4 of incident radiance at chrome's roughness, up to ~0.05 for a mid-reflectivity metal at roughness 1, and no test in the suite can see it. Recorded in the README roadmap with `coatAlbedo`'s identical pre-existing inconsistency
- note: still deferred -- EON's Appendix A albedo inversion, dispersion, the `baseColor`/`transmissionColor` transmission-tint overlap, EON shape-gating on `diffuseKd`, an occlusion-sensitive curved transmissive test (the sphere closes the medium-desynchronisation blind spot but a furnace measures how much energy arrives, not where from, so it structurally cannot see a light leak), and a CIE-integrated fit for the chromium values in place of three representative wavelengths

## Shading pipeline: correctness review (F1-F6)

Line-by-line review of the shader-library work above against the literature, one finding per commit, each measured rather than argued.

- fix: `sampleEon` returned NaN at `diffuseRoughness = 0` -- `eonUniformMixWeight` is exactly 0 there (the value `default.json` ships) and the mixture branch admitted `u.x == 0.0`, computing `0/0`. Every downstream guard is NaN-permeable (`wi.z <= 0`, `pdf <= 1e-8` are both false for NaN), so the NaN reached the accumulator and poisoned that pixel for the rest of the progressive render. `u.x <= pUniform` -> `u.x < pUniform`
- fix: every `bsdf_validate` assertion tested for the *failure* condition, which no NaN satisfies, so a NaN-returning BSDF passed the entire suite -- two white-furnace rows had been printing `nan` and still exiting 0. All now assert the pass condition and negate; `withinBand` replaces the three identical vec3 band checks
- fix: far-side NEE shadow rays used the flat `kRayEpsilon` for both origin offset and `tMin` while the continuation ray already used `transmissionOffsetEpsilon`, so the curvature mismatch that epsilon exists for was unhandled on exactly the rough-glass path the far-side guard was added to light. Filed as false occlusion losing energy; measured *darker* -- a light leak closing, the ray had been escaping through the surface that should have occluded it. Unreachable for shipped glass (smooth -> delta lobe, far-side pdf 0), so demonstrated on a rough-glass variant
- fix: a ray inside an absorbing medium that missed all geometry added full environment radiance with no absorption at all. Now attenuated per channel by `exp(-sigmaA * inf)` -- 0 where absorbing, 1 where not, written per channel because `0 * inf` is NaN. Terminating the path instead (the filed fix) would have broken `integrator_validate`'s open-sided non-absorbing slab, whose rays legitimately miss and must still contribute
- fix: `shadowTerminatorOffset` only ever moves along `+vertexNormal` and was applied unconditionally, so on an inward-going ray (refraction entering, TIR inside) it pushed the origin back through the interface just crossed. No transmission event was recorded, desynchronising the single-level medium stack and carrying "inside glass" state through open space -- 0.93% of glass entries, TIR reflection the dominant source. The projection now mirrors on the far side rather than being skipped there
- refactor: `LobeProbabilities::coatAlbedoAvg` was written and never read (the only consumer is the same-named local in `computeLobeProbabilities`); deleted
- fix: `diffuseRoughness` 0.3 -> 0.0 on `chrome.json`/`glass.json`. Inert for glass (`transmissionFactor=1` -> `diffuseProb` exactly 0) but *not* for chrome: `metallic=1` zeroes the diffuse value while `diffuseProb` stays non-zero, so the diffuse strategy still carried the opaque multi-scatter specular lobe and `sampleEon` was sampling CLTC against a cosine-shaped target. Unbiased either way (1024-pass references at both settings agree to max 2/255); RMS error over the 1976 affected channels down 1.2%
- fix: a `materialOverrides` key matching no glTF node was silently ignored -- override *paths* were validated but node *names* were not, so a one-character typo in `cornell.json` rendered 41% of channels differently and exited 0. `resolvePerInstanceSettings` now checks keys against the instance names before loading any material file and returns `nullopt`, the all-or-nothing startup gate an unloadable override path already had
- feat: `tools/render_beauty` -- headless N-pass beauty render to PNG (minimal zlib-backed writer, no new dependency), the before/after instrument the review is measured with; `--compare` reports a signed mean delta alongside RMS, the directional read the NEE-epsilon and `diffuseRoughness` findings both turned on. `resolvePerInstanceSettings` extracted from `main.cpp` into `scene/material_binding.{h,cpp}` so the tool and the real renderer share one path and comparison images cannot drift from shipped behaviour
- note: the review's remaining findings are deferred, not fixed -- conductor Fresnel is Schlick-only (no complex IOR, so metals go white at grazing), EON's Appendix A albedo inversion is unimplemented, there is no dispersion, `baseColor` and `transmissionColor` both tint transmission, and `diffuseRoughness` still shapes the borrowed multi-scatter sampling lobe whenever `diffuseKd` is 0

## Shader library: EON diffuse, per-object materials, chrome and glass

- feat: EON (Portsmouth, Kutz, Hill 2025, JCGT 14(1)) replaces plain Lambertian as the diffuse lobe -- an energy-preserving rough-diffuse BRDF with its own CLTC importance sampling, now the diffuse model in the OpenPBR spec. Adds `diffuseRoughness` (0 = exact Lambertian limit); `computeLobeProbabilities`/`evaluateSpecularLobe` skip the dielectric escape-table lookup when `transmissionFactor` is zero, removing cost previously paid unconditionally on every opaque shading sample
- feat: materials move out of an inline `scene.json` block into standalone `assets/materials/*.json` files, referenced by path (mirroring `ModelConfig::gltfPath`'s convention), so the shader library can grow without a scene-schema change
- feat: glTF `COLOR_0` (vertex colour) support in `gltf_loader.cpp`, multiplying `baseColor` alongside the base-colour texture and `diffuseColour`
- refactor: `tools/gltf_prep.cpp` (a from-scratch glTF rebuilder) replaced by `tools/gltf_tangent.cpp`, which patches a source glTF in place to synthesize only a missing `TANGENT` accessor -- any other attribute, primitive or material passes through untouched
- feat: `assets/materials/chrome.json` -- polished chrome (`metallicFactor=1.0`, `transmissionFactor=0.0`), validated via a whole-scene `materialPath` swap against the existing GGX + Schlick-Fresnel-for-conductors + Kulla-Conty machinery; no new BSDF code needed
- feat: per-object material assignment -- `SceneConfig::materialOverrides` (glTF node name -> `materials/*.json` path) builds a per-instance `PathTraceSettings` vector, threaded alongside the existing scene-global one through both render paths (`tracePath`/`renderPathTraced`, `shadeRow`/`renderRasterGBuffer`, `PathTraceDriver`) and indexed by `ShadingTriangle::instanceIndex`. `gltf_loader` now captures each `MeshInstance`'s owning node name to key the override lookup
- fix: `multiScatterShape` (the rough-dielectric multiple-scattering compensation lobe) evaluated the transmitted-wi escape probability at the wo-side eta orientation for both its callers, correct only for the reflected-wi caller. `LobeProbabilities` gains `escapeAvgRecip` (the reciprocal-eta average escape); `multiScatterShape` now takes its eta and normalization explicitly, and the transmitted-side caller (`transmitMultiScatter`) passes the reciprocal pair while the reflected-side caller (`evaluateSpecularLobe`) keeps the original orientation. Total energy is unaffected (the hemisphere integral still collapses to `1 - escapeWo` by construction on each side); only the transmitted lobe's angular shape changes. Strict per-direction reciprocity of this term remains a known, accepted limitation of the unidirectional integrator (`tools/bsdf_validate.cpp`'s `checkTransmissionReciprocity` stays scoped to single scatter)
- feat: Beer-Lambert volumetric absorption -- `transmissionColor`/`transmissionDepth` (Arnold `standard_surface` convention, `sigmaA = -log(transmissionColor)/transmissionDepth`), added as optional-with-default fields to `MaterialConfig`/`PathTraceSettings` (default `[1,1,1]`/inert, so existing materials needed no changes). `tracePath` gains a single-level medium stack (`mediumSigmaA`), toggled on `LobeType::Transmission` samples and applied once per bounce against `hit->t`
- fix: refraction on coarsely-tessellated curved geometry (a 500-triangle sphere) was self-intersecting a neighbouring facet almost immediately -- measured `hit->t` ~1e-3, an order of magnitude below the sphere's true ~0.1-0.3 chord length -- because the smooth shading-normal-derived refraction direction diverges from each flat facet's own geometric normal, and the existing `kRayEpsilon` (tuned for floating-point precision) is far smaller than the resulting curvature mismatch. The continuation-ray offset for `Transmission`-type bounces now scales with measured vertex-normal curvature (`sin(angle)` between each pair of a triangle's vertex normals, via cross-product magnitude) times the triangle's longest edge -- exactly zero on any planar patch regardless of its absolute size, growing only where tessellation coarseness actually requires it. An earlier version scaled by raw edge length alone and regressed `integrator_validate`'s flat-slab transmission test (pushed the ray ~283 units past a 2000-unit quad); caught by review, not shipped
- feat: `assets/materials/glass.json` -- smooth/clear dielectric (`metallicFactor=0`, `transmissionFactor=1`, `ior=1.5`, its own `roughnessMin` below diffuse/chrome's shared floor so a genuinely smooth target isn't clamped away)
- feat: Cornell box geometry re-tessellated (three separate node-named meshes, 500 triangles per sphere, replacing a single merged untriangulated mesh) and promoted to `cornell_v001`; `assets/scenes/cornell.json` assigns chrome and glass to the two spheres via `materialOverrides`, leaving the box on the default diffuse material

## Histogram full-range fit

- fix: `hud_overlay.cpp`'s histogram curve (`smoothChannel`) 9-tap-averaged each bin against its raw neighbors before the curve was scaled to the panel, so any non-flat-topped distribution's true peak silently landed below 1.0 -- narrow/spiky AOVs (Beauty, Albedo, Normal, ...) read visibly crushed while wide (Luminance) or near-binary (Sobel/Gabor/IOR) ones didn't. Added `rescaleToUnitPeak`, applied after smoothing in both `drawGrayscaleHistogram` and `drawRgbHistogram`, so every AOV's curve now genuinely fills the panel to its own true peak, preserving `sharedPeak`'s relative-channel-height intent
- docs: §4 Quick roadmap item trimmed to just the still-open half -- `updateOverRangeStats`'s peak-multiple readout is hardcoded to Beauty regardless of the selected AOV; §4 Moderate gains contact-sheet (grid of every AOV) and optic-flow AOV items

## Quick-tier roadmap: JSON config migration, debug keybinds, BounceCount colormap, chromatic aberration

- feat: replaced the hand-rolled `json_scan.{h,cpp}` substring scanner with vendored nlohmann/json (single header, `third_party/nlohmann/json.hpp`, same vendoring convention as `third_party/cgltf` -- not a submodule)
- feat: `profile_config.cpp`/`scene_config.cpp`/`material_config.cpp` rewritten on `.at()` + `json::exception`, so a malformed field now fails with the specific offending key instead of silently returning `nullopt`. A small ADL shim (`json_glm.h`) lets `.get<glm::vec3>()` work directly for the JSON array fields
- feat: `hdriPath` moved from `profile.json`/`ProfileConfig` to `scene.json`/`SceneConfig` -- the environment map is scene-specific, not a session default
- refactor: `profile.json`'s fields, and `ProfileConfig`'s matching struct/load-function order, regrouped into window/camera-pose/lens/controls/render/path-tracer, from a flat unordered list
- feat: `Y`/`I`/`H`/`ESC` debug keybinds -- Luminance-AOV toggle with restore-previous (falling back to Beauty if Luminance was reached some other way, e.g. the HUD dropdown, since the restore sentinel isn't a valid AovId), display-colour invert (wired through the OCIO shaders and both standalone HSV/edge-filter shaders), HUD visibility toggle (gates only `HudOverlay::draw`, keeping ImGui's beginFrame/render pairing intact), and window-close via a new `Window::setShouldClose()`
- fix: dropped the `K` crosshair-toggle keybind; the crosshair overlay itself stays, permanently on
- feat: BounceCount AOV colormapped via Turbo (Mikhailov/Google 2019, 11-stop table, linear interpolation) instead of raw grayscale -- mapped CPU-side in `ensurePathTraceDisplayTexture`, which already only rebuilds once per accumulated pass, so the per-texel map costs nothing extra per frame; replaces the old `exposureEv`-based normalization
- feat: chromatic aberration over Beauty -- radial per-channel UV offset (R toward centre, B away, G unchanged) added to the shared OCIO display-shader builders, gated to Beauty only in `presentFrame`; HUD slider in the Camera section, no keybind
- docs: README's DirectDiffuse/DirectSpecular AOV description corrected from "delighted" to "physical" (base colour included), matching `bdecb41`'s deliberate change; the "Fix DirectDiffuse/DirectSpecular regression" roadmap item moved from Quick to Moderate with the finding attached -- not a regression, and recovering the delighted view needs Albedo (rasterizer-only) and DirectDiffuse (path-tracer-only) simultaneously fresh, which `aovNeedsLightTransport`'s rasterizer/path-tracer mutual exclusion prevents
- docs: mouse-disappears-on-orbit roadmap item dropped; stale `main.cpp:390-427` keybind-block line reference corrected to `:441-464`
- docs: three new §4 Quick roadmap items -- histogram full-range fit + min/max readout (replacing a single peak scalar), gating the live-histogram downsample to Beauty/path-traced AOVs only (it currently runs unconditionally, including on rasterizer-backed AOVs), and a full comment-style audit against `notes/architect.md`'s efficient-comments rule

## Documentation: README accuracy pass

- docs: README §3's Metallic/Roughness/Fresnel/IOR AOV rows corrected. Metallic and IOR are global scalars from `material.json` (`settings.metallicFactor`/`settings.ior`), constant across the whole image despite their "Material"/"Transport" table categories implying per-material data; Roughness is a per-hit texture scaled by a global factor, not a fully per-material quantity either. Fresnel is a bare Schlick term on `f0`, simpler than shading's `mix(dielectric Fresnel, Schlick(f0), metallic)` (`bsdf.cpp`), so it can read a materially different value from what's actually rendered
- docs: README §4's global-illumination roadmap item no longer implies caustics are emergent once area lights exist. Unidirectional path tracing cannot sample specular-diffuse-specular paths at any light count; the claim now points at the bidirectional path tracing item, the transport algorithm caustics actually need
- docs: README §2's "Camera framing overlays" row corrected to describe only the centre crosshair `hud_overlay.cpp`'s `drawFramingOverlays` implements -- no letterbox mask or rule-of-thirds grid exist in the code
- docs: four new §4 roadmap items -- expanded terminal output (launch + loop), the orbit cursor-visibility bug, a JSON-driven camera film-back preset drop-down, and photometric calibration against real light-meter units
- docs: three new §4 Moderate roadmap items -- packet tracing, ray reordering before shading, and deferred/sorted shading by material -- plus clarifying detail folded into four existing items (object instancing, adaptive sample budget, MIP-mapping, GPU backend) instead of duplicating them

## Colour and display: EXR precision, exposure-aware AOVs, output dither

- fix: `loadExr` reads via `Imf::InputFile` with an explicit float `FrameBuffer`, replacing `Imf::RgbaInputFile`'s half decode. Half saturates at 65504 and silently turns a legitimate above-ceiling source value into `inf`, which then propagated through `EnvironmentMap`'s importance-sampling CDFs (`rowSum`/`marginalCdf_`) and corrupted NEE sampling for the whole run with no error anywhere. Every returned image is now scanned and rejected outright if any texel is non-finite
- feat: `loadExr` also checks the EXR's `chromaticities` header attribute against Rec.709 (compared via a default-constructed `Imf::Chromaticities`, itself Rec.709 by OpenEXR's own convention) and warns, non-fatally, on a mismatch -- a linear-but-wrong-gamut asset (ACEScg/P3 HDRIs are the usual culprit) previously rendered with systematically wrong saturation/hue and nothing caught it
- feat: `hsv_display.frag`/`edge_filter.frag` gain a `uExposure` uniform, applied at the single point each shader samples `uHdrColor`, so HSV/Luminance/Sobel/Gabor respond to the exposure slider like Beauty does. Previously these four AOVs displayed at unity gain regardless of exposure while Beauty alone responded
- refactor: `Camera::ev100()` gains a static overload taking `(aperture, shutterSeconds, iso)` directly, and is now the single definition of the formula -- `DebugCameraController::relativeExposureEv()` calls it twice (current vs. profile.json default) instead of inlining the `log2` arithmetic a third time. `Camera::exposure()` deleted entirely: unused outside a debug log line, and its `/1.2` belonged to an absolute photometric model the engine never adopted (the scene isn't calibrated to real-world radiance)
- feat: a HUD readout next to the histogram reports the fraction of texels clipping at the display encode and the peak value as a multiple of display range (e.g. "3.2%, peak 47.8x"), computed from the pre-display-transform float `HdrImage` at the same capture cadence as the existing GPU histogram. The histogram alone bins the post-display-transform, post-8-bit-clamp framebuffer and cannot distinguish "just over 1.0" from "100x over" -- both saturate its bin 255 identically
- feat: triangular-PDF output dither added to all three display fragment shaders (both OCIO LUTs and Raw), just before the final 8-bit quantization -- a screen-space hash subtracted against itself at an offset, static per pixel since this targets a converged (Monte-Carlo-noise-free) image rather than motion
- docs: README §2's tone-mapping and photographic-exposure rows corrected to match what the code does -- a colorimetric-only display encode with no tonal compression (values above 1.0 clip by design), and exposure as a relative-stops delta against the profile default rather than an absolute photometric quantity

## Interactive performance: rasterizer and display path

- test: `tools/raster_bench.cpp`: a timing harness for `renderRasterGBuffer`, the synchronous per-frame render-thread work behind the 15 primary-hit AOVs. Committed rather than thrown away, because every change in this section is a performance claim and the previous workstream's figures are unreproducible without rebuilding its benchmark from prose. Reports best-of-N (run-to-run spread here is +/-10%, wide enough to hide a single change); deliberately outside `add_test`, since a benchmark's pass/fail is a human reading a number
- test: the harness scene is synthetic and sweeps the two axes the rasterizer's cost is actually a function of, neither adjustable in a fixed asset: `--triangles` moves the sub-triangle array across the L2 boundary, and `--layers` sets depth complexity, which is what a depth prepass trades against. Per-layer triangle radius is derived from that layer's frustum cross-section, so raising `--triangles` shrinks triangles instead of piling up overdraw and the two axes stay independent
- note: baseline at the shipped 2048x1152 framebuffer, best-of-five: **135 ms per rasterization** at the scene's 20,561 triangles, and **136 ms at a single triangle**. The cost at the shipped triangle count is entirely fixed per-frame overhead, not scan conversion; triangle count first registers between 20K and 200K (401 ms), which is where the 84-byte-per-sub-triangle array crosses this machine's 12 MB L2. Any estimate of the row scan's cost that assumed the array streams from DRAM does not describe the shipped scene
- feat: `renderScale` and `interactiveRenderScale` in `profile.json`: the path tracer and rasterizer render at a fraction of the framebuffer and the display blit upscales, which it already did for free (`glViewport` targets the framebuffer, the display texture samples `GL_LINEAR`). A 1024x576 window is a 2048x1152 framebuffer on a Retina display, so the renderer was tracing 2.36M paths per pass for a window implying 590K
- feat: the renderer drops to `interactiveRenderScale` on any input change and promotes back to `renderScale` a quarter-second after the last one, the standard progressive-renderer trade of resolution for latency while the camera moves. The promotion needs no separate code path: it changes the trigger, and a changed trigger is already what dispatches
- fix: the trigger state is split into inputs and scale, because the scale is derived from whether the inputs changed. As one struct the promotion to full resolution would read as fresh interaction on the following frame, re-arming the timer it had just satisfied and pinning the renderer at the interactive scale permanently
- note: measured 151 -> 38.6 ms per rasterization at 2048x1152 vs 1024x576, best-of-five: 3.9x against an exact 4x pixel ratio. The path tracer takes the same 4x on paths per pass. Setting both scales equal reproduces the previous behaviour exactly, so the mechanism is inert if unwanted
- feat: the rasterizer runs only when one of its own 15 AOVs is selected. It previously ran unconditionally on every trigger change: a full-screen shade of 15 images on the render thread, on nearly every frame of camera interaction, producing nothing anybody was looking at whenever Beauty was displayed, which is the default. The two AOV sets partition the enum exactly (12 light-transport, 15 rasterizer-backed, `AovId::Count` 27), so the complement of the existing predicate is an exact gate
- feat: the orbit pivot comes from a single Embree ray down the view centre rather than the centre texel of the G-buffer, which is what had forced that rasterization to be unconditional: two million pixels shaded to read one. Also the more accurate instrument: no fill rule, no z-precision, no dependence on the resolution the G-buffer happened to be rendered at
- feat: `aov` joins the trigger state, so switching between two rasterizer-backed AOVs (Normal to Albedo changes neither the camera nor the light-transport predicate) still re-rasterizes
- perf: the G-buffer is allocated once and rendered into in place, each row cleared by the worker about to overwrite it instead of 15 sequential full-image memsets beforehand: 566 MB of allocate-and-zero per call at 2048x1152. Measured 151 -> 125 ms, best-of-five: real, but again an order of magnitude below what the byte count suggests, consistent with the previous section's finding
- fix: `RasterGBuffer` carries a `generation` counter, because reusing the buffer in place leaves its address constant and the display texture cached on exactly that pointer identity: it would have uploaded the first render and then never updated again. Verified separately that a reused, deliberately dirtied buffer produces bit-identical output to a fresh one across all 15 AOVs, which is the failure mode the per-row clear introduces
- perf: `renderRow` visits a per-row bucket of the sub-triangles whose bounding box covers that row, in flat CSR form, instead of scanning the whole sub-triangle array once per scanline. The cost removed is memory bandwidth rather than comparisons: `RasterSubTriangle` is 84 bytes and the scan is linear, so every row streamed the entire array: invisible while that array fits in cache, dominant once it does not
- perf: `buildSubTriangles` clips and projects in parallel over chunks of the triangle list rather than on one thread, each chunk appending to its own vector and the chunks concatenated in order afterwards. That order is what makes the result identical to the sequential build, since the z-test is first-writer-wins at exactly equal depth: verified by hashing all 15 AOVs across three camera poses on a scene seeded with coplanar triangles at equal depth, with and without the change
- note: measured at 2048x1152, best-of-five, by triangle count: 20,561 (the shipped scene): 123.5 -> 114.8 ms; 200K: 423 -> 142; 1M: 2158 -> 284; 5M (the `rkswd_tier_0` tier that also ships): **11,446 -> 604 ms**, 18.9x. The review's "~90 GB/frame" premise was analytical and does not describe the shipped scene at all: at 20,561 triangles the array is 1.7 MB and stays in L2, which is why that case gains 7% rather than an order of magnitude. It describes tier_0 exactly, where a single rasterization took 11.4 seconds
- note: CSR memory is O(sum of row spans): peak RSS at 5M triangles rises 2461 -> 2894 MB. A scene of few very large triangles would need more of it than the sub-triangle array itself, the opposite regime from the one it exists for
- perf: visibility is resolved for the whole row before anything is shaded, so a pixel is shaded exactly once instead of once per depth-record improvement. The single-pass form shaded on every improvement and threw the result away on the next one, paying 8 bilinear fetches, a shading frame and 15 texel writes per discarded surface
- perf: the shading pass walks the row in x order rather than in triangle order, so the 15 AOV writes advance linearly through each image instead of scattering across it: worth ~9% on its own at zero overdraw, where the prepass itself saves nothing
- fix: `raster_bench` emits its depth shells furthest-first. Nearest-first, every deeper shell was rejected on arrival by the z-test, so a pixel accumulated one depth record however high `--layers` went and the flag measured the best case for the code it exists to stress: it reported a 2% regression where there is a 6.5x gain
- note: measured at 2048x1152, 20,561 triangles, minimum of four best-of-five runs, by overdraw: 1 layer: 113 -> 104 ms; 2: 281 -> 127; 4: 593 -> 141; 8: 1164 -> 178, 6.5x. Single-pass cost is linear in overdraw, prepass cost is not; the residual growth is coverage and depth testing, which no prepass removes. This is the worst case by construction, an unsorted scene averaging the harmonic number of records (~2.7 at 8 shells) and a front-to-back one none
- note: the winner index is 4 bytes per pixel alongside the z-buffer, 9.4 MB at 2048x1152. Both passes derive barycentrics from one shared `coverPixel`, so identical source expressions guarantee identical floating-point contraction and the shading pass recomputes bit-for-bit what the depth pass chose on: verified by the same 15-AOV hash across three poses, unchanged at `69c421dfeedf667e`
- perf: channel isolation is a shader uniform rather than a value baked into the uploaded texels. Pressing R/G/B previously re-copied the whole image on the CPU and recreated the GL texture to display data the shader was about to read anyway; it is now a `glUniform1i` against the texture already resident
- feat: `uChannelView` added to the three OCIO display programs (generated source, alongside `uExposure`) and to `edge_filter.frag`. In the edge filter it applies inside `sampleLuminance`, where isolating to grey and then Rec.709-dotting returns that channel unchanged because the weights sum to 1 -- the previous behaviour reproduced exactly, not approximated. `hsv_display.frag` already had its own, applied deliberately to HSV output rather than to source RGB
- fix: `channelViewToBake` drops out of the display-texture cache key, since the texture no longer depends on it. This is also what frees `flipRowsForDisplay` of everything but the flip
- perf: the display texture is uploaded in place rather than destroyed and recreated every frame. `Texture::upload` reallocates storage only when the render resolution actually changes and is otherwise a `glTexSubImage2D`, which matters more now that a resize is routine (the interactive/settled scale switch) rather than window-only
- perf: no mip chain on the display texture. The only consumer is a 1:1 fullscreen blit sampling LOD 0 exclusively, so the chain was regenerated on every upload and never read; min filter drops to `GL_LINEAR`
- perf: the row-order flip moved into `fullscreen_triangle.vert` as `vUv = vec2(p.x, 1.0 - p.y)`, deleting `flipRowsForDisplay` and its full-image scratch copy -- the HdrImage now uploads directly. All five programs sharing that vertex shader sample the same display texture, so the convention change is total; `edge_filter.frag` is invariant under it (Sobel takes `length(gx,gy)`, and reflection permutes the 0/45/90/135 Gabor bank whose max over `abs` is unchanged)
- note: the HUD's GPU memory readout drops the phantom `+1/3` it added for a mip chain that no longer exists, so the display texture now reports what is actually allocated
- perf: `Camera::primaryRay` gains an overload taking a prebuilt `ViewBasis`, and `renderPathTraced` hoists that basis out of its per-sample loop. The aspect-taking overload rebuilds it -- two sin, two cos, an atan, a tan, two normalize, two cross -- on every primary ray, for a value constant across the whole pass. Measured 332 -> 310 ms per pass at 1024x576 (best-of-5, min of three runs); `rasterizer.cpp` already hoisted it this way
- perf: `PathTraceDriver::setSuspended` parks the driver whenever the selected AOV is one it does not produce. Previously, selecting a rasterizer-backed AOV left it accumulating passes of an image no longer on screen, on every core, competing with the rasterizer the render thread runs synchronously
- refactor: `RowThreadPool` renamed to `ThreadPool` (and its file with it). It dispatches an index, not a row -- the path tracer has dispatched tiles through it since the buffer-pipeline work -- so the name named only its first caller
- note: the two thread pools were left separate. With the AOV gate and suspension above, the rasterizer and the driver are never both dispatching, so merging them under a shared dispatch mutex would serialize two things that no longer overlap and add a lock for no measured gain; parked workers cost stack memory and no CPU
- note: `RTC_SCENE_FLAG_ROBUST` was evaluated and rejected. Embree's watertight traversal measured ~24% slower per pass (348 -> 433 ms, single-variable A/B), not the few percent expected, which is too much for an artifact class the renderer already mitigates through ray epsilons, shadow-terminator origins and geometric-normal leak rejection
- note: moving the pixel probe off `glReadPixels` was evaluated and rejected. Reading the CPU image would report scene-referred float radiance, but the probe's purpose is the value actually on screen, so the display-encoded framebuffer read is the correct source and stays

## Transmissive energy conservation: multiple-scattering sampling and back-side NEE

- feat: the transmit-side multiple-scattering lobe gains its own cosine sampling strategy over the far hemisphere, carved out of the transmit selection mass in proportion to the energy each carries. That is what lets the compensation move outside `evaluateTransmissionLobe`'s half-vector rejections and be delivered over the whole hemisphere rather than the refraction-reachable cone alone: energy balance goes from 0.68 to 1.0 at roughness 1.0 exiting. Both strategies gate on the same deficit threshold, so no selection mass ever reaches a lobe of zero value and smooth glass is untouched
- fix: the rough transmission single-scatter value omitted `transmissionFactor`, so a `transmission=0.5` dielectric refracted at full strength on top of a diffuse substrate already scaled by `(1-transmissionFactor)`: 1.42 of the energy it received. Both that factor and `(1-metallic)` now come from `transmitWeight`, which also drops `transmissionFactor` on the exiting side to match `transmitProb` and `transmitPhysicalValue`: inside the medium there is no substrate to withhold anything, so a ray must reflect internally or exit
- fix: a rough transmission sample is MIS-eligible and no longer bypasses the power heuristic. `lastSampleWasTransmission` becomes `lastSampleWasDelta`, derived from `pdfBsdf` returning exactly 0 rather than from the lobe type, which removes the special case entirely
- fix: NEE now samples both sides of a transmissive vertex. The `geoCos > 0 && shadingCos > 0` guard restricted light samples to `wo`'s side, so rough glass lit from behind got BSDF sampling alone; and, paired with the delta MIS bypass, double-counted the overlap where NEE did reach the far side of an exiting vertex. Bounce-0 far-side NEE writes the Refraction bucket
- test: `checkTransmissiveEnergyBalance`: `sampleBsdf` with transmitted draws converted back to the energy domain, where 1.0 is correct at every roughness, side and `transmissionFactor`. `checkFurnace` could see neither defect: it is upper-bound-only, and the entering side's `eta^2` compresses 1.42 to 0.95, under its 1.0 bound. Measured 1.0 within 0.7% across 64 cells
- test: `checkTransmissiveSlab`: a white non-absorbing slab in a uniform environment must be invisible. Two quads with opposing normals are the only configuration in the suite that reaches a transmissive *exiting* vertex; a single quad reads as entering at every hit. Measures 1.22 at roughness 1.0 on the parent commit against 0.99 after
- test: `checkTransmissionReciprocity`: the eta^2-corrected invariant `f_t(wo->wi)*eta_wi^2 == f_t(wi->wo)*eta_wo^2`, which is what holds across a refracting interface in place of Helmholtz symmetry. Catches a misplaced `etaR^2`, a flipped `denom` orientation or an un-flipped `ht`: O(1) errors the furnace and round-trip tests cannot see, since they assert totals in which the two sides' errors cancel. Inverting the correction fails all 360 pairs by exactly `ior^4`. Worst honest discrepancy 6.4e-3 at roughness 0.05 against a 1e-2 bound: `dD/D ~ 4/alpha^2` amplifies a few-ULP `ht` difference at these roughnesses, so `checkReciprocity`'s 1e-4 is not transferable. `wi` is constructed by refracting `wo` through the macro normal rather than sampled, since the lobe is a fraction of a degree wide and an arbitrary far-side direction would pass the check vacuously; the non-zero-pair count is asserted for that reason
- note: scoped to single scatter. `multiScatterShape` evaluates `escapeWi` at the wo-side `eta` for every `wi`, transmitted ones in the other medium included, which is precisely what makes its hemisphere integral come out at `(1 - escapeWo)`; but under the swap both `etaSq` and that `eta` flip. Isolated with no new accessor by testing only below `kMinDeficit`, where the lobe switches itself off. Recorded against README §4's volumetric-and-subsurface item: it blocks bidirectional transport through rough glass, not the unidirectional integrator
- docs: README corrected where this stack invalidated it: rough transmission and four-lobe selection in §1/§2, the multi-scattering-compensation roadmap item retired, Walter 2007 and Heitz et al. 2016 re-annotated as implemented/superseded, and Kulla & Conty 2017 added as the compensation actually used
- docs: new §4 roadmap item for reference scenes and material validation: a Cornell box (Goral et al. 1984) the renderer is measured against rather than merely rendered in, carrying the per-material showcase. `tools/` validates the BSDF and integrator analytically and nothing validates a full scene, so bucket misattribution, light leaks at shared edges and terminator artifacts are visible by eye only. Blocked on area lights (item 0): the environment map is the only light source, so an emissive panel cannot be authored
- docs: new §4 roadmap item for lens and shutter sampling: depth of field and motion blur, both Cook et al. 1984 primary-ray dimensions rather than post-process filters. `Camera` already carries `aperture()`, `focalLengthMm()` and `shutterSeconds()`, but they feed EV100 exposure only, so f/1.4 brightens without shallowing depth of field and the shutter never reaches the sampler. Fixes a dangling §5 citation: Cook 1984 was annotated "named in §4's motion-blur roadmap item" and no such item existed

## Rough transmission: Walter 2007 lobe and transmissive energy compensation

- feat: rough specular transmission (Walter et al. 2007, PBRT-v3's radiance-transport form) replacing a delta lobe at every roughness: `alpha` was computed in the transmission branch and never referenced, so `roughness` silently did nothing on any transmissive material while the reflection lobe on the *same* interface was rough GGX. Frosted glass is now authorable
- feat: below `alpha < 1e-3` transmission stays the original delta lobe, matching PBRT's `EffectivelySmooth`. The existing `kMinAlpha` floor sits inside that region, so smooth glass keeps the exact, noise-free Snell path rather than becoming a stochastic estimate of the same thing: verified by the transmission round-trip test producing byte-identical output. At roughness 0.05 the new rough path reproduces the delta path to 0.2%
- feat: the albedo table gains reflected/transmitted escape channels indexed by (roughness, mu, eta), built with **exact** dielectric Fresnel. A transmissive interface loses energy on a different curve from a reflective one (0.559 vs 0.307 at roughness 1.0) because the below-horizon reflections that drive `E` down are the *valid* side for refraction, so the existing table could not describe it. One axis in `log(eta)` covers entering and exiting, since the two are reciprocals; the VNDF samples are shared with the existing build loop, so the third axis costs only the refraction
- feat: multiple-scattering compensation for the transmissive interface, deficit `1 - (R + T)` redistributed reciprocally across both hemispheres and blended against the opaque compensation by `transmissionFactor*(1-metallic)`, so an opaque material keeps exactly the measured behaviour it already had
- fix: `coatFresnelRatio` hardcoded the entering orientation, so total internal reflection was invisible to the escape budget on the exiting side. Schlick's basis cannot express TIR at all (exact Fresnel is 1.0 inside the cone where Schlick reads ~0.1), which is why the new channels use exact Fresnel rather than a rescaled Schlick split
- fix: the deficit ratio floored only its denominator, breaking the numerator/denominator cancellation as roughness fell and turning a vanishing lobe into a huge one (a round trip reading 1.54 where it should read 1.02); now guarded rather than clamped
- note: energy balance is within ~1% to roughness 0.4 but falls to 0.68 at roughness 1.0 on the TIR-heavy exiting side. The multiple-scattering term sits after `evaluateTransmissionLobe`'s geometric rejections, and those directions have no transmission sampling density: moving it earlier would place energy that can never be sampled, trading a shortfall for bias. Closing it requires giving the multiple-scattering lobe its own cosine sampling strategy on the transmissive side, which is the next change. `checkFurnace` remains upper-bound-only for transmissive materials, so this is not yet gated
- note: the integrator still treats every transmission sample as a delta for MIS purposes (`lastSampleWasTransmission`), and NEE does not yet sample the back side of a transmissive surface. Both follow with the sampling-strategy change

## Energy conservation: multiple scattering and BSDF reciprocity

- feat: Kulla-Conty multiple-scattering compensation (Kulla & Conty 2017): single-scatter GGX discards the energy `smithG2` masks away and never returns it, so a white conductor reflected 0.31 of the light it received at roughness 1.0. A compensation lobe `Fms*(1-E(mu_o))*(1-E(mu_i))/(pi*(1-Eavg))` returns exactly that deficit, driven by a 32x32 directional-albedo table built once at startup (a few ms) by deterministic stratified quadrature of the VNDF weight identity `G2/G1`
- feat: reciprocal diffuse/specular coupling: the diffuse substrate now receives `1 - coatAlbedo` on the way in *and* out, replacing the one-sided `1 - F(mu_o)`. Restores Helmholtz reciprocity (`f(wo->wi) == f(wi->wo)`), required by every bidirectional transport algorithm on the roadmap, and fixes a second energy loss: at roughness 1.0 / `ndotV` 0.4 the macro Fresnel is 0.129 while the coat actually reflects 0.030, and that 10% went to neither lobe
- feat: the specular lobe-selection probability is scaled by `E(mu_o, roughness)`, so VNDF sampling takes the single-scatter share and cosine sampling takes the (cosine-shaped) multiple-scatter share. Without it a rough white metal, whose Fresnel pins `specularProb` to the 0.95 clamp, would sample 69% of its own reflectance only 5% of the time. No new sampling lobe, so the one-sample mixture estimator is unchanged and still unbiased
- feat: the albedo table stores the Schlick-basis split `Ess(f0) = f0*a + b` so one table serves any `f0`, with the single-scatter term rescaled by exact-dielectric/Schlick Fresnel at each direction: without that rescale a smooth white dielectric *creates* ~1.4% energy at grazing, since Schlick under-predicts exact Fresnel there and the substrate is handed the difference
- feat: white furnace floors are now a correctness target, not a regression baseline: 1.0 +/- 0.02 on both bounds, across 28 cells. Half sit deliberately off the table's grid, where the measured value is `E_true + (1 - E_interpolated)` and so tests interpolation error directly, which is why the table needs no public accessor
- feat: `checkReciprocity`: an equality to float precision, not a statistical bound, since every term is symmetric by construction after the coupling change. Fails at 96 of 144 sampled geometries on the parent commit, worst case 0.306 vs 0.179
- note: measured white-furnace energy is now within +/-0.9% of unity at every roughness, angle and material tested, from a prior range of 0.31-1.00. The residual is Monte Carlo noise plus table interpolation
- note: `BsdfSample::rawThroughputWeight` reports a cosine-drawn multiple-scattering sample as diffuse, so the delighted Direct/Indirect Specular AOVs under-count it. Beauty is exact either way (`throughputWeight` is the physical value), and the delighted split is removed entirely when the transport buckets become a true partition
- note: transmission keeps its one-sided `(1-F)*t` energy fraction and its delta lobe; rough transmission and its own multiple-scattering compensation are a separate change

## Validation suite: integrator coverage and two-sided energy bounds

- feat: `tools/integrator_validate.cpp`: first test to exercise `renderPathTraced`/`tracePath` at all. One unoccluded quad under a uniform L0=1 environment has no indirect light, so `maxBounces=0` and `maxBounces=1` must agree and both must equal the analytic single-scatter integral; Russian roulette must not change the answer. Catches the MIS truncation fixed in the previous commit, which read 0.17 against a correct 0.97 at `maxBounces=0`: the "direct lighting only" mode was producing roughly one sixth of the correct radiance
- feat: two-sided white furnace test: a white non-absorbing surface under uniform radiance must return exactly 1.0, so a LOWER bound is what detects energy loss; the previous suite asserted only `Lo <= bound` everywhere and was structurally blind to it. Floors are a measured regression baseline, not a correctness target: single-scatter GGX retains 1.00/0.92/0.31 at roughness 0.05/0.5/1.0 (white conductor, normal incidence), quantifying the multiple-scattering compensation that is still outstanding
- feat: transmission round-trip test: asserts the non-symmetric `eta^2` radiance-compression factors cancel over enter-then-exit, the invariant `bsdf.cpp`'s own comment claims and nothing tested. Pairs the exiting leg at the Snell-refracted angle, not the incident one, which would put it past the critical angle
- feat: env-map pdf consistency test: `pdf(importanceSampleDirection(u).direction)` must equal that sample's own pdf, on a structured map under non-zero rotation. Untested before, and a mismatch silently corrupts every MIS weight in the renderer
- fix: the pdf-normalization check's `ndotV<0` rows integrated the +z hemisphere while `pdfBsdf` mirrors `wi` into `wo`'s hemisphere, so they measured exactly 0 and passed `<= 1.0` vacuously: the exiting side was never tested. Now integrates the hemisphere the density actually occupies
- fix: corrected a comment claiming `Material`/`MeshInstance` need a live GL context; both are plain data, and that claim was why the suite had no integrator-level test
- chore: `enable_testing()` + `add_test`: all five tools now run under `ctest --test-dir build`; previously they were buildable but nothing invoked them
- note: the pdf check stays upper-bound-only by design. VNDF reflection sampling is not normalized over the hemisphere (below-horizon samples are discarded), so the true integral is the horizon-clipped mass, which has no closed form: the white furnace test measures it instead

## Light-transport correctness: MIS truncation, NEE ordering, sampling

- fix: the terminal BSDF-sampled ray is now traced for its environment miss: the loop previously sampled a direction at the final bounce and built a ray it never intersected, while NEE at that vertex stayed MIS-weighted down against a counterpart that never fired, losing `bsdfPdf^2/(bsdfPdf^2+lightPdf^2)` of its direct lighting (approaching 100% where `bsdfPdf >> lightPdf`, and applying to every second surface vertex at `maxBounces=1`); the loop now runs one iteration past `maxBounces`, guarded so it can only collect a miss, never shade
- fix: a failed `sampleBsdf` no longer skips that vertex's NEE: the two are independent estimators of independent directions, and `nullopt` (below-horizon VNDF reflection, underflowed mixture pdf, transmission lobe with no mass) says nothing about the BSDF's value toward the light; also stops the Shadow AOV reporting "occluded" where sampling merely failed
- fix: bounce-0 NEE writes both lobe buckets deterministically instead of routing by whichever lobe the continuation ray drew: that routing wrote one bucket per sample, thinning DirectDiffuse/DirectSpecular in expectation by each lobe's selection probability (a Fresnel- and view-angle-dependent factor clamped to [0.05,0.95]), so an evenly lit flat wall read as a Fresnel gradient
- fix: NEE samples environment radiance nearest-texel (`EnvironmentMap::sampleDirectionNearest`, sharing `equirectTexelOf` with `pdf()`), matching the piecewise-constant cell its pdf is derived from; bilinear there bled a bright texel's energy into neighbours whose density is correctly low, spiking `f/pdf` into fireflies at exactly the small bright features the luminance CDF exists to find. The miss path keeps bilinear, where the env pdf enters only a bounded MIS weight
- fix: Russian roulette survival ceiling 0.95 → 1.0: a path carrying full throughput was killed 5% of the time and the survivors reweighted, unbiased but strictly variance-increasing for no compensating saving
- note: the four validation tools pass, but none can detect the energy loss the first two fixes remove: every furnace and pdf assertion is one-sided. A two-sided furnace test, a transmission round-trip test and an env-map pdf-consistency test follow separately

## Synchronous CPU rasterizer for primary-hit AOVs

- feat: `rasterizer.h/.cpp`: row-parallel edge-function rasterization (Pineda 1988) with a near-plane Sutherland-Hodgman clip, computing the 15 primary-hit-only G-buffer AOVs synchronously every frame instead of via the async path tracer's convergence loop; `gbuffer_shading.h/.cpp` factors the material/shading sampling shared with the path tracer out of `path_tracer.cpp`
- feat: Wireframe and BoundingBox merged into one combined AOV: real screen-space line rasterization z-tested against scene depth (white mesh edges, yellow bounding-box edges drawn on top), replacing the old per-pixel analytic distance tests
- feat: path-traced accumulation only restarts when the selected AOV actually needs light-transport data, avoiding wasted CPU on a raster-only AOV during camera movement
- chore: `tools/rasterizer_validate.cpp` cross-checks the rasterizer against a fresh Embree oracle, including box-edge occlusion and wireframe coverage
- chore: debug camera profile reframed (`filmBack` matched to window aspect, camera pulled back for a wider view), `maxSamples` settled at 4
- docs: README updated (pipeline, component reference, AOV table, roadmap item 0 retired now implemented)

## AOV correctness fixes

- fix: specular lobe's throughput isolated from diffuse admixture in DirectSpecular/IndirectSpecular AOV routing
- fix: Depth AOV auto-ranges display exposure to the actual max depth visible in-frame, replacing a `farClip`-based normalization that read real scenes as black; Depth/BounceCount no longer clamp to white; HUD pixel-probe units corrected
- fix: Shadow AOV re-averages across progressive passes (`PathTraceGBuffer` → `PathTraceDynamic`) instead of freezing as binary speckle from pass 1; confirmed HDR exposure correctly has no effect on it (occlusion is exposure-invariant)

## HUD polish, new debug AOVs & config split

- feat: HDRI Exposure slider, per-AOV pixel probe (native units, on-screen composited value for Beauty/post-filter AOVs), configurable progressive-accumulation max-samples cap
- feat: Shadow, Wireframe, Bounding Box AOVs: binary NEE occlusion, barycentric edge distance, ray-vs-AABB slab test
- feat: `material.json`/`profile.json` split: renderer/material defaults moved out of `scene.json`; configurable texture/HDRI asset paths; Bump texture wired up (previously loaded, unused)
- fix: `maxBounces=0` now renders direct lighting only; pixel probe reads the true on-screen pixel; specular lobe isolated from diffuse in NEE's DirectSpecular/IndirectSpecular routing
- chore: comments consolidated to single-line, technical-only style; default ISO 400, `maxBounces` 0→1, max-samples cap enabled by default

## Rasterizer removal: path tracer as sole renderer

Follow-on to the path tracer becoming the continuously-converging primary renderer (NEE/MIS against the environment map, full G-buffer/transport-component AOV set); the GPU rasterizer it ran alongside as a fallback was by then fully redundant and is removed.

- feat: `presentFrame` shows the path-traced result exclusively, with a black clear until the first pass publishes; no rasterizer fallback branch
- feat: orbit-pick pivot resolved from the path tracer's own G-buffer (world-space hit position + hit mask at its centre pixel), replacing a GL depth-buffer readback + view/projection unprojection
- refactor: `Camera::viewMatrix`/`projectionMatrix`, `Window`'s resize callback, and `HdrFramebuffer`'s GL depth/colour attachments removed: nothing traces a raster depth or colour buffer anymore
- refactor: `Material`'s textures are CPU-only (`HdrImage`); GPU texture upload, `MeshInstance`'s GPU `Mesh`, and the bump-map detail-normal texture (rasterizer-only consumer) are gone
- refactor: punctual lights (`Light`, `kMaxLights`, `SceneConfig::lights`) removed: the rasterizer was their only consumer; the path tracer has always lit purely from the environment map
- refactor: `SceneStats` drops draw-call/culled counts (no per-instance culling without a raster draw loop); HUD drops the "Draw calls"/"Mtri/s" readouts and the GPU timer's geometry-pass split
- chore: deleted `pbr.vert/frag`, `sky.frag`, `equirect_to_cubemap.frag`, `prefilter_specular.frag`; `cubemap_texture`, `env_prefilter_pass`, `mesh`, `hdr_framebuffer`, `sh_irradiance`, `frustum` (header+source); `tools/furnace_test.cpp` and its CMake target (validated `pbr.frag`'s analytic BRDF/SH-IBL, both deleted; `bsdf_validate` already covers the path tracer's own BSDF)
- docs: README rewritten to describe the shipped CPU path tracer + thin OpenGL display/HUD layer directly, replacing the phase-by-phase dual-renderer build history

**Phase 5: Materials & recursive transport complete.**

## Phase 0: Foundation

### A: Scaffolding
- chore: CMake project scaffold (C++20, `engine` + `gen_test_pattern` targets)
- chore: Pitchfork dir layout (`include/engine/{platform,gfx,scene}`, `src/`, `assets/`, `tools/`)
- chore: stub sources for all planned modules (window, gl_debug, shader_program, mesh, texture, hdr_framebuffer, post_process_pass, ocio_display_transform, camera)
- deps: `glm`, `opencolorio` via Homebrew

### B: Window & GL context
- feat: `Window` (GLFW, GL 4.1 core fwd-compat, RAII, move-only, resize-callback hook)
- feat: `gl_debug`: `checkError`, `GL_CALL` macro, `khrDebugAvailable`
- feat: main loop: glfwInit/GLEW init → poll/clear/swap → glfwTerminate

### C: Camera
- feat: `Camera`: position/yaw/pitch, film back + focal length → derived vertical FOV, `viewMatrix`/`projectionMatrix`
- note: aperture/shutter/ISO/exposure deferred to Stage F (first real consumer is the OCIO exposure uniform)

### D: HDR FBO + polygon
- feat: `Mesh` (RAII VAO/VBO/EBO, `createQuad`), `ShaderProgram` (`loadFromFiles`/`loadFromSource` → `optional`), `Texture` (`GL_RGBA16F` upload, placeholder checkerboard)
- feat: `HdrFramebuffer` (RGBA16F color + depth renderbuffer, completeness check, `resize`/`bind`)
- feat: `PostProcessPass`: attribute-less-VAO fullscreen-triangle blit
- feat: `quad.vert`/`.frag`, `fullscreen_triangle.vert`, `passthrough.frag` (placeholder, superseded by Stage F's OCIO shader)
- feat: render loop now draws checkerboard quad into HDR FBO, blits to screen; `Window`'s resize callback wired to `HdrFramebuffer::resize`
- note: unencoded checkpoint (expected washed out until Stage F)

### E: Test EXR
- feat: `gen_test_pattern`: writes 700x100 calibration EXR (black/18%grey/white/R/G/B/ramp)
- feat: `Texture::createFromExr`: `RgbaInputFile` load, half→float, exception boundary → `optional`
- feat: quad now displays `test_pattern.exr`
- note: `*.exr` now gitignored: generated assets are local-only, not committed
- fix: `Texture::createFromExr` now flips row order on load: EXR row 0 is the image top, but `glTexImage2D` row 0 is texture `v=0` (bottom); previously every EXR-loaded texture rendered upside down, invisible on the row-uniform `test_pattern.exr`, confirmed on a real HDRI

### F: OCIO viewer LUT + exposure
- feat: `OcioDisplayTransform`: sRGB/Rec.1886-Rec.709 viewer LUTs via OCIO's real Display/View API (`Un-tone-mapped` view, no filmic tone-mapping), built once at startup, no LUT textures
- feat: `Camera` gains `aperture`/`shutterSeconds`/`iso`, `ev100()`, `exposure()` (Filament/Frostbite EV100 model)
- feat: `Window::setKeyCallback`; debug `L` key cycles sRGB → Rec709 → Raw (unencoded passthrough)
- fix: corrected an earlier design assumption: OCIO 2.5.2 has no raw `CURVE - LINEAR_to_sRGB` builtin; empirically verified the correct construction against the installed library
- note: exposure uniform left neutral (EV=0) rather than seeded from `Camera::exposure()`: the calibration pattern isn't scene-referred radiance

### G: Verify
- feat: `logColorCheck`: `glReadPixels` numeric check (black/grey/white/ramp) vs hand-computed bytes, logs at startup and on every LUT toggle
- chore: removed `Texture::createPlaceholderCheckerboard` and `assets/shaders/passthrough.frag` (zero remaining callers/loads)
- verified: sRGB grey=118, Rec709 grey=125, Raw grey=46 (all match hand-computed expectations); ramp monotonic; resize correct, no retina 2x bug

**Phase 0: Foundation complete.**

## Phase 1: Debug HUD & system feedback

### A: Vendor ImGui
- deps: Dear ImGui v1.92.9 (`third_party/imgui` git submodule, not `docking`); first non-Homebrew dependency; vendored as source since Homebrew has no `find_package`-consumable ImGui+backends
- chore: CMake wiring: ImGui core + GLFW/OpenGL3 backend sources added to `engine`, `IMGUI_IMPL_OPENGL_LOADER_GLEW` (reuses GLEW's resolved GL pointers instead of ImGui's own loader), configure-time guard for an uninitialized submodule

### B: ImGui bring-up
- feat: `Window::nativeHandle()` accessor; ImGui's GLFW backend chains to `Window`'s already-installed key callback with zero changes to `window.cpp`
- feat: `engine::debug::HudOverlay`: owns the ImGui context + GLFW/OpenGL3 backend lifetime, composites after the OCIO tonemap pass (never into the linear HDR FBO)
- fix: stale "Phase 1's WASD/QE/R debug camera" comment in `window.h` → Phase 3, matching the README renumbering

### C: GPU/system readout
- feat: `engine::debug::system_info::queryGpuInfo()`: `GL_VENDOR`/`GL_RENDERER`/`GL_VERSION` + monitor refresh rate, queried once at startup
- verified: Apple M1 / 4.1 Metal - 90.5

### D: Frame-timing HUD
- feat: `engine::debug::FrameStats`: 120-entry ring buffer, fps/avg/min/max
- feat: `engine::debug::GpuTimer`: double-buffered `GL_TIME_ELAPSED` query, non-blocking read; `gpuTimerQueryAvailable()` runtime check follows `khrDebugAvailable()`'s don't-assume precedent
- feat: `Mesh::triangleCount()`; geometry + post-process passes each wrapped in a `GpuTimer` in `main.cpp`
- verified: ~118-120 FPS (caps at monitor refresh), sparkline live, geom/post ms and Mtri/Mpix per s all plausible

### E: Memory HUD
- feat: `engine::debug::memory_tracker`: `residentSetBytes()` (`mach_task_basic_info`), one atomic GPU-byte counter (`trackGpuAlloc`/`trackGpuFree`/`gpuAllocatedBytes()`)
- feat: byte-tracking hooks in `Texture`, `Mesh`, `HdrFramebuffer`; move ctor/assign now exchange the tracked byte count alongside the GL handle, else a move silently double-frees the *accounting* (not the GL object)
- verified: GPU alloc MB hand-computed vs HUD readout matches exactly (2048x1152 HDR FBO + EXR texture ≈ 27.5 MB); resizing tracks proportionally with no upward drift

### F: Styling pass
- feat: borderless/pinned-top-left HUD panel, cyan section headers, dark near-opaque background, no `imgui.ini` written

### G: Verify
- verified: visual match against reference layout; `L`-key LUT toggle regression-checked (sRGB→Rec709→Raw still correct post-ImGui-integration)

**Phase 1: Debug HUD & system feedback complete.**

## Phase 2: Geometry, textures & basic material

### A: glTF loading
- deps: vendor cgltf (single-header C99 parser, jkuhlmann/cgltf v1.15, MIT); two plain files, not a submodule; one-shot asset load, not a hot path
- feat: `Mesh`'s Vertex layout gains normal/tangent (vec4, `.w` = bitangent handedness), attribute locations 2/3
- feat: `Texture::createFromFloatPixels`: mipmaps, `GL_REPEAT`, trilinear filtering; `createFromExr`'s row flip dropped (EXR's row-0-top already matches glTF's v=0-top)
- feat: `engine::scene::loadGltf()`: position/normal/uv/tangent via cgltf's own accessor helpers, recursive node-transform accumulation, all 6 material textures resolved via `Texture::createFromExr`; roughness/bump/specular (no standard glTF slot) read back from the material's `extras` JSON via a small targeted string scan
- fix: `extrasTextureIndex`'s integer parse hardened (`from_chars`, not `atoi`: `atoi` can't distinguish "parsed 0" from "failed to parse")
- verified: the real 2.55M-vertex/5M-triangle/6-EXR test asset loads in ~1.1s (Release build)

### B: First real draw
- feat: the loaded stump model replaces the placeholder quad; new `pbr.vert`/`pbr.frag` (unlit base-color only for now), per-instance `uModel` + a CPU-computed `uNormalMatrix`
- chore: removes `Mesh::createQuad()`, `quad.vert`/`.frag`, the test-pattern texture load, `logColorCheck()`: all zero remaining callers
- fix: depth testing scoped to just the scene draw (enabled/disabled per frame): the post-process present pass draws at a fixed NDC z, so leaving depth test on for it would depend on undefined leftover depth-buffer contents
- fix: vsync re-enabled: an uncapped CPU submitting draw calls for this 5M-triangle scene faster than the GPU could drain them reproduced a genuine GPU driver hang on the test machine
- verified: 50+s continuous runtime, ~35-37 FPS, ~24-28ms real GPU geometry time/frame, no hangs; numeric pixel-sampling of screenshots confirms a coherent tree-stump silhouette renders in the expected viewport region

### C: Culling & material completeness
- feat: `Mesh` computes a model-space AABB once at construction; `frustumIntersectsAabb` (Gribb/Hartmann plane extraction) added as infrastructure
- feat: frustum culling wired into the render loop, reusing the per-instance world-space AABB already computed for the World Position AOV
- fix: previously-dead `Material::baseColorFactor` now actually multiplied into shading
- perf: per-instance world-space AABBs precomputed once after load instead of every frame

### D: AOV selector & shading
- feat: AOV debug dropdown (Beauty, Albedo, Normal, GeomNormal, Roughness, UV, WorldPos, Tangent, Metallic, ObjectID, AO) + independent R/G/B channel isolation; non-Beauty AOVs force OCIO's Raw passthrough, restoring the user's LUT choice on Beauty
- feat: active LUT shown in the debug HUD
- feat: tangent-space normal mapping (bump-derived detail normal blended in) + basic Lambertian/Fresnel shading (F0 = mix(specular, baseColor, metallic)) against one fixed test light
- feat: authored AO texture multiplied into Beauty shading: a practical simplification (no ambient term exists yet to occlude), not physically exact
- chore: default asset switched to tier1 LOD (36.5k triangles) for faster shader iteration

### E: Verify
- chore: final Phase 2 cleanup pass: removed a dead `<cstdlib>` include and comments that had drifted out of sync with the code they described
- docs: README corrected to match Phase 2's actual implementation (AOV selector moved from Phase 3, normal-mapping description corrected, AOV table's AO/Metallic/Roughness rows corrected)
- build: default to Release when `CMAKE_BUILD_TYPE` is unset: Debug measured ~40x slower on this project's CPU-bound glTF load (47s+ vs ~1.1s)

**Phase 2: Geometry, textures & basic material complete.**

## Phase 3: Scene controls

### A: Data-driven scene setup
- feat: minimal JSON scanner (`json_scan.cpp`) + `scene.json`/`profile.json` loaders: scene/camera setup now data-driven instead of hardcoded in `main.cpp`

### B: Debug camera & histogram
- feat: `DebugCameraController`: WASD/QE fly, R reset, LMB-drag orbit around a depth-sampled pivot (`HdrFramebuffer::sampleDepth`)
- feat: `engine::debug::Histogram`: double-buffered PBO readback of the current AOV, live RGB channel display
- feat: scene stats readout (draw calls, triangle counts, viewport resolution); camera & lens readout, debug camera controls, and viewport/scene stats wired into the HUD
- chore: fly speed tuned from 3.0 to 1.0 m/s for precise navigation at this scene's scale

### C: Remaining utility AOVs
- feat: Alpha, Depth (planar camera-space Z, Arnold/RenderMan/OpenEXR convention), Luminance, and HSV AOVs added
- feat: Sobel/Gabor edge-detection AOVs: new two-pass `edge_filter.frag` (fixed 3×3 Sobel kernel / 4-orientation Gabor bank) consuming the Luminance AOV, reusing the existing generic `PostProcessPass`; Gabor kernel weights precomputed once on the CPU
- refactor: Beauty's shading math factored into `shadeBeauty()`, shared across Beauty/Luminance/HSV/Sobel/Gabor without duplicating it in branches that don't need it
- chore: all 17 AOV indices renumbered to match the README's §3 AOV reference table order

### D: System readout
- feat: GPU refresh rate (`glfwGetVideoMode`) and system RAM (`sysctl`/`host_statistics64`, resampled on the existing 250ms throttle) added to the debug HUD

**Phase 3: Scene controls complete.**

## Phase 4: Direct lighting & acceleration

- feat: BVH: binned-SAH, CPU-built, one-time (`bvh.h/.cpp`); Phase 5 infrastructure only, not yet wired into rendering, validated by `tools/bvh_validate.cpp` against a brute-force reference
- feat: punctual lighting: point/directional light list (replacing the single fixed test light), analytic Cook-Torrance GGX (D, Smith height-correlated visibility, Schlick Fresnel)
- feat: IBL: SH-9 diffuse irradiance (Ramamoorthi & Hanrahan 2001, no texture) + GGX-prefiltered specular cubemap (Karis 2013 split-sum) + analytic DFG polynomial approximation (Lazarov 2013, no baked LUT) + Turquin 2019 multiplicative multi-scatter energy compensation
- fix: two real energy-conservation bugs caught by `tools/furnace_test.cpp` (Monte Carlo furnace test): a missing diffuse/specular energy split, and compensation overshoot at high roughness/near-white F0 (now clamped to the DFG approximation's measured accuracy)
- feat: screen-space AO dropped from scope (a baked AO pass already exists); AO now modulates only the new ambient/IBL term
- feat: HUD sky background toggle (raw equirect) and a 0–359 degree HDR rotation control: diffuse and specular both rotate via the query direction rather than re-baking, so sky/diffuse/specular stay in sync at zero extra bake cost
- feat: IBL AOV shows lighting only (diffuse albedo excluded; specular keeps its real F0)
- feat: `scene.json`'s single `light` object replaced by a `lights` array (empty is a valid IBL-only scene); `json_scan` gains array-of-objects parsing
- refactor: debug HUD panel reordered (Histogram/AOV moved above Camera), Sky/HDR rotation controls moved into their own HDRI section
- docs: README updated throughout (phase table, component table, AOV table, open questions, references)

**Phase 4: Direct lighting & acceleration complete.**

## Phase 5: Materials & recursive transport

### A: Plumbing
- feat: `ShadingTriangle`/`ShadingVertex` (`shading_scene.h/.cpp`): per-triangle world-space normal/uv/tangent/position, parallel-indexed to `Bvh`'s triangle list; built in `gltf_loader.cpp` from data `Mesh` would otherwise discard once uploaded to the GPU
- feat: `Bvh::Hit` gains barycentric u/v (Moller-Trumbore convention, w=1-u-v on v0)
- feat: `EnvironmentMap` retains the equirect `HdrImage` the rasterizer's IBL bake discards after startup, for path-traced miss-ray lookups; direction-to-UV mapping matches `sky.frag` exactly
- feat: `MaterialTexture`: every material texture now retained on both GPU (rasterizer) and CPU (`HdrImage`, path tracer's per-sample UV lookups); `HdrImage::sampleBilinear` added
- feat: `Material::ior`/`transmissionFactor` (`KHR_materials_ior`/`KHR_materials_transmission`)
- feat: `SceneConfig` gains `samplesPerPixel`/`maxBounces`/`russianRouletteStartBounce`
- chore: `Texture::createFromExr` removed: dead code once `MaterialTexture` decodes an EXR once and reuses it for both the GPU upload and the CPU copy

### B: Sampler + BSDF core
- feat: `Sampler`: randomized Halton (radical inverse + per-pixel Cranley-Patterson rotation) for the first 32 dimensions, PCG32 (O'Neill 2014) beyond that
- feat: stochastic metallic-roughness BSDF (`bsdf.h/.cpp`): exact dielectric Fresnel (PBRT's `FrDielectric`) + Snell refraction with TIR, Schlick conductor Fresnel, Heitz 2018 GGX VNDF importance sampling, three-way stochastic lobe selection (rough specular reflection / Lambertian diffuse / smooth delta transmission), combined via a one-sample mixture estimator
- feat: `tools/bsdf_validate.cpp`: pdf-normalization and furnace-test checks, including colored conductors and the transmissive dielectric's exiting/TIR side
- fix: three real energy-conservation bugs caught during development: diffuse lobe missing its (1-F) factor; transmission's `transmissionFactor` cancelling out of its own throughput ratio; diffuse lobe's pdf incorrectly gated on the same term as its value (starved the MIS mixture denominator specifically for colored, non-white conductors)
- fix: opaque materials (`transmissionFactor=0`) no longer trigger transmission/TIR logic when grazing-angle normal mapping pushes the local view direction to the "wrong" side of the shading normal
- note: Sobol+Owen scrambling, Tokuyoshi & Eto 2023's bounded VNDF sampling, and Heitz et al. 2016's stochastic multi-scatter random walk were the original plan but are deliberately not implemented: each depends on precise constants/derivations not safely reproducible from memory alone; the better-established alternatives above are used instead, with the originals kept as documented future upgrades (README §5)

### C: Integrator
- feat: `Camera::primaryRay`: pinhole ray generation sharing `viewMatrix()`'s forward/right/up basis
- feat: `renderPathTraced` (`path_tracer.h/.cpp`): BVH intersect, material/shading resolution (tangent-space normal mapping only, no bump-detail blend), BSDF sampling, geometric-normal-consistency rejection (normal-map light-leak mitigation, in place of Schüßler et al. 2017's full two-facet reconstruction), Russian roulette from `russianRouletteStartBounce`, Chiang/Li/Burley 2019 shadow-terminator-corrected ray origins, environment radiance on a miss; row-parallel `std::thread`, one thread per hardware core, dynamic scheduling via an atomic row counter
- note: no NEE this phase (Phase 7 scope): punctual lights have no hittable geometry, so path-traced radiance comes from the environment map only

### D: Engine integration
- feat: on-demand "Path Traced" HUD section (Enable checkbox, Beauty/IOR/BounceCount AOV combo distinct from the rasterizer's own, Render button, status readout)
- feat: `presentFrame` blits the cached path-traced result through the existing OCIO/post-process path when active; rasterizer keeps running underneath regardless: deliberate simplification, this is an on-demand feature, not real-time
- feat: `Texture::id()` public accessor, needed for the path-traced display texture to reuse `PostProcessPass::draw`'s raw-GL-id present path

### E: Verify
- verified: `bvh_validate`/`furnace_test`/`bsdf_validate` all pass; visual smoke test in the running app via a temporary auto-trigger hook (reverted before commit): IOR AOV shows a correct stump silhouette against black with the LUT correctly forced to Raw; Beauty AOV shows a coherent, plausible image
- docs: README updated (phase table, component reference, AOV reference, references)
