# Pipeline

A CPU path tracer. For each pixel it follows light backwards from the camera, bouncing off surfaces until the ray reaches a light or is killed, and averages that over many passes — so the image arrives noisy and cleans up as it converges. A second, far cheaper pass rasterizes the geometry-only debug channels every frame, so those stay sharp and instant while the camera moves.

## How a frame is made

**Startup.** glTF geometry and materials load once into a CPU-resident scene: per-vertex shading data (`ShadingTriangle`) and world-space triangles build an Embree BVH (`EmbreeAccel`) — the tree that makes ray-vs-scene intersection sub-linear — and textures stay CPU-side as `HdrImage`, sampled per ray. An equirectangular HDR environment map loads alongside with a luminance CDF, so a light can be sampled directly rather than found by chance: next-event estimation, NEE. Area lights (`scene.json`'s `lights`) go into the same BVH (`appendQuadLights`), so they occlude and are camera-hittable with no second intersection path.

**Per frame.** Any camera or scene change triggers two independent lanes:

- **Path tracer** — `PathTraceDriver` hands a request to a background pool (one worker per core, row-parallel) and restarts accumulation. Pinhole rays, Embree intersection, then per bounce: sample the BSDF — how the material scatters light ([materials](#materials--lighting)) — and sample a light by NEE, the two combined by multiple importance sampling (MIS) so neither's weak case shows (power heuristic, Veach 1997). Russian roulette terminates recursion, with shadow-terminator-corrected secondary origins (Chiang/Li/Burley 2019) and Beer-Lambert absorption inside transmissive media. Beauty and the nine other path-traced AOVs — an AOV being one selectable output channel, 30 in all — accumulate per pass and publish lock-free to the render thread.
- **Rasterizer** — a synchronous CPU pass (`rasterizer.cpp`) scan-converts the 14 primary-hit AOVs on the render thread every frame: the G-buffer, the surface data visible directly from the camera. It shares `gbuffer_shading.h`'s material sampling with the path tracer, but uses no Embree, no BSDF and no recursion. It is watertight — vertices snap to a fixed-point grid whose precision is derived per frame to keep the int64 edge functions exact, so two triangles sharing an edge cover each pixel centre on it exactly once. That is what keeps these AOVs glitch-free during camera motion; the traced lane restarts only when the selected AOV needs light transport.

**Display.** The render thread blits the selected AOV through OCIO's display transform and draws the HUD, converging over later passes rather than blocking on one long render. No GPU rasterization anywhere: OpenGL is the window, the OCIO blit and ImGui.

# Components

## Camera & display

| Feature | Mechanism |
|---|---|
| Camera / lens | Position/yaw/pitch, film-back and focal length to a derived vertical FOV, feeding rectilinear (pinhole) primary rays — the geometric ground truth the ray and BSDF math is measured against |
| Fisheye lens | Kannala-Brandt polynomial `r(theta) = f·(theta + k1·theta³ + k2·theta⁵ + k3·theta⁷ + k4·theta⁹)`, selected per render, inverted per ray by safeguarded Newton — the OpenCV `fisheye` convention, so a measured lens's radial geometry drops in |
| Photographic exposure | EV100 from aperture/shutter/ISO, as a relative-stops delta against `profile.json`'s default triple (Filament/Frostbite) — a familiar brightness control, not an absolute photometric quantity |
| Linear light pipeline | `HdrImage`/`loadExr`; all shading in linear light, display-encoded only at the final blit — a precondition for correct PBR colour math |
| Display transform | OCIO Display/View (sRGB, Rec.709, Raw), cycled with `L` — a colourimetric encode only, no tone mapping, so values above 1.0 clip honestly instead of hiding under a filmic shoulder |

## Scene loading & geometry

| Feature | Mechanism |
|---|---|
| glTF loading | cgltf, geometry only: vertices baked to world-space triangles at load and every instance given `makeDefaultMaterial()`; glTF materials and image URIs are ignored, the scene JSON binds textures — nothing GPU-resident is needed once the CPU scene exists |
| Tangent-space normal mapping | glTF-supplied per-vertex tangents, Gram-Schmidt re-orthogonalized per ray — surface micro-detail without extra geometry |
| Ray acceleration | Intel Embree's SIMD BVH, CPU, built once at load — sub-linear intersection, without which recursion is unaffordable |
| Primary-hit rasterizer | Edge-function scan conversion (Pineda 1988) with Sutherland-Hodgman frustum clipping, per-frame fixed-point snapping and the top-left fill rule (D3D11.3 spec) — the 14 instant G-buffer AOVs, row-parallel, every frame |

## Materials & lighting

| Feature | Mechanism |
|---|---|
| Stochastic BSDF | Four lobes, stochastically selected, one-sample mixture estimator: EON rough diffuse (Portsmouth/Kutz/Hill 2025), GGX specular sampled by visible normals (Heitz 2018), Walter 2007 rough dielectric transmission with exact Fresnel and total internal reflection, and exact complex-IOR conductor Fresnel via Gulbrandsen 2014's reflectivity/edge-tint parameterisation. Kulla-Conty compensation on both interfaces keeps it energy-conserving at every roughness. Transmission falls back to a Snell delta lobe below the smooth-roughness threshold and at an index-matched interface |
| Volumetric absorption | Beer-Lambert extinction inside a `transmissionFactor > 0` material (`transmissionColor` over `transmissionDepth`, Arnold/OpenPBR convention), single-level medium stack — tinted glass, thick or thin, with no in-scattering ([roadmap](ROADMAP.md) transport #1) |
| Per-object materials | `SceneConfig::materialOverrides` maps a glTF node name to a `materials/*.json`, indexed per instance by `ShadingTriangle::instanceIndex` through both lanes — different objects in one scene carry different materials |
| Per-object textures | `SceneConfig::textures` maps a glTF node name to `{slot: EXR path}` (`baseColorTexture`, `normalTexture`, `bumpTexture`, `roughnessTexture`, `specularTexture`), the only texture source; an unbound slot keeps its neutral default — the scene owns its texture paths, so a DCC re-export that rewrites image URIs cannot break the binding. Validated and loaded before any instance changes; an unknown node, slot or file aborts the scene, as does any unknown scene key, so a retired `textureOverrides` or `texturePath` cannot load silently untextured |
| Constant shading | `shadingModel: "constant"` (`constant.json`): the hit emits `resolveBaseColor` (texture × `diffuseColour` × COLOR_0) two-sided and terminates, scattering nothing — an unlit, measurable surface. It emits at every bounce, so other surfaces see it as light; it is not in `LightSet`, so its BSDF-sampled hit takes MIS weight 1 and NEE never double-counts it |
| Area lights | Rectangular emitters (`scene.json`'s `lights`), one-sided by default, geometry in the BVH and solid-angle NEE sampling (Ureña/Fajardo/King 2013), MIS'd like the environment — the emissive-panel Cornell box, and ReSTIR's prerequisite light set ([roadmap](ROADMAP.md) transport #2) |
| Environment lighting | Equirect HDR map: BSDF-sampled misses plus luminance-importance-sampled NEE, MIS-combined — image-based lighting, one member of `LightSet`. No punctual or directional lights, which have no hittable geometry |
| Environment-light toggle | HUD "Environment Light" (`environment.lightEnabled`, `--env-light`) drops the environment from NEE, MIS and every miss including the background — unlike "Show/Hide Background", which only hides the camera-visible sky, so an HDRI scene can be reduced to its area lights alone |
| Russian roulette | Past `russianRouletteStartBounce`, paths die with a throughput-derived probability and survivors are reweighted by `1/p` — finite recursion, unbiased estimator |
| Progressive accumulation | Each background pass re-traces at the current camera and averages in, restarting on any change — interactive without waiting out one long render |

## Debug tooling & telemetry

| Feature | Mechanism |
|---|---|
| Startup spec block | GPU and driver, CPU topology, compiler/build flags, git SHA, library versions and the scene's load and BVH-build cost, on stdout — makes every timing number attributable, and catches a wrong-adapter bug before it looks like a render bug |
| Render telemetry (`-stats`) | Terminal dashboard redrawn in place: render-thread stages, path-trace phases, ray counts with Mray/s, and a `cpu total` / `frame measured` / `unaccounted` reconciliation — where every millisecond goes, and what it cannot account for |
| Frame pacing | `DisplayLink` (`display_link.mm`) wakes the render loop once per vblank of the window's own display, at swap interval 0 behind a one-frame GPU fence; NSGL's swap interval lets two swaps through per refresh on current macOS and GLFW sleeps a fixed 60 Hz while occluded, so neither is used. `render.vsync: false` runs uncapped, still one frame in flight. `swap_ms` contains a synchronous WindowServer query inside NSOpenGL's flush and is almost entirely off-CPU, so a per-frame maximum of `swap_ms` or `frame_ms` measures WindowServer, not the pathtracer |
| Frame-timing HUD | Ring buffer of recent frame times, rolling FPS/avg/min/max, and a GPU timer query around the post-process blit |
| Memory HUD | Live RAM plus GPU allocation tracked at alloc/free (the display texture is the only GPU allocation left) — catches a regression before VRAM exhaustion |
| Scene stats | Object, triangle and point counts, plus the authored render resolution |
| Debug camera controls | WASD/QE fly, `0` reset, LMB-drag orbit around a pivot read from the path tracer's own G-buffer at the centre pixel |
| Camera framing overlay | Centre crosshair on the foreground overlay, never in the AOV buffers being debugged |
| AOV selector | Dropdown across the full AOV set, plus R/G/B channel-isolation hotkeys |
| Live histogram | Per-channel histogram of the displayed image — catches exposure, clipping and colour-space bugs one still frame hides |
| Benchmark log | Append-only JSON Lines, one record per timing run (`bench_log.h`): git SHA, Mach-O `LC_UUID`, host topology, argv, resolved config, per-iteration samples, rusage, output CRC. `bench_compare` runs randomized interleaved trials (Abedi & Brecht 2017) and reports a Hodges-Lehmann ratio with its exact Wilcoxon interval — a timing claim stays resolvable when run-to-run drift exceeds the effect |

## Session settings (`profile.json`)

Session settings live in `assets/config/profile.json`.

`render.width`/`render.height` are the authored image in **pixels** and the single authority on what gets traced: the GUI, `render_beauty`, `raster_bench` and the C/Python API all size from them. `renderScale` and `interactiveRenderScale` are fractions of that, not of the window.

The window is an independent viewport. It opens 1:1 in points — a 1024x576 image is a 512x288-point window on a 2x display — and resizes freely without changing what is traced or restarting an accumulation: the image is letterboxed to preserve aspect and magnified with `GL_NEAREST`, so one traced pixel reads as one block rather than an interpolated value that was never rendered, and a 320x180 render can be inspected full-screen. The pixel probe reads nothing over a bar; the histogram bins the image rect alone.

`camera.lens` selects the projection and always carries the polynomial, so the HUD dropdown (`Rectilinear` / `Fisheye Polynomial`) can switch either way without reloading. `projection` is `"rectilinear"` (the pinhole) or `"fisheyePolynomial"`; `radialCoefficients` are `k1..k4` of `r(theta) = f·(theta + k1·theta³ + k2·theta⁵ + k3·theta⁷ + k4·theta⁹)`, dimensionless and identical to what OpenCV's `fisheye` module or COLMAP's `OPENCV_FISHEYE` reports; `maxFieldOfViewDegrees` is the full angle across the image circle. Loading rejects a field of view outside `(0, 360]` and any coefficient set whose `r(theta)` is not *provably* monotone over it, because a non-monotone radius has no unique inverse for `primaryRay` to recover.

The image circle's radius is `r_max = f·theta_d(theta_max)`, measured against the gate, so the focal length is what makes the projection visible: at the shipped 70 mm a 180-degree lens puts a 110 mm circle behind a 13.37 mm gate, the frame is the central 11 degrees, and the render reads as a long lens with faint barrel distortion — physically correct and visually indistinguishable from the rectilinear projection (measured relMSE 3.1% on a Cornell box). The circle enters that gate below 8.68 mm, which is why the HUD's focal slider floors at 6 mm. Short focal lengths are the fisheye's regime, on this camera as on a real one, and there is deliberately no mode that derives the focal length from the gate: it would make the focal slider inert.

Only the radial geometry of a calibration transfers: one focal length means `fx == fy`, and the image circle is centred on the sensor, so a calibrated camera's principal point and pixel aspect are not reproduced. A worked equisolid-angle authoring, whose coefficients are the degree-9 Taylor expansion of `2f·sin(theta/2)`:

```json
"focalLengthMm": 8.0,
"lens": { "projection": "fisheyePolynomial", "maxFieldOfViewDegrees": 180.0,
          "radialCoefficients": [-0.0416667, 0.000520833, -3.100198e-06, 1.0764e-08] }
```

Where the circle falls inside the gate the corners are unimaged and render black in every AOV; the samples still carry their reconstruction-filter weight, so the circle edge antialiases. A fisheye has no rasterizer projection, so the 14 scan-converted G-buffer AOVs are refused rather than approximated: the headless request fails, and the HUD greys those entries out and falls back to Beauty.

`render.vsync` caps the frame rate to the display's vblank (`true`) or runs uncapped (`false`). Uncapped, the render thread competes with the trace workers for cores: `pass_ms` **1.58x** on cornell, **1.47x** on a 4K-textured asset, 8 cores.

`render.textureBitDepth` sets CPU storage for the environment HDRI and every material texture: `16` (default) or `32`. Every shipped EXR is half on disk, so `16` is lossless for them — renders are bit-identical, `pass_ms` is **0.978x** (half the bytes per texel fetch), peak RSS drops **2103 -> 1281 MB** on a 4K-textured asset, exactly the analytic 822 MB. A texel that overflows binary16 is rejected at load, not silently clamped. The environment's CDFs are built from the stored values, so sampling stays proportional to the radiance returned at either depth.

`render.displayBitDepth` sets the display texture's GL storage: `16` (`GL_RGBA16F`) or `32` (`GL_RGBA32F` — twice the memory and sampling bandwidth, `present_gpu_ms` **1.44x**, though `upload_ms` falls as the driver's float-to-half conversion disappears).

## Benchmark tooling

`pathtracer -bench` also takes `-bench-aovs "Beauty,Sobel,Direct Diffuse,Normal,Beauty"`, walking that sequence and timing each switch into a `stage_wall_ms` column; the first entry is an unmeasured warm-up, so every switch is timed from a converged image. `config` carries only configured inputs, never a measured one — the refresh a run was paced at is a per-frame `refresh_hz` sample, since a measured double cannot satisfy an equality contract.

`bench_compare` prints B/A with a distribution-free confidence interval, and says "not resolved" when that interval contains 1. Its `compare` and `history --tool T` read records already in the log (unpaired, so drift is not controlled). `--metric` picks any samples column — its mean per event, since frame count scales with run duration — or a rusage field such as `user_s`.

## Testing

Every validator runs on a shared harness (`tools/check.h`): each check is registered by name and discovered into `ctest` as its own entry (`<suite>.<check>`), labelled by speed (`fast`/`slow`) and kind (`exact`/`statistical`). `ctest -L fast -L exact` is the sub-second pre-commit gate; the full suite is the pre-merge gate. Monte Carlo bands come from the run's own variance over independent scramble seeds at one family-wise significance level (`tools/stats.h`), not hand-picked. `render_beauty --assert-deterministic` and `--assert-converged` gate the shipping pipeline with no golden image.

# Material Library

Presets live in `assets/materials/*.json`, parsed into `MaterialConfig` (`scene_config.h`). A scene picks a default with `SceneConfig::materialPath` and overrides individual objects with `materialOverrides` — e.g. `cornell.json`'s `{"sphere01": "materials/chrome.json", "sphere02": "materials/glass.json"}` ([per-object materials](#materials--lighting)). Add one by dropping a JSON file in and pointing at it; no code or schema change. Textures bind the same way, per node in the scene: `macbeth.json`'s `textures` gives both grids `textures/macbeth.exr`, and `stump.json` binds the stump's five 4K maps; every glTF carries geometry only.

| File | metallic | transmission | roughness (factor / min) | Notes |
|---|---|---|---|---|
| `principled.json` | 0.0 | 0.0 | 1.0 / 0.045 | Rough dielectric, the only preset with a bump (`bumpStrength: 1.0`). The reference preset, the only one declaring every field. No shipped scene uses it |
| `clay.json` | 0.0 | 0.0 | 0.5 / 0.045 | Neutral matte dielectric, no bump |
| `chrome.json` | 1.0 | 0.0 | 0.05 / 0.045 | Measured chromium (Johnson & Christy 1974), `diffuseColour` and `edgeTint` verbatim `tools/metal_fit` output, so the grazing reflectance dip is the measured one. `colour.chrome_matches_measured_chromium` fails if this file drifts from the fit |
| `constant.json` | — | — | — | `shadingModel: "constant"`: no lighting, beauty = the base-colour texel; any BSDF key is rejected. `macbeth.json` binds it to `grid01` for the unlit ColorChecker |
| `glass.json` | 0.0 | 1.0 | 0.02 / 0.01 | Schott N-BK7, `ior: 1.5168` at the d line with `abbe: 64.17` for dispersion, tinted by `transmissionColor` over `transmissionDepth: 0.4` world units |

## `MaterialConfig`

| Field | Meaning |
|---|---|
| `diffuseColour` | Multiplies `baseColorTexture`, and is the conductor lobe's `f0` tint. A **reflection** quantity throughout — it never tints transmitted light. On the diffuse lobe it is the **observed** albedo (OpenPBR's `base_color`), not EON's ρ; the two differ once `diffuseRoughness > 0`, and `eonAlbedoInversion` (`bsdf.cpp`) maps between them |
| `metallicFactor` / `transmissionFactor` | Lobe selection — a material is dielectric, conductor or transmissive, not blended between (every shipped file uses 0.0 or 1.0) |
| `roughnessFactor` | Multiplies the roughness texture sample, before the `roughnessMin`/`roughnessMax` clamp |
| `roughnessMin` / `roughnessMax` | Per-material clamp on that sample; a material can floor below the shared 0.045 (glass uses 0.01) for a genuinely smooth GGX lobe |
| `ior` | Dielectric IOR, non-metal lobes only |
| `abbe` | Abbe number, dispersion strength for the dielectric and transmissive lobes; 0 = no dispersion |
| `diffuseRoughness` | EON rough-diffuse parameter r ∈ [0,1]; 0 = Lambertian, and the roughness at which `diffuseColour` and EON's ρ coincide |
| `bumpStrength` | Scales the bump texture's per-texel height difference |
| `transmissionColor` / `transmissionDepth` | The **only** tint on transmitted light (Arnold/OpenPBR convention). `transmissionDepth > 0` gives interior Beer-Lambert absorption, `sigmaA = -log(transmissionColor)/transmissionDepth`; `transmissionDepth 0` applies the tint once per surface crossing, so a closed solid reads its square. Default `[1,1,1]`/`0.0` is a no-op either way |
| `shadingModel` | `"standard"` (default) is the BSDF above; `"constant"` admits only `diffuseColour` and leaves every BSDF field at its identity, which the G-buffer AOVs still read |
| `edgeTint` | Gulbrandsen 2014 edge tint, `metallicFactor > 0` only. Default `[1,1,1]`: white is the no-dip edge Schlick always produced |

# AOV

31 selectable channels (`aov.h`), in HUD order. **Source** is which producer computes one: `traced` accumulates over passes (10 lanes), `raster` is exact and instant every frame (14), `filter` is an image-space pass over a finished Beauty (7).

Filters run on the CPU only (`debug/aov_filters.cpp`), for both the viewer and the headless API — there is no GLSL copy. `evaluateFilterAov` is the single dispatch, with no `default` arm, so `-Werror` rejects a new filter AOV that nothing routes. In the viewer one filter evaluation is cached per published pass (`FilterCache`), keyed on the pass's owner, generation and sample count, so a filter costs nothing per displayed frame; `filterMs` on the `-stats` dashboard and in the benchmark log reports it.

Display exposure now follows `aovCarriesRadiance`, not the AOV's producer: the six radiance lanes and the four filters that are positively homogeneous of degree one in radiance (Luminance, Sobel, Gabor, DoG) take the photographic exposure, and everything else — ratios, reflectances, counts, lengths, directions, frequencies — stays at unity gain. `relativeExposureEv()` is 0 at the authored camera, so the default appearance of every AOV is unchanged.

### Signed preview

`DoG` and `Colour Opponent` are responses of a zero-mean operator: zero is the operator's own centre and both signs are equally meaningful. A plain gain clips the negative half to black, so `aovIsBipolar` routes them through an affine display map instead — `0.5 + value/(2·range)`, zero landing exactly on mid-grey. The whole display path is that one affine map, `gain ⊙ value + offset`, carried as two `vec3` uniforms on the shader and as two `glm::vec3` arguments on the CPU encode `render_beauty` and the C ABI share, so a PNG and the viewer agree by construction. Radiance is the case where the gain is a scalar photographic exposure broadcast to three lanes and the offset is zero. It is a *preview*: `aovCarriesRadiance` still describes the value, the EXR, the HUD probe and the C ABI all read the raw signed float, and no headless output changes.

The range is not the peak. A rendered signed response is heavy-tailed — at 2048×1152 a Cornell box emitter puts `DoG`'s maximum 36× above its own 99th percentile — so scanning for `max |value|` leaves 97% of the display range unused. The range is instead `min(peak, σ·sqrt(2 ln n))`, with σ the RMS about zero and the cap the concentration point of the maximum of `n` standard normals (Cramér 1946). A field with no tail keeps its true peak and never clips; a field with one outlier lets that outlier saturate rather than crush everything else. Both limbs are gated, in either direction.

**The range is per lane, not per image.** Lanes of one AOV can be different physical quantities. `Colour Opponent`'s `l - l_white` is a dimensionless cone fraction spanning `[-0.150, +0.170]` over the entire Rec.709 gamut, while its `s - s_white` is S excitation per unit luminance and reaches `+13.087` on the blue primary — a 77× disparity. Pooling both into one range is a category error, and it cost the `l` axis all but 20 of 256 display levels on a Cornell box, hiding the red and green walls entirely. Equalising the two axes inside the filter was the alternative, and it has no principled form: that needs Mullen 1985's chromatic CSFs, which wait on a spectral path (ROADMAP), so any scaling written now would be an invented constant. Ranging each lane on its own statistics is exact by construction and collapses to the scalar case for a broadcast AOV, whose three lanes are replicas and therefore range identically.

**A lane the AOV does not define stays black.** `aovChannels` reports what the operator means, so for `Colour Opponent`'s third lane — a structural zero, never a measurement — the map takes zero gain and zero offset rather than sending it to mid-grey. Otherwise an absent lane is indistinguishable from a genuine zero response, and the constant it contributes tints the whole frame. `UV`, whose third lane is also structurally zero, already read black. A scalar AOV is the exception and must not be blanked: `writeScalar` broadcasts it across all three lanes, so all three are ranged and its preview stays grey.

Direction and position lanes are signed but deliberately excluded: a normal's components span a sphere, not a response about zero.

### Log preview, and the one shared display decision

`SNR` is positive, dimensionless and unbounded, so neither an exposure nor a bipolar map suits it. At unity gain every texel at or above 1.0 encodes to 255 and the frame is solid white, carrying exactly one bit: *is this texel above its own noise floor*. Its preview is therefore decibels, `20·log10(SNR)` — the factor is 20, not 10, because `mu/SE` is a ratio of like-dimensioned amplitudes whose power ratio is its square (EMVA 1288's convention for imaging SNR).

**Both anchors of the window are definitional, so neither is an authored constant.** The floor is **0 dB, the ratio 1**, where a texel's value equals its own uncertainty and nothing is resolved. The ceiling is **`10·log10(n)` dB, the ratio sqrt(n)**: since `SNR = sqrt(n)·(mu/sigma)`, that is the SNR of a texel whose *per-sample* coefficient of variation is unity, so the grey reads how far in decibels a texel has come from its noise floor toward that reference. The decibel factor cancels between the two, so the implementation is `2·ln(SNR)/ln(n)` and no base conversion is written. This window is not invariant to the pass count — `grey = 1 − log(CV)/log(sqrt(n))`, so a fixed-`CV` texel drifts toward white as `n` grows, which is the convergence progress one selects `SNR` to watch. True invariance would need a second anchor on `log(CV)` beyond `CV = 1`, and there is no definitional one.

The window deliberately saturates the easy majority. On a Cornell box at 64 passes 78% of texels already exceed `sqrt(n)`, and at 1024 passes 67% do; the remaining fifth to third is the noisy tail — caustics, indirect corners, grazing geometry — and it holds 211 distinct display levels where the linear preview held one. Nothing sits at the floor, so the full range serves the texels that are actually undecided.

A log is not affine, so it cannot ride the shader's two `vec3` uniforms. It is a **pre-map**: one CPU pass over the frame, at upload rather than per frame, writing the display's texels. `Bounce Count`'s Turbo false colour (Mikhailov 2019) is the same mechanism and the same cost, its domain the real bound `maxBounces + 1`. Both live in `aovDisplay`, which is the one function making every display decision that has to read the values — the bipolar per-lane range, `Depth`'s auto-range and these two pre-maps — so the viewer and `render_beauty` share it and a PNG matches the viewer by construction rather than by two copies staying level. The photographic exposure is the one arm left to the caller, because the slider moves without rebuilding the texture; `aovTakesDisplayExposure` states its precedence once, since an auto-ranged AOV has already absorbed the scene's scale and applying an exposure on top would range it twice.

## Scale space

`debug/scale_space.h` is the one Gaussian facility, shared by DoG and Gabor. The kernel is Lindeberg's **discrete** Gaussian `T(n;t) = e^{-t} I_n(t)` (Lindeberg 1990), not a sampled continuous Gaussian, because only it satisfies the discrete scale-space axioms and because the exactness is what makes the validators exact rather than approximate:

| Property | Sampled Gaussian | `e^{-t} I_n(t)` |
|---|---|---|
| `sum w = 1` | needs renormalising | exact, by `sum I_n(t) = e^t` |
| variance | approximately `t` | exactly `t` |
| semi-group `T(t1)*T(t2) = T(t1+t2)` | approximate | exact on the grid |
| transfer function | approximate | exactly `exp(-t(1 - cos w))` |

`std::cyl_bessel_i` does not exist on libc++, so `I_n(t)` comes from the ascending series `I_n(t) = sum_k (t/2)^(n+2k) / (k! (n+k)!)`. Every term is positive, so there is no cancellation and the sum is relatively accurate to double precision; the magnitude is carried in the exponent and the series summed relative to its own first term, so neither a large order nor a large `t` can overflow or underflow it. Miller's downward recurrence was rejected: it needs a starting order chosen by a rule of thumb, which is exactly the authored constant this codebase's numerics avoid.

**Every threshold in the file is float32's unit roundoff, 2^-24**, and nothing else:

- **Truncation.** Taps are emitted outward until the discarded tail mass falls below 2^-24, below which it cannot perturb a float result. That lands near `R = 5.4 sqrt(t)`, wider than the customary 4σ, which is the point. Truncated weights are renormalised over the emitted taps, not against the infinite sum: a kernel with gain `1 - 2^-24` would compound across successive diffusions.
- **Inner scale.** The finest scale the base grid represents faithfully is where the transfer at the grid Nyquist `w = pi` reaches 2^-24: `exp(-2t) = 2^-24`, so `t = 12 ln2 = 8.3178` (σ = 2.884 px). Below that a level still carries energy at Nyquist and is not a sampling of a band-limited function.

DoG is the first band of an octave ladder (Burt & Adelson 1983's Laplacian pyramid, Lindeberg's kernels): the fine level at the inner scale `t0`, the coarse level one octave up at `4·t0`, reached from the fine level by a `3·t0` diffusion that the semi-group makes exact. One octave is not a tuning choice: successive-octave DoG has ≈1.2-octave bandwidth, matching the measured human spatial-frequency channels (Wilson & Bergen 1979; De Valois et al. 1982). The band is **empty where the coarse step's kernel support outgrows the frame**, since such a level reports the mirrored boundary at every sample, not the image.

Rejected, with reasons, so they are not revisited:

- *Recursive IIR* (Deriche 1993; Young & van Vliet 1995; Alvarez & Mazorra 1994). O(1)/px, but rejected on **correctness**: it approximates the Gaussian, so the semi-group is approximate; its boundary initialisation is itself an approximation (Triggs & Sdika 2006 exists to patch it); and it is not a symmetric FIR, so it does not annihilate affine fields exactly. That last property is what gives "DoG reads zero on a ramp".
- *Burt & Adelson's 5-tap binomial kernel.* Only approximately Gaussian, and its variance is not `t`.

Two implementation choices carry the exactness. **Convolution is centre-relative**, `out = c + sum_n w_n ((l - c) + (r - c))`: on an affine field every tap pair cancels bit-exactly, so diffusion reproduces a ramp identically and DC gain is exactly 1 by construction rather than by normalisation. It also reduces cancellation in DoG, which differences two nearly equal blurs. **The boundary mirrors about the edge sample without repeating it** (whole-sample symmetry), so every tap lands on real data and the finite operator stays diagonal in the cosine basis; a mirror turns a ramp into a tent, so validators scope to the interior by the radius the facility reports.

## Estimator noise

`SNR` reports how well each published texel is known, not how noisy one sample is: `beauty` is a mean, so the denominator is the standard error **of the mean**, `sqrt(M2 / (n(n-1)))`. The *value* is linear, matching the repo's raw-value convention — the EXR, the HUD probe and the C ABI all read the raw ratio; only the preview is in decibels, exactly as the signed preview is a preview of a raw signed float.

**Single-image spatial estimators are rejected on principle.** Immerkaer 1996 (whose 3x3 mask is a scaled discrete Laplacian) and Donoho & Johnstone 1994's MAD of the finest wavelet subband both estimate *one global sigma under additive white Gaussian noise*. Monte Carlo render noise violates that on three counts: it is heteroscedastic by orders of magnitude between a directly lit diffuse texel and a caustic, it is signal-dependent, and it is spatially correlated through NEE, environment importance sampling and the shared Owen-scrambled Sobol sequence. Both would also report a tessellated silhouette as noise.

The estimate is instead exact and free: `accumulateMean` already holds this pass's radiance in the destination and the previous mean in the source, which is precisely what Welford's recurrence needs, so the second moment is one extra pass over `beauty` in a loop that was already streaming it. `renderPathTraced` is untouched, so the hot path pays nothing. It is Welford 1962 in West 1979's `(x - m_prev)(x - m_new)` form, the same recurrence and the same `invN` the ten RGB lanes use; `running_m2_matches_batch_variance` gates it against the two-pass sum of squared deviations under the forward-error bound `running_mean_matches_batch_mean` already derives.

**One float per texel, not an eleventh image.** 9 MiB a pool slot at 2048x1152 against 36 MiB for a full RGBA lane, and luminance M2 is computed on the per-pass luminance directly, so it is exact for luminance — it is *not* recoverable from three per-channel moments, which would need the inter-channel covariance shared paths induce.

The headless path accumulates by naive summation and one divide, not Welford, and that is deliberately unchanged: the moment is carried *beside* the sum, reading the previous mean back out of it, so every published mean stays bit-identical to what it was.

## Oriented bands

`Gabor` is a 2-D Morlet wavelet bank (Morlet 1982; Antoine & Murenzi 1996) — an isotropic Gaussian envelope times a plane wave, minus its own mean. Its three parameters were authored (`sigma = 1.4`, `lambda = 4.0`, `gamma = 0.5`, on a 5x5 support); all three now follow from the shared scale space and from one another.

**Envelope.** `sigma^2 = innerScaleVariance()`, the finest scale the grid resolves at float32 precision and DoG's fine scale, so the two operators resolve the same band. `sigma = 2.884 px`.

**Carrier.** The half-response radial bandwidth fixes `sigma/lambda` alone (Petkov 1995 eq. 4): `sigma/lambda = (1/pi) sqrt(ln2/2) (2^b + 1)/(2^b - 1)`. The bandwidth is DoG's own octave spacing, `b = 1`, so `sigma*omega0 = 2 sqrt(ln2/2) * 3 = 3.532` and `lambda = 5.130 px` — a carrier 2.57x inside the grid Nyquist, which is checked rather than assumed.

**Orientations.** Because `sigma*omega0` depends only on `b`, so does the angular half-response width: the filter is a Gaussian bump of radial standard deviation `1/sigma` centred at radius `omega0`, half response at a chord of `sqrt(2 ln2)/sigma`, giving `2 asin(sqrt(2 ln2)/(2 sigma omega0))` either side, **38.4 degrees** in total. Covering a half turn at half response or better therefore needs `ceil(180/38.4) = 5` orientations. There is no free choice here: the bandwidth fixes the count. A half turn suffices because the bank is complex and `theta` and `theta + pi` are conjugates with the same magnitude.

**Quadrature, so no carrier phase to choose.** The old bank used an odd (sine) carrier, which was an authored decision between edge and ridge sensitivity. The Morlet response is complex and the AOV reports its magnitude, which is phase-invariant.

**Implementation: modulate, blur, correct.** `(f * G_theta)` factors as `exp(i omega0 x.u) [(f exp(-i omega0 x.u)) * g_sigma]`, an exact identity, so every orientation reduces to two separable Gaussian convolutions of the shared `diffuse` rather than a rotated 2-D kernel. The final remodulation is a rotation by the carrier phase, which the magnitude discards, so it is not computed. At the derived scale a direct 2-D bank would need a 35x35 support per orientation, 6125 taps a texel; this is 11 separable passes instead.

**Admissibility, exactly, at the border too.** A Morlet wavelet is defined with its own mean subtracted. Rather than the ideal `exp(-sigma^2 omega0^2 / 2)`, the term used is the blurred plane wave itself, taken under the same mirror the image gets — which is what a constant field actually produces at that texel. That field is separable, so it is the product of one width-long and one height-long 1-D blur and costs nothing, and it zeroes the DC response everywhere rather than only in the interior. Zero to rounding, not bitwise: the image goes through one fused 2-D blur and the correction through a product of two 1-D ones, and float multiplication does not distribute. Measured at 1.0 to 1.4 float epsilons of the field, over fields spanning four decades. `filters_are_zero_on_a_constant_field` asserts it on a frame smaller than the kernel, where every texel is a border texel.

**The response changed, and the unit changed with it.** The envelope has unit DC gain, so the reported magnitude is the amplitude of the oriented component in radiance units, directly comparable against Luminance; the old bank's weights were unnormalised, so its peak ran about 43x higher with no unit attached. The old bank's passband also sat near pixel Nyquist, so most of what it reported on an unconverged frame was Monte Carlo noise rather than scene structure. `results/wave-morlet-gabor/` holds the before and after.

## Cone space

`scene/cone_space.h` takes linear Rec.709 to cone excitations and then to the two cardinal chromatic axes. Every link is exact on the data this repository already holds, which decides which fundamentals are usable.

**Hunt-Pointer-Estevez, not Smith-Pokorny.** Smith & Pokorny 1975 is the basis MacLeod & Boynton 1979 and every standard DKL implementation are built on, and it was the first choice — but its fundamentals are defined as a linear transform of the **Judd-Vos modified** 2° colour matching functions, not the CIE 1931 ones `cie_1931.inc` tabulates. Nor can that be bridged: Rec.709's primaries are specified as CIE 1931 *chromaticities*, not spectra, so they have no Judd-Vos tristimulus values at all, and no exact route from this codebase's RGB to any Judd-Vos-based cone space exists. Hunt-Pointer-Estevez (Estevez 1979; Hunt 1998 App. 1) is stated **as** a 3×3 on CIE 1931 2° XYZ, normalised so equal energy excites the three cones equally, so the chain is exact by construction and needs no new `.inc`.

The cost is stated rather than hidden: `L + M` is not `V(lambda)`. The Judd modification exists precisely to make luminous efficiency a linear combination of the fundamentals, and *no* fundamentals on the unmodified 1931 basis can satisfy it, so the `(l, s)` plane is not strictly isoluminant. The `s` axis is therefore rescaled by white's `L + M`, so one unit is one S excitation at the achromatic point's cone sum. That is MacLeod & Boynton's own unit where `L + M` would be luminance, and `l` is left untouched.

**MacLeod-Boynton's form, not DKL.** `(l, s) = (L, S)/(L+M)` is exactly invariant to scaling by any positive scalar, which is the property that matters on unbounded scene-referred HDR: it needs no exposure decision. DKL needs an adaptation level, so it would.

The white is Rec.709's own, which is `xyzToRec709()`'s construction point, so RGB `(v, v, v)` is the achromatic origin by definition rather than by a second spectral integration. Each opponent numerator's three RGB coefficients therefore sum to zero, and a zero-sum row is `r.x (R-G) + r.z (B-G)` identically — so the basis stores those two coefficients and an achromatic texel reads **exactly** zero at any intensity, not merely within a rounding of it. Same device as the convolution's centre-relative form.

## Utility

| AOV | Source | Meaning |
|---|---|---|
| Beauty | traced | Final accumulated radiance — the primary output |
| Wireframe | raster | White mesh-triangle edges plus a per-instance bounding box in that instance's ObjectID hue, z-tested against scene depth — triangle density and object extent in one view |
| Alpha | raster | 1.0 on a primary hit, 0.0 on a miss — a real coverage mask, since this renderer is not opaque-only by construction |
| Depth | raster | Planar camera-space Z (Arnold/RenderMan/EXR "Z" convention) at the primary hit, for depth compositing |
| Lookahead | raster | `clamp(1 - Z/lookaheadDistance, 0, 1)`: 1 at the camera plane, 0 at `lookaheadDistance` — proximity on a declared scale, where Depth is unbounded. A miss also reads 0, so Alpha separates "too far" from "nothing there" |
| HSV | filter | Beauty in HSV — isolates hue and saturation shifts a pure RGB view hides |
| Luminance | filter | Rec.709 luminance of Beauty — perceived brightness without colour |
| Sobel | filter | 3×3 Sobel gradient magnitude of Luminance — a cheap edge signal |
| Gabor | filter | Peak quadrature magnitude of a 5-orientation 2-D Morlet bank over Luminance — directional structure Sobel's isotropic magnitude cannot distinguish. Every parameter is derived from the shared scale space; see Oriented bands below. Zero on a constant field to one float epsilon, border included |
| DoG | filter | Signed difference of Luminance at the inner scale and one octave above it, the first band of Burt & Adelson 1983's Laplacian pyramid (σ ratio 2, not Marr & Hildreth 1980's 1.6 LoG fit) — the retinal centre-surround band next to pixel Nyquist. Signed, not a magnitude: polarity separates a bright blob from a dark one, and Sobel already reports magnitude. Exactly zero on any affine field, so it does not fire on a smooth gradient. Zero everywhere on a frame narrower than the coarse step's kernel support |
| WorldPos | raster | Raw world-space primary-hit position — geometry and UV placement independent of shading |
| UV | raster | Interpolated UV at the primary hit, fractional part |

## Perceptual

Observer models over Beauty, as against the Utility block's image-space derivative operators. Derivation under Cone space above.

| AOV | Source | Meaning |
|---|---|---|
| Colour Opponent | filter | Cone-excitation displacement from Rec.709 white, 2 channels: **R** is `l - l_white`, the L-versus-M axis, **G** is `s - s_white`, the S-versus-(L+M) axis, scaled to one unit per S excitation at white's `L + M`. Exactly zero on any achromatic texel at any intensity, and exactly invariant to a gain, so it needs no exposure decision on unbounded scene-referred data |

## Material

| AOV | Source | Meaning |
|---|---|---|
| Normal | raster | Normal-mapped shading normal — the normal shading actually uses |
| GeomNormal | raster | Interpolated vertex normal, before normal mapping — separates a bad normal map from a bad base mesh |
| Albedo | raster | Base-colour texture sample — texture data isolated from lighting |
| Metallic | raster | Per-instance `metallicFactor`, uniform within an object's triangles — which instance carries which value |
| Roughness | raster | Roughness texture times a per-instance factor, floored at that material's own `roughnessMin` |
| Tangent | raster | Shading tangent basis at the primary hit — the basis normal mapping uses |
| ObjectID | raster | Per-instance index, false-coloured (`falseColorForId`) — an isolation mask for compositing |
| AO | traced | Cosine-weighted obscurance: one hemisphere ray per sample bounded by `aoMaxDistance`, each hit weighted `1 - (1 - t/aoMaxDistance)^2` so occlusion grades with proximity. 1.0 is unoccluded, the opposite polarity to Shadow — contact and corner darkening off the geometry alone |

## Transport

| AOV | Source | Meaning |
|---|---|---|
| Fresnel | traced | Expected Fresnel reflectance over the **visible** microfacet normals, `E[F]` from one VNDF draw per sample — the distribution `sampleBsdf` itself draws from, so it is roughness-dependent where a macro-normal value cannot be: on an `ior` 1.5 dielectric at `n.wo` 0.05 it reads 0.7521, 0.4406 and 0.1692 at the roughness floor, 0.3 and 0.6, against 0.7521 throughout for a macro normal. Full RGB, since a conductor's Fresnel is chromatic by construction |
| IOR | raster | Per-instance dielectric IOR, -1 on a miss — the raw index driving Fresnel and transmission |
| BounceCount | traced | Mean path termination depth per pixel — Russian roulette and termination behaviour |
| SNR | filter | Rec.709 luminance of Beauty over the standard error of that mean, `mu / sqrt(M2 / (n(n-1)))`. 0 below two passes, where a variance is undefined, and 0 on a texel every pass agreed on. Reads the Welford second moment `renderPathTraced` never touches — see Estimator noise below |

## Lighting

| AOV | Source | Meaning |
|---|---|---|
| DirectDiffuse | traced | Diffuse radiance from a path's first (bounce-0) surface, base colour included, in Beauty's units |
| IndirectDiffuse | traced | Diffuse radiance from later bounces — the bounced contribution alone |
| DirectSpecular | traced | Specular reflection one bounce from the camera |
| IndirectSpecular | traced | Specular reflection from later bounces |
| Refraction | traced | Radiance from any path that sampled a transmission lobe (a sticky bucket) — glass and transmissive transport |
| Shadow | traced | Binary NEE occlusion toward the sampled light at the primary hit, re-averaged across passes into continuous penumbra density — direct-light visibility without material or light colour |

# References

- Kajiya, J.T. (1986). The rendering equation. SIGGRAPH.
- Veach, E. (1997). Robust Monte Carlo Methods for Light Transport Simulation. PhD thesis, Stanford: MIS, its support condition (§9.2, asserted by `bsdf_validate`'s `strategy_coverage`), and NEE.
- Veach, E., Guibas, L.J. (1995). Bidirectional estimators for light transport. EGRW: vertex connection — [roadmap](ROADMAP.md) transport #4, not implemented.
- Arvo, J., Kirk, D. (1990). Particle Transport and Image Synthesis: Russian roulette termination.
- Christensen, P.H., Jarosz, W. (2016). The Path to Path-Traced Movies. FnT CGV: production grounding.
- Sobol, I.M. (1967); Joe, S., Kuo, F.Y. (2008). SIAM JSC 30(5); Bratley, P., Fox, B.L. (1988). ACM Alg. 659: the sequence, its direction numbers (`sobol_direction_seeds.inc`) and the expanding recurrence.
- Burley, B. (2020). Practical Hash-based Owen Scrambling. JCGT 9(4): the scramble, set-index shuffling and padding in `Sampler`, using Vegdahl's constants as shipped by Cycles.
- Cranley, R., Patterson, T.N.L. (1976). Randomization of number theoretic methods: the per-pixel toroidal shift blue-noise dithering drives.
- Georgiev, I., Fajardo, M. (2016). Blue-noise Dithered Sampling. SIGGRAPH Talks: the tiled blue-noise shift, adopted at d = 1; the annealed d-dimensional matrix is open in the [roadmap](ROADMAP.md).
- Pascale, D. (2006). RGB coordinates of the Macbeth ColorChecker, BabelColor; spectra from BabelColor (2012) ColorChecker_RGB_and_spectra: the 30-chart average `tools/colorchecker_texture` integrates into `macbeth.exr`.
- CIE 167:2005. Recommended Practice for Tabulating Spectral Data for Use in Colour Computations: the Sprague (1880) interpolant and boundary coefficients `resampleSprague` implements.
- Ulichney, R.A. (1993). The void-and-cluster method for dither array generation. SPIE 1913: the mask's construction, re-runnable in `tools/bluenoise_mask.cpp` rather than a lifted tile.
- Dupuy, J., Jakob, W. (2018). An Adaptive Parameterization for Efficient Material Acquisition and Rendering. ACM ToG 37(6): one interpolant for value and density, the construction both transmissive multiple-scattering shares use.
- Zwicker, M. et al. (2015). Adaptive Sampling and Reconstruction for Monte Carlo Rendering. CGF STAR: denoising survey — [roadmap](ROADMAP.md) transport #6, not implemented.
- Xiao, L. et al. (2020). Neural supersampling for real-time rendering. SIGGRAPH — [roadmap](ROADMAP.md) transport #7, not implemented.
- Ho, J. et al. (2020). NeurIPS; Rombach, R. et al. (2022). CVPR: diffusion foundations — [roadmap](ROADMAP.md) transport #8, not implemented.
- Novák, J. et al. (2018). Monte Carlo Methods for Volumetric Light Transport. CGF STAR — [roadmap](ROADMAP.md) transport #1, not implemented.
- Jensen, H.W. et al. (2001). SIGGRAPH; Christensen, P.H., Burley, B. (2015): BSSRDF and diffusion profiles — [roadmap](ROADMAP.md) transport #1, not implemented.
- Cook, R.L., Torrance, K.E. (1982). ACM ToG: BRDF and Fresnel foundations.
- Walter, B. et al. (2007). Microfacet models for refraction through rough surfaces: GGX, the rough-refraction BTDF, and the per-microfacet reflect/refract choice `facetReflectProbability` implements.
- Heitz, E. (2014). Understanding the Masking-Shadowing Function: the height-correlated Smith term (`smithVisibility`). Its transmissive Beta form is deliberately not used — see the [roadmap](ROADMAP.md)'s Smith-exact entry.
- Heitz, E. (2018). Sampling the GGX Distribution of Visible Normals. JCGT 7(4): the VNDF routine the specular lobe uses.
- Heitz, E. et al. (2016). Multiple-scattering microfacet BSDFs with the Smith model: the reference instrument for exit distributions, and the statement of the energy single scatter discards.
- Kulla, C., Conty, A. (2017). Revisiting Physically Based Shading at Imageworks. SIGGRAPH course: the shipped directional-albedo multiple-scattering compensation, on both interfaces.
- Turquin, E. (2019). Practical multiple scattering compensation: evaluated against Heitz 2016 and not adopted; crossover in the [roadmap](ROADMAP.md).
- Guy, R., Agopian, M. (2018). Filament §4.4.2: the cancellation-free Trowbridge-Reitz denominator `distributionGGX` evaluates.
- Gulbrandsen, O. (2014). Artist Friendly Metallic Fresnel. JCGT 3(4): the conductor reflectivity/edge-tint parameterisation, with two documented departures from its listing (see `bsdf.cpp`).
- Portsmouth, J., Kutz, P., Hill, S. (2025, rev. 2026-02-04). EON: A Practical Energy-Preserving Rough Diffuse BRDF. JCGT 14(1): the rough-diffuse lobe, its compensation, CLTC sampling, and Appendix A's albedo inversion (`eonAlbedoInversion`).
- Pharr, M., Jakob, W., Humphreys, G. Physically Based Rendering: `FrDielectric`, `EffectivelySmooth`, and the index-matched delta routing `transmissionIsRough` implements.
- OpenPBR Surface specification; Autodesk Arnold `standard_surface`: the transmission-tint convention this pipeline follows, and the meaning of `base_color`.
- Adobe. OpenPBR BSDF reference implementation, `openpbr_constants.h`: the 620/540/450 nm triple `kRgbWavelengthsNm` takes, and the "discrete RGB bands" limitation ([roadmap](ROADMAP.md) transport #5).
- OpenPBR: Novel Features and Implementation Details (arXiv:2512.23696): the throughput-weighted channel selection the dispersive path uses.
- Khronos. `KHR_materials_dispersion`: Cauchy's relation inverted from an Abbe number, implemented by `cauchyIor`.
- Dupuy, J., Benyoub, A. (2023); Tokuyoshi, Y., Eto, K. (2023): newer VNDF refinements, surveyed, not implemented (Heitz 2018 used instead).
- Schüßler, V. et al. (2017). Microfacet-based normal mapping: not implemented; a geometric-normal-consistency rejection is used instead.
- Belcour, L. (2018). Layered materials by atomic decomposition. ACM ToG: not implemented (single-layer only).
- Chiang, M.J.-Y., Li, Y., Burley, B. (2019). Taming the Shadow Terminator. JCGT 8(4): the secondary-ray origin correction.
- Debevec, P. (1998). Rendering synthetic objects into real scenes: HDR image-based lighting.
- Goral, C.M. et al. (1984). SIGGRAPH: the emissive-panel Cornell box, implemented (`cornell.json`).
- Ureña, C., Fajardo, M., King, A. (2013). An Area-Preserving Parametrization for Spherical Rectangles. CGF: the quad light's solid-angle NEE sampler, as Arnold's `quad_light` and PBRT 12.5.3.
- Lambert, J.H. (1760). Photometria; Baum, D.R. et al. (1989). SIGGRAPH: the closed-form Lambertian-polygon irradiance `integrator_validate` uses as an independent analytic reference.
- Miller, G. (1994); Landis, H. (2002): the cosine-weighted distance-bounded AO the AO AOV path-traces, whose pdf cancels to the mean of the visibility term.
- Zhukov, S. et al. (1998); Iones, A. et al. (2003); surveyed in Mendez-Feliu, A., Sbert, M. (2009): obscurance, the distance falloff rho(x) = 1 - (1-x)^2 the AO AOV uses.
- Bitterli, B. et al. (2020). ReSTIR. SIGGRAPH — [roadmap](ROADMAP.md) transport #2, not implemented: reservoir resampling needs many lights to be worth it.
- Sobel, I., Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing. Stanford AI Project: the Sobel AOV's fixed kernel.
- Gabor, D. (1946). Theory of communication. J. IEE 93(26); Daugman, J.G. (1985). JOSA A 2(7): the oriented Gabor filter the Morlet bank is the wavelet form of.
- Morlet, J. et al. (1982). Geophysics 47(2); Antoine, J.-P., Murenzi, R. (1996). Signal Processing 52(3): the 2-D Morlet wavelet, its admissibility correction and the isotropic-envelope form the Gabor AOV uses.
- Petkov, N. (1995). Biological Cybernetics 76(2) eq. 4: the half-response bandwidth in octaves fixes sigma/lambda, which is where the carrier comes from instead of an authored wavelength.
- Field, D.J. (1987). JOSA A 4(12); Kovesi, P. (1999). Videre 1(3): log-Gabor banks and the coverage argument behind the orientation count.
- Lindeberg, T. (1990). Scale-space for discrete signals. IEEE TPAMI 12(3); Lindeberg, T. (1994). Scale-Space Theory in Computer Vision, ch. 3-4: the discrete Gaussian `e^-t I_n(t)`, its axiomatic uniqueness, and the semi-group and diffusion identities `scale_space.h` is built on.
- Koenderink, J.J. (1984). The structure of images. Biol. Cybern. 50(5); Witkin, A.P. (1983). Scale-space filtering. IJCAI: scale space as the causal one-parameter family DoG and Gabor sample.
- Burt, P.J., Adelson, E.H. (1983). The Laplacian pyramid as a compact image code. IEEE Trans. Comm. 31(4): the band DoG is the first of, with the binomial kernel replaced by Lindeberg's.
- Marr, D., Hildreth, E. (1980). Theory of edge detection. Proc. R. Soc. B 207: DoG as the retinal centre-surround operator, and the zero-crossing reading of the Laplacian.
- Estevez, O. (1979). On the fundamental data-base of normal and dichromatic colour vision. PhD thesis, Amsterdam; Hunt, R.W.G. (1998). Measuring Colour, 3rd ed. App. 1: the cone fundamentals stated as a 3x3 on CIE 1931 2-degree XYZ, which is what makes the Colour Opponent chain exact on `cie_1931.inc`.
- Smith, V.C., Pokorny, J. (1975). Vision Research 15(2): the cone fundamentals MacLeod-Boynton and DKL are built on, surveyed and **not** adopted — they are defined on the Judd-Vos modified CMFs, which this repository's observer is not and cannot be bridged to.
- MacLeod, D.I.A., Boynton, R.M. (1979). JOSA 69(8): the `(L, S)/(L+M)` chromaticity whose exact invariance to a positive gain is why the Colour Opponent AOV needs no exposure decision.
- Derrington, A.M., Krauskopf, J., Lennie, P. (1984). J. Physiol. 357: the cardinal chromatic axes the MacLeod-Boynton plane spans; DKL itself needs an adaptation level, so it is not what the AOV reports.
- Immerkaer, J. (1996). Fast noise variance estimation. CVIU 64(2); Donoho, D.L., Johnstone, I.M. (1994). Biometrika 81(3): single-image noise estimators surveyed and rejected, both assuming one global sigma of additive white Gaussian noise, which Monte Carlo render noise is not.
- Wilson, H.R., Bergen, J.J. (1979). Vision Res. 19(1); De Valois, R.L. et al. (1982). Vision Res. 22(5): the ≈1.2-octave spatial-frequency channel bandwidth the one-octave DoG band matches.
- Abramowitz, M., Stegun, I.A. (1964) 9.6.10: the ascending series for `I_n(t)`, used in place of the absent `std::cyl_bessel_i`.
- Deriche, R. (1993); Young, I.T., van Vliet, L.J. (1995); Alvarez, L., Mazorra, L. (1994); Triggs, B., Sdika, M. (2006): recursive O(1) Gaussian approximations, surveyed and **not** adopted — an approximate semi-group and an asymmetric kernel would give up the exact affine and ramp invariants the validators assert.
- CIE 018:2019 Table 6 (ISO/CIE 11664-1:2019): the 1931 2° colour-matching functions at 1 nm, `cie_1931.inc`.
- ISO/CIE 11664-2:2022 Table B.1: D65 at 1 nm, `cie_1931.inc`.
- CIE 015:2018: tristimulus integration at the 1 nm interval, `cie::reflectanceToRec709`.
- ITU-R BT.709-6 (2015); SMPTE RP 177-1993: the primaries and the matrix derivation, `cie::xyzToRec709`.
- Johnson, P.B., Christy, R.W. (1974). Phys. Rev. B 9, 5056: the measured chromium `(n, k)` behind `chrome.json`.
- Wilkie, A. et al. (2014). Hero wavelength spectral sampling. CGF — [roadmap](ROADMAP.md) transport #5, not implemented; the shipped dispersion commits to one RGB channel instead.
- OpenEXR / Academy Software Foundation: the linear HDR pipeline and exposure.
- Chandrasekhar, S. (1960). Radiative Transfer. Dover: polarised transport, underlying a CPL filter — [roadmap](ROADMAP.md), physical camera filters, not implemented.
- Khronos. glTF 2.0 specification: the scene/mesh/material interchange format.
- Mikkelsen, M.S. (2008). MikkTSpace: not implemented (glTF-supplied tangents only).
- Wald, I. et al. (2014). Embree. ACM ToG: the CPU ray-scene intersection kernels behind `EmbreeAccel`.
- Pineda, J. (1988). A parallel algorithm for polygon rasterization. SIGGRAPH: the edge-function rasterizer.
- Sutherland, I.E., Hodgman, G.W. (1974). CACM: the view-frustum polygon clip.
- Blinn, J.F., Newell, M.E. (1978). SIGGRAPH: the per-vertex outcodes ahead of that clip.
- Microsoft. Direct3D 11.3 Functional Spec §3.4; Khronos. Vulkan, Rasterization: fixed-point snapping and the top-left fill rule, with precision derived per frame from the int64 exactness bound.
- Giesen, F. (2013). Triangle rasterization in practice: integer edge functions, the top-left bias, incremental row stepping.
- Williams, L. (1983). Pyramidal Parametrics. SIGGRAPH: MIP-mapping — [roadmap](ROADMAP.md), texture minification, not implemented.
- Cook, R.L., Porter, T., Carpenter, L. (1984). Distributed Ray Tracing. SIGGRAPH — [roadmap](ROADMAP.md), depth of field and motion blur, not implemented.
- Kannala, J., Brandt, S.S. (2006). IEEE TPAMI: the polynomial fisheye projection `Camera` builds under `LensProjection::FisheyePolynomial`, in the `k1..k4` parameterisation OpenCV's `fisheye` module and COLMAP's `OPENCV_FISHEYE` report.
- Press, W.H. et al. Numerical Recipes 3rd ed. §9.4 (`rtsafe`): the safeguarded Newton that inverts `r(theta)` per ray, bisecting whenever a step leaves the bracket or fails to halve it, so it stops at 4 ulps of `thetaMax` within `2·21` iterations for any lens.
- Farouki, R.T., Rajan, V.T. (1987/88). CAGD: the power-to-Bernstein basis change and de Casteljau subdivision behind `kannalaBrandtIsInvertible`, whose convex-hull test proves `r'(theta) > 0` rather than sampling for it.
- Kalibera, T., Jones, R. (2013). Rigorous Benchmarking in Reasonable Time. ISMM: the process invocation as the unit of replication.
- Abedi, A., Brecht, T. (2017). ICPE: randomized multiple interleaved trials, which `bench_compare run` automates; Mytkowicz, T. et al. (2009). ASPLOS: the ordering and environment bias randomization removes.
- Hodges, J.L., Lehmann, E.L. (1963); Hollander, M., Wolfe, D.A., Chicken, E. (2014) 3.2/4.3: the shift estimators and exact rank intervals in `tools/stats.h`, on the log scale so a shift is a ratio (Fleming, P.J., Wallace, J.J. (1986). CACM 29(3)); Hoefler, T., Belli, R. (2015). SC: nonparametric intervals for performance data.
- Welford, B.P. (1962); West, D.H.D. (1979): the incremental running mean `PathTraceDriver` publishes; Chan, T.F. et al. (1983); Higham, N.J. (2002) §1.9 and ch. 3: the forward-error bound `driver_validate` derives for it.
- Autodesk Arnold `options.stats_file`: the benchmark log's append-per-record format; LLVM `GenerateVersionFromVCS.cmake`: the build-time VCS stamp.
- Apple (2023). `NSView.displayLink(target:selector:)`; Energy Efficiency Guide, QoS classes: the vblank callback `DisplayLink` paces on. GLFW issues #1990/#2249 and PR #2277 record why `NSOpenGLCPSwapInterval` is not used.
