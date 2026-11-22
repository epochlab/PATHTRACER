# PBR Pathtracer

*A physically-based spectral, unidirectional Monte Carlo path tracer with progressive sampling: hero-wavelength transport projected to ACEScg, Embree-accelerated, stochastic BSDF combined with environment and area-light NEE via MIS over a per-path vertex graph, behind a thin OpenGL display/HUD layer.*

![Sample](sample.png)

## Build

C++20, built with CMake. Currently developed against macOS only.

```
brew install cmake glfw glew glm imath openexr opencolorio openimageio embree open-image-denoise openvdb boost
```
```
git submodule update --init --recursive
cmake -B build
cmake --build build
```

## Run

```
./build/pathtracer [-scene path/to/scene.json] [-preset beauty|proxy|research] [-stats] [-bench log.jsonl] [-bench-aovs beauty,normal,...] [-size WxH] [-spp N]
```

## Presets

`assets/config/profile.json` holds every setting. Each file in `assets/config/presets/` is a named preset and an RFC 7396 merge
patch over it. Select one with `-preset NAME`, `render_beauty --preset NAME`, `Renderer(scene, preset="NAME")` or
`pt_renderer_open(..., "NAME", ...)`. `beauty.json` is empty, so beauty is `profile.json` as is. An override may only replace keys
`profile.json` has, kind for kind, and anything else is refused at load.

| Preset | Size | spp | Interactive scale | Camera |
| --- | --- | --- | --- | --- |
| beauty (default) | 2048x1152 | 256 | 0.1 | square pixels, rectilinear lens |
| `proxy` | 512x256 | 16 | 1.0 | as beauty |
| `research` | 64x32 | 64 | 1.0 | hexagonal pixels, omnidirectional lens |

`proxy` is the benchmark workload: `-bench`, `render_beauty` and `gbuffer_bench` default to its size and samples.

A preset sets the sensor, never the pose: a scene's `camera` block places it. `scenes/cornell_research.json` is Cornell seen
from inside the box, the scene `research` is made for.

## Test

```
cmake -B build -DPython3_EXECUTABLE=/path/to/python   # with pytest, numpy, OpenEXR and Imath, so api.python is registered
ctest --test-dir build -j4
ctest --test-dir build -C Calibration -L calibration -j8   # opt-in, ~40 min: how often each statistical test fails by chance
```

Every statistical test draws its seed from the harness and derives its band from its own standard error and the suite's alpha.
Calibration reports UNSEEDED or UNDERIVED for a test that does not, and fails it.

The `display_gl` tests open a hidden OpenGL window and report Not Run where none can be created.

## Controls

| Input | Action |
| --- | --- |
| `W` / `S` | Fly forward / back |
| `A` / `D` | Fly left / right |
| `Q` / `E` | Fly down / up |
| LMB drag | Orbit about the surface under the view centre |
| `0` | Reset camera to the profile's pose |
| `L` | Cycle viewer LUT: sRGB -> Rec.709 -> Raw |
| `R` / `G` / `B` | Isolate a channel; the active one again clears it |
| `I` | Invert the display |
| `H` | Toggle the HUD |
| `Cmd` + `-` / `=` | Zoom the image out / in by powers of two; it opens fitted to the window and never zooms past fit |
| `Esc` | Quit |

## Python

```
cmake --build build --target pathtracer_c
pip install -e python
```

The package finds the library in the source tree only; for any other install set `PATHTRACER_LIB` to the built `pathtracer_c` library. The library is not bundled, and there is no Windows candidate.

```python
from pathtracer import Renderer

renderer = Renderer("scenes/cornell.json", preset="proxy")  # beauty when omitted
frame = renderer.render(aovs=("beauty", "depth", "normal"), width=128, height=64, samples=8, seed=1)

frame["beauty"]   # (64, 128, 3) float32, linear ACEScg radiance, luminance in cd/m^2
frame["depth"]    # (64, 128, 1) float32, camera-space Z
```

## Roadmap

### Current

| Area | Item | Detail | Complexity | ETA | Status |
| --- | --- | --- | --- | --- | --- |
| Colour | Infrared and ultraviolet | Visible wavelengths only. Transport over 300-1000 nm, IR and UV sensor and false-colour film sensitivities, measured IR reflectance (Gates et al. 1965). A custom UV/IR scene with known reflectances checks it. | high | 3 h | ready |
| Performance | A/B tuning | Small local speed-ups, kept only on a significant paired-run gain on the proxy profile with output inside calibrated bands. | medium | 3 h | ready |

### Queued

| Area | Item | Detail | Complexity | ETA | Status |
| --- | --- | --- | --- | --- | --- |
| Camera | Sensor spectral sensitivity | Renders are ACEScg, not a sensor's response. A chosen camera's published spectral responsivities give its native response, then its IDT to ACEScg (Langlands and Fascione 2020). Worth it only against a real camera to match: the CIE observer is already exact colorimetry. Colour and luminance checked on the macbeth scene against the chart's reference values. | high | 3 h | ready |
| Camera | White balance | No white balance: lights and the HDRI show their true cast. Per-channel gains on the camera-native response so a chosen kelvin (blackbody or daylight) or grey card reads neutral, before the IDT, as in camera. Never edits lights. Checked on the macbeth scene: its grey patches read neutral. Waits for sensor spectral sensitivity. | medium | 1 h | blocked |
| Camera | Synthetic eye | Renders are colour images. Photoreceptor catches from spectral sensitivity templates (Govardovskii et al. 2000) for a chosen eye, output per receptor rather than through ACEScg. | medium | 2 h | ready |
| Material | Measured material library | Only chrome, gold, N-BK7 glass and the ColorChecker are measured. Silver, copper and aluminium from their shipped n,k; water (Hale and Querry 1973 index, Pope and Fry 1997 absorption); kaolinite clay (USGS splib07); marble from calcite reflectance with Jensen et al. (2001) scattering. | medium | 2 h | ready |
| Camera | Lens model | Renders cannot match a calibrated camera. Principal point and pixel aspect on the Kannala and Brandt (2006) model; a measured-lens preset. | medium | 1 h | ready |
| Camera | Sensor noise model | Images are noise-free. Shot, read and dark noise and quantisation per EMVA 1288, from the calibrated luminance. | medium | 2 h | ready |
| Material | Dwivedi subsurface walks | Subsurface walks follow the phase function alone. Dwivedi-guided walks (Křivánek and d'Eon 2014), preferred over diffusion profiles in production (Fascione et al. 2018). | medium | 2 h | ready |
| Volume | Equi-angular sampling | Single scattering near small lights is noisy. Equi-angular distance sampling (Kulla and Fajardo 2011) in a custom distance CDF combining phase function, transmittance and incoming light from a light-tree beam query (Fascione et al. 2018). | medium | 2 h | ready |
| Sampling | Blue noise beyond one dimension | The mask spreads error in one dimension only. Blue-noise-distributed Sobol scrambling and ranking (Heitz et al. 2019). | medium | 2 h | ready |
| Sampling | Splat-time outlier rejection | Fireflies survive into the mean. Density-based outlier rejection when samples are splatted (DeCoro et al. 2010); opt-in because it biases the mean, off for calibrated runs. | low | 1 h | ready |
| Volume | Rayleigh scattering | Media scatter grey with a phase function chosen per medium. Wavelength-dependent Rayleigh extinction (inverse fourth power) and its phase function for clear air and thin hazes, spectral (Fascione et al. 2018). | medium | 1 h | ready |
| Light | IES profiles | Lights emit uniformly. Measured distributions (IES LM-63), importance-sampled from the table and scaled by the existing lumens. | medium | 1 h | ready |
| Camera | Film halation | Highlights have no film halo. Light reflected off the film base re-exposes the red-sensitive layer; a red-weighted glare from the bloom pyramid, picture AOVs only. | medium | 1 h | ready |
| Texture | Metameric pairs | Patches differ in RGB or not at all. Pairs with different spectra but one RGB under a chosen illuminant, for colour constancy and spectral validation. | medium | 1 h | ready |
| Texture | Calibration targets | Only the ColorChecker. Slanted edge (ISO 12233), Siemens star, USAF 1951 resolution chart and a grey step wedge, analytic and footprint-filtered. | medium | 2 h | ready |
| Output | Geometry AOVs | No material, facing or curvature channels. Material ID beside objectID, facing ratio (N.V) and mean curvature (Meyer et al. 2003), with curvature as a shading input. | medium | 2 h | ready |
| Texture | Procedural stimuli | Patterns need image files. Sine and square gratings (orientation, spatial frequency, phase, contrast), Gabor patches, plaids, radial, concentric and log-polar gratings, bars, step edges, checkerboards and Mondrians (Land and McCann 1971), box-filtered by the ray footprint. | medium | 3 h | ready |
| Camera | Event camera output | Log-contrast events for neuromorphic sensors, timed by interpolating between rendered frames (Rebecq et al. 2018). | medium | 3 h | ready |
| Sampling | Many-light sampling | Lights are picked uniformly. Light tree by power and orientation (Conty Estevez and Kulla 2018), every emissive surface in it so emission_color gets NEE; BSDF-aware selection at the first vertex and a screen-space visibility correction cache (Fascione et al. 2018). | high | 6 h | ready |
| Sampling | Path guiding | Paths sample only the BSDF and the lights. Learned directional distributions per region (Müller et al. 2017), unbiased under MIS. | high | 6 h | ready |
| Light | Physical sky | Daylight is an HDRI. Sun and sky from time of day and location, with Rayleigh and Mie scattering, spectral (Wilkie et al. 2021). | high | 4 h | ready |
| Light | Image-textured area lights | Area lights emit one colour. An emission texture over the light, importance-sampled by its luminance, for screens and stimulus displays. | medium | 2 h | ready |
| Volume | Thin emissive media | Emission is collected at scatter points only. Line integration of emission along the ray for flames and glows (Simon et al. 2017). | medium | 2 h | ready |
| Texture | Procedural noise | No noise inputs. White, pink (1/f) and band-pass noise, Perlin (2002), Worley (1996) and fBm, band-limited to the ray footprint. | medium | 2 h | ready |
| Texture | UDIM and Ptex | One image per input. UDIM tiles and per-face Ptex (Burley and Lacewell 2008) through the OIIO texture system. | medium | 2 h | ready |
| Material | Measured BRDFs | Materials are analytic. Tabulated measured BRDFs (Dupuy and Jakob 2018), spectral, importance-sampled from the table. | medium | 2 h | ready |
| Colour | Film grading | No looks. Bleach bypass (retained silver as added neutral density) and three-strip Technicolor dye transfer (Kalmus 1938) in print density, before the view transform. | medium | 2 h | ready |
| Texture | Minification filtering | Unfiltered maps and checker. OIIO mips by ray-cone width with EWA filtering (Greene and Heckbert 1986), LEAN normals (Olano and Baker 2010), analytic box-filtered checker. | high | 3 h | ready |
| Camera | Multi-viewpoint compound eyes | Facets share one viewpoint and angle. Per-facet position, axis and acceptance from measured eye data (Millward et al. 2022). | high | 3 h | ready |
| Camera | Anamorphic and distortion | Anamorphic squeeze and Brown-Conrady distortion (Brown 1966), Newton-inverted for motion vectors and texture filtering. | high | 2 h | ready |
| Material | Measured skin | Skin is RGB subsurface. Spectral layered skin from melanin and haemoglobin (Donner and Jensen 2006) and measured skin spectra. | high | 3 h | ready |
| Light | Polarisation | No polarisation state. Stokes-Mueller transport (Wilkie and Weidlich 2012) with a polarised sky (Wilkie et al. 2004). | high | 6 h | ready |
| Output | Upscaling | Spatial or temporal (Xiao et al. 2020); temporal reads motionVector. Never replaces the converged Beauty. | medium | 3 h | ready |
| Output | Generative channel | Diffusion refinement AOV and latents (Ho et al. 2020; Rombach et al. 2022) from the Denoised AOV. Never replaces Beauty. | high | 6 h | ready |
| Geometry | Scene graph and instancing | Every copy is stored separately. Embree two-level instancing under a transform hierarchy. | high | 5 h | ready |
| Geometry | OpenUSD scenes | Read USD meshes, transforms, instancing, UsdLux lights and UsdShade OpenPBR materials onto the scene graph. Waits for scene graph. | high | 6 h | blocked |
| Geometry | Subdivision surfaces | Smooth surfaces from coarse meshes via OpenSubdiv at load, honouring USD subdivision schemes. Waits for OpenUSD. | high | 2 h | blocked |
| Geometry | Geometry motion vectors | Moving geometry reads as static. Reproject hits by previous instance transforms and USD time-sampled points. Waits for OpenUSD. | medium | 2 h | blocked |
| Geometry | Displacement and dicing | Displacement maps only bump the normal. View-adaptive dicing into micropolygons, displaced and welded watertight once at load into the single BVH (Fascione et al. 2018). Waits for subdivision surfaces. | high | 6 h | blocked |
| Sampling | Manifold next event estimation | Caustics through glass converge slowly. Manifold NEE (Hanika et al. 2015) connects shading points to lights through specular chains, Weta's production caustic method (Fascione et al. 2018). Heavy compute: a prototype with unbiased solution weighting (Zeltner et al. 2020) and MIS against BSDF sampling cost 1.5x per pass on cornell and 25x on a choppy ocean, and lost on time to equal noise; casters must be opted in per object. | high | 4 h | ready |
| Sampling | Bidirectional paths | Light and eye subpaths joined by MIS (Veach and Guibas 1995) over the path vertex graph so caustics converge. Vertex connection and merging and photon methods are not planned: costly, biased, and they fail in hair (Fascione et al. 2018). | high | 8 h | ready |
