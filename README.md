# PBR Pathtracer

*A physically-based unidirectional Monte Carlo path tracer with progressive sampling: Embree-accelerated, stochastic BSDF combined with environment and area-light NEE via MIS, behind a thin OpenGL display/HUD layer.*

![Sample render](sample.png)

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
./build/pathtracer [-scene path/to/scene.json] [-stats] [-bench log.jsonl] [-bench-aovs beauty,normal,...] [-size WxH] [-max-samples N]
```

## Test

```
cmake -B build -DPython3_EXECUTABLE=/path/to/python   # with pytest, numpy, OpenEXR and Imath, so api.python is registered
ctest --test-dir build -j4
ctest --test-dir build -C Calibration -L calibration -j8   # opt-in, hours: how often each statistical test fails by chance
```

The `display_gl` tests open a hidden OpenGL window and report Not Run where none can be created.

## Controls

| Input | Action |
| --- | --- |
| `W` / `S` | Fly forward / back |
| `A` / `D` | Fly left / right |
| `Q` / `E` | Fly down / up |
| LMB drag | Orbit about the surface under the view centre |
| `0` | Reset camera to the `profile.json` pose |
| `L` | Cycle viewer LUT: sRGB -> Rec.709 -> Raw |
| `R` / `G` / `B` | Isolate a channel; the active one again clears it |
| `I` | Invert the display |
| `H` | Toggle the HUD |
| `Esc` | Quit |

## Python

```
cmake --build build --target pathtracer_c
pip install -e python
```

The package finds the library in the source tree only; for any other install set `PATHTRACER_LIB` to the built `pathtracer_c` library. The library is not bundled, and there is no Windows candidate.

```python
from pathtracer import Renderer

renderer = Renderer("scenes/cornell.json")
frame = renderer.render(aovs=("beauty", "depth", "normal"), width=128, height=64, samples=8, seed=1)

frame["beauty"]   # (64, 128, 3) float32, linear ACEScg radiance
frame["depth"]    # (64, 128, 1) float32, camera-space Z
```

## Roadmap

### Problems

Output that is wrong, lossy or unverified, most serious first.

| Area | Item | Detail | Complexity | Status |
| --- | --- | --- | --- | --- |
| Texture | Minification filtering | Distant textures still shimmer after a bounce, and the random checker is never filtered. Normal, bump and roughness maps also need filtering that keeps surfaces from shading flat. | high | ready |
| BSDF | Normal-map noise | On normal-mapped glossy metal, glass and coated surfaces, light that bounces between facets more than once is estimated with infinite variance, so its fireflies fade slower than usual. Rough glass also fails its sampling-against-evaluation calibration (`results/untested_paths/bias`). | high | ready |
| Validation | Hand-set test tolerances | 33 random tests still use fixed seeds or hand-set tolerances instead of bands from the suite's error rate; `ctest -C Calibration -L calibration` lists them as UNDERIVED. | medium | ready |

### Features

Capability the renderer lacks, most valuable first.

| Area | Item | Detail | Complexity | Status |
| --- | --- | --- | --- | --- |
| Sampling | Many-light sampling | Lights are picked uniformly, so a small dim light gets as many shadow rays as a large bright one. A light tree would pick by power and orientation, and ReSTIR would reuse picks across pixels. | high | ready |
| Sampling | Adaptive sampling | Spend more samples on noisy pixels and fewer on clean ones. The per-pixel noise estimate it needs already exists. | medium | ready |
| Camera | Scene graph and instancing | Every copy of an object is stored separately. Keeping objects in a hierarchy would let copies share memory and enable motion blur. | high | ready |
| Transport | Volume emission | Media that glow, such as fire or a lit gas, gathered where a ray collides inside them. OpenPBR's surface has no such input, so it would be a volume material key. | medium | ready |
| Colour | Photometric calibration | Link the renderer's light values to real units (lux, candela), so exposure can be checked against a real light meter. | medium | ready |
| Camera | Lens model | Most fisheye lenses are only approximated and lose accuracy near the edge. Real cameras also need an off-centre lens and non-square pixels. | medium | ready |

### Parked

Refinements, speculative or blocked; not scheduled.

| Area | Item | Detail | Complexity | Status |
| --- | --- | --- | --- | --- |
| BSDF | Smith-exact transmission | Light that passes through rough glass after bouncing between tiny surface facets uses an approximate, physically lopsided shape. | high | ready |
| Texture | Lookup cost | A filtered texture read costs about twice a direct one (12.2 ms against 5.6 ms); filtering after bounces would cut the cost. | medium | ready |
| Sampling | Blue noise beyond one dimension | Blue noise makes noise look finer, but the mask spreads error well in one dimension only; a multi-dimensional mask would fix it. | medium | ready |
| Colour | Measured-observer cones | Model colour vision from measured human eye response. Needs light tracked by wavelength, so it waits for spectral rendering. | high | blocked |
| Camera | Anamorphic and distortion | Stretched widescreen lenses and real lens distortion. Each needs an inverse to keep motion vectors and texture filtering right. | high | ready |
| Camera | Multi-viewpoint compound eyes | Facets share one viewpoint and one acceptance angle. CompoundRay (Millward et al. 2022) places each on its own eye surface with its own axis and angle, from measured eye data. | high | ready |
| Camera | Hexagonal denoising | Denoised stays refused on hexagonal pixels: OIDN's network is trained on square images, so it needs one trained on offset rows. | high | blocked |
| Camera | Subdivision surfaces | Smooth curved surfaces from coarse meshes, subdivided at load. Needs a scene-file flag and waits for the scene graph. | high | blocked |
| AOV | Contrast sensitivity | An output showing which detail a human can actually see. Needs brightness in real units, so it waits for photometric calibration. | high | blocked |
