# PBR Pathtracer

*A physically-based unidirectional Monte Carlo path tracer with progressive sampling: Embree-accelerated, stochastic BSDF combined with environment-map NEE via MIS, behind a thin OpenGL display/HUD layer.*

![Sample render](sample.png)

## Build

C++20, built with CMake. Currently developed against macOS only.

```
brew install cmake glfw glew glm imath openexr opencolorio openimageio embree open-image-denoise openvdb
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
```

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
| Validation | Curved transmissive epsilons | Removing the offset that stops rays re-hitting curved glass still passes every test. Needs a test against a long reference render. | medium | ready |
| Texture | Minification filtering | Distant textures still shimmer after a bounce. Normal, bump and roughness maps also need filtering that keeps surfaces from shading flat. | high | ready |
| Validation | Untested paths | No tests yet for the Gabor filter's strength, GPU and CPU display matching, screen upload, the HUD histogram, or false tolerance failures. | medium | ready |
| API | Display optics | Bloom and colour fringing exist only in the viewer, so Python output lacks them. Add CPU versions, off by default, that match the viewer. | medium | ready |

### Features

Capability the renderer lacks, most valuable first.

| Area | Item | Detail | Complexity | Status |
| --- | --- | --- | --- | --- |
| Transport | Bounded media | Light scattering inside closed objects such as frosted glass, milk or skin. Needs tracking which object a ray is inside when they nest. | high | ready |
| Sampling | Adaptive sampling | Spend more samples on noisy pixels and fewer on clean ones. The per-pixel noise estimate it needs already exists. | medium | ready |
| Camera | Scene graph and instancing | Every copy of an object is stored separately. Keeping objects in a hierarchy would let copies share memory and enable motion blur. | high | ready |
| Colour | Photometric calibration | Link the renderer's light values to real units (lux, candela), so exposure can be checked against a real light meter. | medium | ready |
| Camera | Lens model | Most fisheye lenses are only approximated and lose accuracy near the edge. Real cameras also need an off-centre lens and non-square pixels. | medium | ready |

### Parked

Refinements, speculative or blocked; not scheduled.

| Area | Item | Detail | Complexity | Status |
| --- | --- | --- | --- | --- |
| BSDF | Smith-exact transmission | Light that passes through rough glass after bouncing between tiny surface facets uses an approximate, physically lopsided shape. | high | ready |
| Texture | Lookup cost | A filtered texture read costs about twice a direct one (12.2 ms against 5.6 ms); filtering after bounces would cut the cost. | medium | ready |
| Texture | Aliasing test asset | A finely noisy material that makes texture shimmer easy to see, to compare before and after filtering. | low | ready |
| Sampling | Blue noise beyond one dimension | Blue noise makes noise look finer, but the mask spreads error well in one dimension only; a multi-dimensional mask would fix it. | medium | ready |
| Colour | HDRI white balance | An HDRI keeps the colour cast of the light it was shot under; white balancing from its colour temperature to daylight would fix it. | medium | ready |
| Colour | Measured-observer cones | Model colour vision from measured human eye response. Needs light tracked by wavelength, so it waits for spectral rendering. | high | blocked |
| Camera | Anamorphic and distortion | Stretched widescreen lenses and real lens distortion. Each needs an inverse to keep motion vectors and texture filtering right. | high | ready |
| Camera | Hexagonal neighbourhoods | Edge detection, the Gabor and DoG filters, denoising and bloom do not work on hexagonal pixels yet; each needs a hexagonal version. | high | ready |
| Camera | Compound-eye optics | Model an insect eye, where each facet sees a soft-edged cone of directions, by blurring each camera ray's direction. | medium | ready |
| Camera | Subdivision surfaces | Smooth curved surfaces from coarse meshes, subdivided at load. Needs a scene-file flag and waits for the scene graph. | high | blocked |
| AOV | Contrast sensitivity | An output showing which detail a human can actually see. Needs brightness in real units, so it waits for photometric calibration. | high | blocked |
