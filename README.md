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

frame["beauty"]   # (64, 128, 3) float32, linear Rec.709 radiance
frame["depth"]    # (64, 128, 1) float32, camera-space Z
```

## Roadmap

| Area | Item | Detail | Complexity | Status |
| --- | --- | --- | --- | --- |
| Transport | Volume emission | Emission and temperature grids as volume radiance, gathered at real collisions by the same spectral tracking, with blackbody colour from temperature. The absorbed event becomes an emission vertex rather than a termination. | medium | ready |
| Transport | Bounded media | OpenPBR's `transmission_scatter` and `subsurface_radius` as homogeneous media inside closed surfaces, tracked by the same free-flight and ratio-tracking code against a constant majorant. Needs a nested medium stack in place of today's single level. | high | ready |
| Validation | Grazing transmission | The transmissive energy test scores normal incidence only. Binned radially at roughness 0.2 it reads 0.9450 at 90% of the silhouette radius and 0.9020 at 95%, against 1.0000 at the centre, non-monotonic in roughness and absent by 0.7, measured before the OpenPBR interface. Instrument and re-measure first, then fix any loss. It gates authoring rough glass. | high | ready |
| Validation | Curved transmissive epsilons | Reverting either curvature-scaled epsilon leaves the suite green. Needs rough glass against an opaque backdrop, which has no closed form, so a converged-reference or two-estimator invariant. | medium | ready |
| Validation | Untested paths | Gabor response magnitude, GPU and CPU display parity, the display texture upload (the suite has no GL context), the histogram binning (extract it so the CPU can test it), and a standing false-rejection rate for derived tolerance bands. | medium | ready |
| BSDF | Smith-exact transmission | Refraction’s shadowing is height-correlated Smith, `B(1 + Λ_o, 1 + Λ_i)` (Heitz 2014), and the share of an interface’s multiple scattering that leaves reflected comes from Smith random walks (Heitz et al. 2016). Both are exact in energy. The transmitted multiple-scattering lobe is still a `(1 − escape) cos` shape, which cannot follow its collapse toward −wo near η = 1, and is not reciprocal, so transmission reciprocity is asserted for single scatter only. A directional reference from the same walks would replace it. | high | ready |
| BSDF | Interpolated-normal residual | The microsurface is built on the interpolated vertex normal, so a sample between it and the flat face normal still ends the path at the geometric side rule, which loses energy on coarse meshes. The paper leaves this open (its Fig. 18). | medium | ready |
| Texture | Minification filtering | Primary hits ship: base colour, specular and a shown sky are filtered over the camera differential’s stratum. Secondary vertices still read level 0 at a point, so the stump’s 4K maps keep about 2000 level-0 tiles each resident; they need differentials propagated through each BSDF bounce. Normal, bump and roughness need appearance-preserving filtering first: prefiltered normals shorten and shade flatter (Toksvig 2005), 13% brighter on the minified stump, so their lost slope variance must move into roughness (Olano and Baker 2010). | high | ready |
| Texture | Lookup cost | A TextureSystem lookup (handle, tile hash, micro-cache, filter) replaces a direct bilinear over a half array: steady-state passes on the stump are 12.2 ms against 5.6 at 256 × 128, and cornell’s environment adds about 19%. Load falls from 493 to 32 ms and the cache bounds memory. Secondary differentials cut the level-0 traffic; a per-thread handle cache gained nothing. | medium | ready |
| Texture | Aliasing test asset | A high-frequency noise material that exposes minification aliasing before and after. | low | ready |
| Sampling | Adaptive sampling | Variance-driven per-pixel sample budget. The error estimate it needs already ships (the SNR’s replicate means). Foveal density is the same mechanism with an eccentricity weight. | medium | ready |
| Sampling | Blue noise beyond one dimension | The mask is a scalar array, so a pixel's shift lies on a line. An annealed d-dimensional matrix (Georgiev and Fajardo 2016) changes the baked table and its lookup, not the sampler. | medium | ready |
| Colour | Out-of-gamut metals | Gold’s F0 red is 1.038 in linear Rec.709, outside `base_color`’s unit range, so the F82-tint metal cannot represent it and clamping would silently change hue; copper, silver, nickel, iron and titanium fit. Render in ACEScg with the OCIO scene space to match. | high | ready |
| Colour | Photometric calibration | Tie radiometric output to lux and candela through the CIE ȳ curve, so EV100 can be checked against a light meter. | medium | ready |
| Colour | HDRI white balance | Light temperature sets quad-light emission only; an HDRI keeps the colour it was measured under. Balancing it is chromatic adaptation, not a tint: a von Kries transform in CAT16's cone space (Li et al. 2017) from a Planckian white at the authored kelvin to D65, one 3 × 3 matrix on environment radiance beside its exposure. | medium | ready |
| Colour | Measured-observer cones | Hunt-Pointer-Estevez is the only cone basis exact on the CIE 1931 observer, so `L + M` is not V(λ) and the opponent plane is not strictly isoluminant. Closing that needs the Judd-Vos modified matching functions and a spectral description of the primaries, so it lands with spectral transport. Chromatic contrast sensitivity (Mullen 1985) goes with it. | high | blocked |
| Camera | Lens model | Equisolid, stereographic and orthographic projections are reachable only as truncated degree-9 polynomials, with error growing toward the circle edge; equidistant is exact at k = 0. Each family is a closed-form θ(r) with no root-finding. Also a calibrated principal point and non-square pixels: one focal length means fx = fy and a centred circle, so only a calibration's radial geometry transfers. | medium | ready |
| Camera | Anamorphic and distortion | Anamorphic squeeze scales the horizontal half-width in the view basis; it needs the lens model's non-square pixels, carried through the pinhole matrix and the G-buffer's pixel Jacobian. Brown-Conrady radial and tangential distortion (Brown 1966) is a sensor-plane remap before projection, and an STMap is the same remap read from a texture. Each needs its inverse for motion vectors and its Jacobian for ray differentials. | high | ready |
| Camera | Hexagonal neighbourhoods | The hexagonal lattice renders every per-pixel AOV, but refuses Sobel, Gabor, DoG and Denoised and draws no bloom: square-grid stencils, a square-grid pyramid and a network trained on square images. Each needs its hexagonal form, a six-neighbour gradient and Gaussians and Morlet bank sampled on the lattice (Middleton and Sivaswamy 2005), and so does the error spectrum's square FFT. Denoising needs a resample to square and back, or a network trained on hexagonal data. | high | ready |
| Camera | Compound-eye optics | The hexagonal lattice places ommatidia but not their optics: each one's acceptance angle, a Gaussian angular sensitivity (Land 1997), belongs to the lens model as a per-cell angular filter on the primary ray. | medium | ready |
| Camera | Scene graph and instancing | Geometry is baked to world space and duplicated per node. A scene graph would unlock motion blur and real instancing. | high | ready |
| Camera | Subdivision surfaces | OpenSubdiv evaluates the limit surface of a control cage, but glTF has no subdivision representation and the loader takes triangles only, so a mesh needs a scene-JSON flag and tessellation before the BVH build. Embree has native subdivision geometry. Follows the scene graph. | high | blocked |
| AOV | Shadow sentinel | Shadow stays 1 when no light sample is taken, the same as fully shadowed. A separate valid-mask channel marks where a light sample was taken, so every consumer can read it. | medium | ready |
| AOV | Contrast sensitivity | Blocked on photometric calibration: the model needs luminance in cd/m². Use Barten 1999 (physical parameters), not Mannos-Sakrison or Daly (fitted coefficients, no luminance dependence). Weight octave-band contrast on the shared scale space (Mantiuk et al. 2006; HDR-VDP-2), not an FFT multiply, which needs padding decisions and ignores luminance. Contrast is Peli 1990, `c_k = B_k / L_<k`, exposure-invariant, over Michelson or a log difference. Luminance only. | high | blocked |
| API | Display optics | Bloom and chromatic aberration run only in the viewer’s display shaders, so Python and `render_beauty` output carry neither. Each needs a CPU form applied to scene-linear Beauty before exposure, as the viewer applies it: the equal-weight Gaussian pyramid (Jimenez 2014), energy-conserving, and the radial per-channel offset. Exposed as `render` arguments that leave the output bit-identical at 0, with a GPU parity check. Bloom stays refused on the hexagonal lattice. | medium | ready |
