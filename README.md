# PBR Pathtracer

*A physically-based unidirectional Monte Carlo path tracer with progressive sampling: Embree-accelerated, stochastic BSDF combined with environment-map NEE via MIS, behind a thin OpenGL display/HUD layer.*

![Sample render](sample.png)

## Build

C++20, built with CMake. Currently developed against macOS only.

```
brew install cmake glfw glew glm imath openexr opencolorio openimageio embree open-image-denoise
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
| Library | Denoise on the GPU | Denoised runs Open Image Denoise’s CPU device on the render thread, once per published pass: 8 to 15 ms at 256 × 128 with the shipped `fast`, unprefiltered setting, 122 to 139 ms at `high` with prefiltered albedo and normal. Its Metal device runs on Apple-silicon GPUs and is untried here; moving the call off the render thread would also keep the HUD responsive at `high`. The CPU device’s thread count has no effect on Apple silicon, where the convolutions run through BNNS. | medium | ready |
| Library | OpenVDB | Load VDB density grids as heterogeneous media, each bounded by its active-voxel box as a medium interface in the BVH. Absorption only first, which the integrator already carries: transmittance by ratio tracking against a majorant from the grid’s tile maxima, next to the closed-form Beer-Lambert segment for homogeneous media. Scattering and emission from the grid need in-scattering and phase functions, which stay blocked. NanoVDB is the read-only form for a GPU backend. | high | ready |
| Validation | Grazing transmission | The transmissive energy test scores normal incidence only. Binned radially at roughness 0.2 it reads 0.9450 at 90% of the silhouette radius and 0.9020 at 95%, against 1.0000 at the centre, non-monotonic in roughness and absent by 0.7, measured before the OpenPBR interface. Instrument and re-measure first, then fix any loss. It gates authoring rough glass. | high | ready |
| Validation | Curved transmissive epsilons | Reverting either curvature-scaled epsilon leaves the suite green. Needs rough glass against an opaque backdrop, which has no closed form, so a converged-reference or two-estimator invariant. | medium | ready |
| Validation | Untested paths | Gabor response magnitude, GPU and CPU display parity, the display texture upload (the suite has no GL context), the histogram binning (extract it so the CPU can test it), and a standing false-rejection rate for derived tolerance bands. | medium | ready |
| BSDF | Smith-exact transmission | Refraction’s shadowing is height-correlated Smith, `B(1 + Λ_o, 1 + Λ_i)` (Heitz 2014), and the share of an interface’s multiple scattering that leaves reflected comes from Smith random walks (Heitz et al. 2016). Both are exact in energy. The transmitted multiple-scattering lobe is still a `(1 − escape) cos` shape, which cannot follow its collapse toward −wo near η = 1, and is not reciprocal, so transmission reciprocity is asserted for single scatter only. A directional reference from the same walks would replace it. | high | ready |
| BSDF | Normal-map third bounce | Normal and bump maps shade through the microfacet-based model (Schüßler et al. 2017) to its second order: light the mapped facet sends into the groove’s wall that needs a third bounce to escape is dropped. The bumped stump is 8 to 9% darker for it; a white diffuse facet tilted 17° keeps 84 to 93% of its albedo, and at 46° about half. The paper’s full random walk recovers it, at the cost of stochastic evaluation in NEE. | medium | decision |
| BSDF | Interpolated-normal residual | The microsurface is built on the interpolated vertex normal, so a sample between it and the flat face normal still ends the path at the geometric side rule, which loses energy on coarse meshes. The paper leaves this open (its Fig. 18). | medium | ready |
| Texture | Minification filtering | Primary hits ship: base colour, specular and a shown sky are filtered over the camera differential’s stratum. Secondary vertices still read level 0 at a point, so the stump’s 4K maps keep about 2000 level-0 tiles each resident; they need differentials propagated through each BSDF bounce. Normal, bump and roughness need appearance-preserving filtering first: prefiltered normals shorten and shade flatter (Toksvig 2005), 13% brighter on the minified stump, so their lost slope variance must move into roughness (Olano and Baker 2010). | high | ready |
| Texture | Lookup cost | A TextureSystem lookup (handle, tile hash, micro-cache, filter) replaces a direct bilinear over a half array: steady-state passes on the stump are 12.2 ms against 5.6 at 256 × 128, and cornell’s environment adds about 19%. Load falls from 493 to 32 ms and the cache bounds memory. Secondary differentials cut the level-0 traffic; a per-thread handle cache gained nothing. | medium | ready |
| Texture | Aliasing test asset | A high-frequency noise material that exposes minification aliasing before and after. | low | ready |
| Sampling | Adaptive sampling | Variance-driven per-pixel sample budget. The error estimate it needs already ships (the SNR’s replicate means). Foveal density is the same mechanism with an eccentricity weight. | medium | ready |
| Sampling | Blue noise beyond one dimension | The mask is a scalar array, so a pixel's shift lies on a line. An annealed d-dimensional matrix (Georgiev and Fajardo 2016) changes the baked table and its lookup, not the sampler. | medium | ready |
| Colour | Out-of-gamut metals | Gold’s F0 red is 1.038 in linear Rec.709, outside `base_color`’s unit range, so the F82-tint metal cannot represent it and clamping would silently change hue; copper, silver, nickel, iron and titanium fit. Needs a gamut mapping or a wider rendering space such as ACEScg with the OCIO scene space to match. | high | decision |
| Colour | Photometric calibration | Tie radiometric output to lux and candela through the CIE ȳ curve, so EV100 can be checked against a light meter. | medium | ready |
| Colour | Kelvin light temperature | Author lights by colour temperature on the Planckian locus. Validated against the ColorChecker scene. | low | ready |
| Colour | Measured-observer cones | Hunt-Pointer-Estevez is the only cone basis exact on the CIE 1931 observer, so `L + M` is not V(λ) and the opponent plane is not strictly isoluminant. Closing that needs the Judd-Vos modified matching functions and a spectral description of the primaries, so it lands with spectral transport. Chromatic contrast sensitivity (Mullen 1985) goes with it. | high | blocked |
| Camera | Lens model | Equisolid, stereographic and orthographic projections are reachable only as truncated degree-9 polynomials, with error growing toward the circle edge; equidistant is exact at k = 0. Each family is a closed-form θ(r) with no root-finding. Also a calibrated principal point and non-square pixels: one focal length means fx = fy and a centred circle, so only a calibration's radial geometry transfers. | medium | ready |
| Camera | Anamorphic and distortion | Anamorphic squeeze scales the horizontal half-width in the view basis; it needs the lens model's non-square pixels, carried through the pinhole matrix and the G-buffer's pixel Jacobian. Brown-Conrady radial and tangential distortion (Brown 1966) is a sensor-plane remap before projection, and an STMap is the same remap read from a texture. Each needs its inverse for motion vectors and its Jacobian for ray differentials. | high | ready |
| Camera | Overscan | Trace a margin beyond the frame, 5% of width and height on each edge (about 21% more pixels), with framing unchanged inside it. Motion vectors are already defined off-frame, since both views project the point; what is missing is history: a temporal consumer such as upscaling reprojecting outside the previous frame finds no pixel. The margin also gives the neighbour-reading filters real neighbours at the border, not clamped ones. EXR stores it as a data window larger than the display window and the viewer crops it; the C and Python output must choose cropped or full. | medium | ready |
| Camera | Depth of field | Thin-lens sampling in the primary ray plus a focus distance. Cheaper after adaptive sampling. | low | ready |
| Camera | Hexagonal pixels | Sensor and compound-eye emulation on a hexagonal lattice, output natively. Pixels are square end to end: jitter and sampler keying, a film filter sampled per axis, row-major buffers, the EXR and `(H, W, C)` array contract, and the neighbour-reading filters. Needs an offset-row lattice in the rectangular buffer and a radially symmetric filter over six neighbours (Mersereau 1979). Open: an output contract that carries the row offset. | high | decision |
| Camera | Scene graph and instancing | Geometry is baked to world space and duplicated per node. A scene graph would unlock motion blur and real instancing. It would also carry subdivision surfaces: OpenSubdiv evaluates the limit surface of a control cage, but glTF has no subdivision representation and the loader takes triangles only, so a mesh needs a scene-JSON flag and tessellation before the BVH build. Embree has native subdivision geometry. | high | decision |
| Camera | Motion blur | Needs scene animation and multi-timestep geometry; neither exists. | high | blocked |
| Camera | Polarising filters | Need polarised transport (Chandrasekhar 1960). | high | blocked |
| AOV | Shadow sentinel | Shadow stays 1 when no light sample is taken, the same as fully shadowed. It needs a third state every consumer can read. | medium | decision |
| AOV | Depth auto-range | A serial per-texel scan on the render thread. Publish the range from the G-buffer pass instead. | low | ready |
| AOV | Contrast sensitivity | Blocked on photometric calibration: the model needs luminance in cd/m². Use Barten 1999 (physical parameters), not Mannos-Sakrison or Daly (fitted coefficients, no luminance dependence). Weight octave-band contrast on the shared scale space (Mantiuk et al. 2006; HDR-VDP-2), not an FFT multiply, which needs padding decisions and ignores luminance. Contrast is Peli 1990, `c_k = B_k / L_<k`, exposure-invariant, over Michelson or a log difference. Luminance only. | high | blocked |
| AOV | Spherical harmonics AOV | No SH infrastructure exists. Needs a use-case write-up before it can be sized. | high | decision |
| API | Python tests in ctest | The pytest suite runs outside ctest, so the Python binding is not gated with the rest of the checks. The C ABI itself is covered by `c_abi_validate`, which links only the shared library. | low | ready |
| API | Packaging | The Python package finds the library only in the source tree, so a non-editable install needs `PATHTRACER_LIB`. The library is not bundled and there is no Windows candidate. | medium | decision |
