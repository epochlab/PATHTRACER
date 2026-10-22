"""Contract tests for the Python binding.

The strongest check here is ``test_beauty_matches_render_beauty_cli``: the renderer is deterministic by construction
(``render_beauty --assert-deterministic`` gates it), so Python's Beauty must be EXACTLY equal to the CLI's EXR at the
same scene, resolution, seed and pass count. Any tolerance there would hide the bugs worth catching.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from pathtracer import (AOVS, LENS_PROJECTIONS, PIXEL_LATTICES, Renderer, aov_channels, aov_needs_samples, display_encode,
                        lattice_preview, meter_ev100)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE = "scenes/cornell.json"


def _rotation_xyz(degrees: tuple[float, float, float]) -> np.ndarray:
    """Rz @ Ry @ Rx in float64, written from the axis rotations themselves as the docstring states them."""
    x, y, z = (math.radians(angle) for angle in degrees)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(x), -math.sin(x)], [0.0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0.0], [math.sin(z), math.cos(z), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


@pytest.fixture(scope="module")
def renderer() -> Renderer:
    with Renderer(SCENE) as instance:
        yield instance


def test_aov_table_is_populated() -> None:
    assert len(AOVS) == 39
    names = ("beauty", "depth", "lookahead", "normal", "sobel", "luminance", "gabor", "hsv", "dog", "colourOpponent", "contrastSensitivity",
             "motionVector", "snr", "sampleCount", "directVolume", "indirectVolume", "directSubsurface", "indirectSubsurface")
    for name in names:
        assert name in AOVS


def test_aov_names_are_exact() -> None:
    assert aov_channels("bounceCount") == 1
    for name in ("bounce-count", "Bounce Count", "BounceCount", "HSV"):
        with pytest.raises(ValueError, match="unknown AOV"):
            aov_channels(name)


def test_shapes_and_dtypes_match_the_declared_channels(renderer: Renderer) -> None:
    names = ("beauty", "depth", "normal", "uv")
    frame = renderer.render(aovs=names, width=32, height=24, samples=2)
    for name in names:
        array = frame[name]
        assert array.shape == (24, 32, aov_channels(name))
        assert array.dtype == np.float32
        assert array.flags["C_CONTIGUOUS"], "torch.from_numpy needs a contiguous buffer to share memory"


def test_same_seed_reproduces_exactly(renderer: Renderer) -> None:
    first = renderer.render(aovs=("beauty",), width=32, height=24, samples=4, seed=3)["beauty"]
    second = renderer.render(aovs=("beauty",), width=32, height=24, samples=4, seed=3)["beauty"]
    assert np.array_equal(first, second)


def test_different_seeds_differ(renderer: Renderer) -> None:
    first = renderer.render(aovs=("beauty",), width=32, height=24, samples=4, seed=3)["beauty"]
    second = renderer.render(aovs=("beauty",), width=32, height=24, samples=4, seed=9)["beauty"]
    assert not np.array_equal(first, second)


def test_depth_is_positive_inside_the_box_and_far_on_the_background(renderer: Renderer) -> None:
    frame = renderer.render(aovs=("depth",), width=64, height=48)
    depth = frame["depth"][..., 0]
    assert np.isfinite(depth).all()
    assert (depth >= 0.0).all()
    # A hit's t is at least the near clip, so depth > 0 is exactly the primary-hit mask.
    hit = depth > 0.0
    assert hit.any(), "the default camera should see geometry"
    # The Cornell box encloses the camera, so every primary ray hits something well inside the far clip.
    assert depth[hit].max() < renderer.default_camera.far_clip


def test_lookahead_is_depth_on_the_profile_horizon(renderer: Renderer) -> None:
    """The lane's whole contract: clamp(1 - Z/lookaheadDistance, 0, 1) against the Depth lane of the same render."""
    horizon = json.loads((REPO_ROOT / "assets" / "config" / "profile.json").read_text())["pathTracer"]["lookaheadDistance"]
    frame = renderer.render(aovs=("lookahead", "depth"), width=64, height=48)
    lookahead, depth = frame["lookahead"][..., 0], frame["depth"][..., 0]
    hit = depth > 0.0
    assert hit.any(), "the default camera should see geometry"
    expected = np.clip(1.0 - depth[hit] / horizon, 0.0, 1.0)
    assert np.allclose(lookahead[hit], expected, rtol=0.0, atol=1e-6)
    # A primary miss reads 0, the same value geometry beyond the horizon reads -- depth is what separates the two.
    assert (lookahead[~hit] == 0.0).all() if (~hit).any() else True
    assert ((lookahead >= 0.0) & (lookahead <= 1.0)).all()


def test_normals_are_unit_length_where_geometry_was_hit(renderer: Renderer) -> None:
    frame = renderer.render(aovs=("normal", "depth"), width=64, height=48)
    hit = frame["depth"][..., 0] > 0.0
    lengths = np.linalg.norm(frame["normal"][hit], axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_gbuffer_aovs_need_no_samples() -> None:
    assert aov_needs_samples("beauty")
    assert aov_needs_samples("sobel"), "filters read Beauty, so they do need light transport"
    assert not aov_needs_samples("depth")
    assert not aov_needs_samples("normal")


def test_gbuffer_only_request_skips_the_path_tracer(renderer: Renderer) -> None:
    """A depth+normal request must not path-trace, which is visible as an order-of-magnitude time difference."""
    size = {"width": 96, "height": 96}

    start = time.perf_counter()
    renderer.render(aovs=("depth", "normal"), samples=64, **size)
    gbuffer_seconds = time.perf_counter() - start

    start = time.perf_counter()
    renderer.render(aovs=("depth", "normal", "beauty"), samples=64, **size)
    with_beauty_seconds = time.perf_counter() - start

    assert gbuffer_seconds * 10 < with_beauty_seconds, (
        f"G-buffer-only took {gbuffer_seconds:.4f}s against {with_beauty_seconds:.4f}s with Beauty; "
        "the path tracer appears to be running for a request that needs no light transport"
    )


def test_every_aov_renders(renderer: Renderer) -> None:
    camera = dataclasses.replace(renderer.default_camera, pixel_lattice="square")
    for name in AOVS:
        array = renderer.render(aovs=(name,), camera=camera, width=32, height=32, samples=2)[name]
        assert array.shape == (32, 32, aov_channels(name))
        # SNR's +inf is a positive signal every replicate agreed on exactly, its true value; any other non-finite texel is a defect.
        admitted = np.isposinf(array) if name == "snr" else np.zeros_like(array, dtype=bool)
        assert (np.isfinite(array) | admitted).all(), f"{name} produced a non-finite value"


def test_sample_count_is_the_cap_or_the_checkpoint_a_texel_stopped_at(renderer: Renderer) -> None:
    profile = json.loads((REPO_ROOT / "assets" / "config" / "profile.json").read_text())["pathTracer"]
    replicates, adaptive, samples = profile["replicates"], profile["adaptive"], 16
    counts = renderer.render(aovs=("sampleCount",), width=32, height=32, samples=samples, seed=3)["sampleCount"]
    if adaptive["threshold"] == 0:
        assert (counts == samples).all()
        return
    # A stopped texel holds whole rounds whose per-replicate count is a power of two, from minSamples on.
    checkpoints = [replicates * 2**j for j in range(samples.bit_length()) if adaptive["minSamples"] <= replicates * 2**j < samples]
    assert np.isin(counts, [*checkpoints, samples]).all()


def test_multi_aov_request_matches_single_requests(renderer: Renderer) -> None:
    names = ("beauty", "depth", "normal", "sobel")
    size = {"width": 32, "height": 32, "samples": 4, "seed": 5}
    together = renderer.render(aovs=names, **size)
    for name in names:
        alone = renderer.render(aovs=(name,), **size)[name]
        assert np.array_equal(alone, together[name]), f"{name} changed when requested alongside others"


def test_camera_override_changes_the_image(renderer: Renderer) -> None:
    default = renderer.default_camera
    moved = dataclasses.replace(default, rotation=(0.0, 25.0, 0.0))
    size = {"width": 32, "height": 32, "samples": 2}
    assert not np.array_equal(
        renderer.render(aovs=("depth",), **size)["depth"],
        renderer.render(aovs=("depth",), camera=moved, **size)["depth"],
    )


def test_motion_vector_follows_the_previous_camera(renderer: Renderer) -> None:
    """Zero without a previous view; a turn to the left (positive y) since the previous view moves every pixel right."""
    default = renderer.default_camera
    size = {"width": 32, "height": 24}
    still = renderer.render(aovs=("motionVector",), **size)["motionVector"]
    assert still.shape == (24, 32, 2)
    assert (still == 0.0).all(), "no previous camera did not read exactly zero motion"
    previous = dataclasses.replace(default, rotation=(0.0, -0.5, 0.0))
    turned = renderer.render(aovs=("motionVector",), previous_camera=previous, **size)["motionVector"]
    assert (turned[..., 0] > 0.0).all(), "a turn to the left did not move every pixel to the right"


def test_fisheye_lens_changes_the_image(renderer: Renderer) -> None:
    default = renderer.default_camera
    # Short focal length, or the 180-degree image circle dwarfs the gate and the frame is the central few degrees of the fisheye.
    fisheye = dataclasses.replace(default, lens="fisheye_polynomial", focal_length_mm=10.0)
    size = {"width": 32, "height": 32, "samples": 2, "seed": 5}
    assert not np.array_equal(
        renderer.render(aovs=("beauty",), **size)["beauty"],
        renderer.render(aovs=("beauty",), camera=fisheye, **size)["beauty"],
    )


def test_fisheye_gbuffer_depth_is_the_ray_distance(renderer: Renderer) -> None:
    """Depth is |worldPos - eye| under every lens: the one depth a fisheye past 90 degrees or the lat-long still defines."""
    for lens in LENS_PROJECTIONS:
        camera = dataclasses.replace(renderer.default_camera, lens=lens, focal_length_mm=10.0)
        frame = renderer.render(aovs=("depth", "worldPos"), camera=camera, width=48, height=32)
        hit = frame["depth"][..., 0] > 0.0
        assert hit.any(), f"{lens}: the camera should see geometry"
        distance = np.linalg.norm(frame["worldPos"][hit] - np.asarray(camera.position, dtype=np.float32), axis=-1)
        assert np.allclose(frame["depth"][..., 0][hit], distance, rtol=1e-5, atol=0.0), lens


def test_omnidirectional_pixels_are_the_documented_longitude_and_latitude(renderer: Renderer) -> None:
    """Each hit's direction from the eye lands at its own pixel centre under Camera's lat-long formula, so the docstring is the code."""
    # Pitched, yawed and rolled at once, so the frame is the documented Rz @ Ry @ Rx and not merely the identity.
    camera = dataclasses.replace(renderer.default_camera, lens="omnidirectional", rotation=(20.0, 30.0, 40.0), pixel_lattice="square")
    width, height = 64, 32
    frame = renderer.render(aovs=("worldPos", "depth"), camera=camera, width=width, height=height)
    hit = frame["depth"][..., 0] > 0.0
    assert hit.any(), "the lat-long should see the scene"
    rotation = _rotation_xyz(camera.rotation)
    right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
    view = frame["worldPos"][hit].astype(np.float64) - np.asarray(camera.position)
    longitude = np.arctan2(view @ right, view @ forward)
    colatitude = np.arctan2(np.hypot(view @ right, view @ forward), view @ up)
    rows, columns = np.nonzero(hit)
    # 16 ulps of the ray's trig arguments (up to pi), plus float32 worldPos and eye rounding seen over |view|; longitude divides by sin.
    magnitude = np.abs(frame["worldPos"][hit]).max(axis=-1) + max(map(abs, camera.position))
    angular = 16.0 * np.finfo(np.float32).eps * (math.pi + magnitude / np.linalg.norm(view, axis=-1))
    assert (np.abs(longitude - math.pi * (2.0 * (columns + 0.5) / width - 1.0)) <= angular / np.sin(colatitude)).all()
    assert (np.abs(colatitude - math.pi * (rows + 0.5) / height) <= angular).all()


def test_hexagonal_pixels_are_the_documented_centres(renderer: Renderer) -> None:
    """Through the lat-long, each hit's longitude and colatitude are Camera's hexagonal centre formula over the film it documents."""
    camera = dataclasses.replace(renderer.default_camera, lens="omnidirectional", rotation=(20.0, 30.0, 40.0), pixel_lattice="hexagonal")
    width, height = 64, 32
    frame = renderer.render(aovs=("worldPos", "depth"), camera=camera, width=width, height=height)
    hit = frame["depth"][..., 0] > 0.0
    assert hit.any(), "the lat-long should see the scene"
    rotation = _rotation_xyz(camera.rotation)
    right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
    view = frame["worldPos"][hit].astype(np.float64) - np.asarray(camera.position)
    longitude = np.arctan2(view @ right, view @ forward)
    colatitude = np.arctan2(np.hypot(view @ right, view @ forward), view @ up)
    rows, columns = np.nonzero(hit)
    row_pitch = math.sqrt(3.0) / 2.0
    centre_x = columns + 0.25 + 0.5 * (rows % 2)
    centre_y = (rows + 0.5) * row_pitch
    magnitude = np.abs(frame["worldPos"][hit]).max(axis=-1) + max(map(abs, camera.position))
    angular = 16.0 * np.finfo(np.float32).eps * (math.pi + magnitude / np.linalg.norm(view, axis=-1))
    assert (np.abs(longitude - math.pi * (2.0 * centre_x / width - 1.0)) <= angular / np.sin(colatitude)).all()
    assert (np.abs(colatitude - math.pi * centre_y / (height * row_pitch)) <= angular).all()


def test_hexagonal_lattice_refuses_denoised_by_name(renderer: Renderer) -> None:
    camera = dataclasses.replace(renderer.default_camera, pixel_lattice="hexagonal")
    with pytest.raises(RuntimeError, match="denoised runs a network trained on square images"):
        renderer.render(aovs=("denoised",), camera=camera, width=8, height=8, samples=1)


def test_pixel_lattice_round_trips_and_rejects_unknown_names(renderer: Renderer) -> None:
    authored = json.loads((REPO_ROOT / "assets" / "config" / "profile.json").read_text())["camera"]["pixelLattice"]
    assert renderer.default_camera.pixel_lattice == authored
    assert PIXEL_LATTICES == ("square", "hexagonal")
    with pytest.raises(ValueError, match="triangular"):
        renderer.render(aovs=("beauty",), camera=dataclasses.replace(renderer.default_camera, pixel_lattice="triangular"),
                        width=8, height=8, samples=1)


@pytest.mark.parametrize("lattice", PIXEL_LATTICES)
def test_acceptance_angle_blurs_beauty_and_leaves_the_gbuffer(renderer: Renderer, lattice: str) -> None:
    """A facet's lobe replaces the film filter for beauty; the G-buffer's depth stays on the centre ray, float for float."""
    assert renderer.default_camera.acceptance_angle_degrees == 0.0
    camera = dataclasses.replace(renderer.default_camera, pixel_lattice=lattice)
    kwargs = {"aovs": ("beauty", "depth"), "width": 32, "height": 24, "samples": 4}
    plain = renderer.render(camera=camera, **kwargs)
    facets = renderer.render(camera=dataclasses.replace(camera, acceptance_angle_degrees=5.0), **kwargs)
    assert not np.array_equal(plain["beauty"], facets["beauty"])
    assert np.array_equal(plain["depth"], facets["depth"])
    with pytest.raises(RuntimeError, match="acceptance angle .* outside"):
        renderer.render(camera=dataclasses.replace(camera, acceptance_angle_degrees=400.0), **kwargs)


@pytest.mark.parametrize("lattice", PIXEL_LATTICES)
@pytest.mark.parametrize("lens", ["fisheye_polynomial", "omnidirectional"])
def test_lattice_preview_draws_each_texel_from_its_nearest_centre(renderer: Renderer, lattice: str, lens: str) -> None:
    """Every texel holds the pixel whose documented centre is nearest, by brute force over all of them, the lat-long's wrapping."""
    camera = dataclasses.replace(renderer.default_camera, lens=lens, pixel_lattice=lattice)
    width, height, scale = 12, 8, 6
    image = np.arange(width * height, dtype=np.float32).reshape(height, width, 1)
    preview = lattice_preview(image, camera, scale=scale)
    row_pitch = math.sqrt(3.0) / 2.0 if lattice == "hexagonal" else 1.0
    assert preview.shape == (round(height * row_pitch * scale), width * scale, 1)
    rows, columns = np.divmod(np.arange(width * height), width)
    centre_x = columns + (0.25 + 0.5 * (rows % 2) if lattice == "hexagonal" else 0.5)
    centre_y = (rows + 0.5) * row_pitch
    texel_y, texel_x = np.mgrid[0:preview.shape[0], 0:preview.shape[1]]
    dx = ((texel_x + 0.5) * width / preview.shape[1])[..., None] - centre_x
    if lens == "omnidirectional":
        dx -= width * np.round(dx / width)
    distance = np.hypot(dx, ((texel_y + 0.5) * height * row_pitch / preview.shape[0])[..., None] - centre_y)
    chosen = np.take_along_axis(distance, preview.astype(np.int64), axis=-1)[..., 0]
    assert (chosen <= distance.min(axis=-1) + 1e-5).all()


def test_lattice_preview_rejects_bad_input(renderer: Renderer) -> None:
    image = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="channels"):
        lattice_preview(image[..., 0], renderer.default_camera)
    with pytest.raises(ValueError, match="scale"):
        lattice_preview(image, renderer.default_camera, scale=0)
    with pytest.raises(TypeError, match="display_encode its result"):
        lattice_preview(display_encode(image), renderer.default_camera)
    with pytest.raises(ValueError, match="triangular"):
        lattice_preview(image, dataclasses.replace(renderer.default_camera, pixel_lattice="triangular"))


def test_unknown_lens_is_rejected_before_the_abi(renderer: Renderer) -> None:
    with pytest.raises(ValueError, match="pinhole"):
        renderer.render(aovs=("beauty",), camera=dataclasses.replace(renderer.default_camera, lens="pinhole"),
                        width=8, height=8, samples=1)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"near_clip": 0.0}, "clips"),
        ({"far_clip": 0.01}, "clips"),
        ({"rotation": (0.0, math.nan, 0.0)}, "rotation"),
        ({"focal_length_mm": 0.0}, "focal length"),
        ({"position": (math.nan, 0.0, 6.0)}, "position"),
        ({"film_back_mm": (36.0, math.inf)}, "film back"),
        ({"iso": math.nan}, "ISO"),
        ({"aperture": 1e30}, "EV100"),
        ({"fisheye_field_of_view_degrees": 400.0}, "field of view"),
    ],
)
def test_invalid_camera_is_rejected_by_name(renderer: Renderer, fields: dict, reason: str) -> None:
    camera = dataclasses.replace(renderer.default_camera, **fields)
    with pytest.raises(RuntimeError, match=reason):
        renderer.render(aovs=("depth",), camera=camera, width=8, height=8)


def test_invalid_previous_camera_is_rejected_by_name(renderer: Renderer) -> None:
    previous = dataclasses.replace(renderer.default_camera, near_clip=0.0)
    with pytest.raises(RuntimeError, match="previousCamera: clips"):
        renderer.render(aovs=("motionVector",), previous_camera=previous, width=8, height=8)


def test_unbounded_far_clip_renders(renderer: Renderer) -> None:
    camera = dataclasses.replace(renderer.default_camera, far_clip=math.inf)
    size = {"width": 16, "height": 16}
    assert np.array_equal(renderer.render(aovs=("depth",), camera=camera, **size)["depth"],
                          renderer.render(aovs=("depth",), **size)["depth"])


def test_default_camera_reports_the_profile_lens(renderer: Renderer) -> None:
    camera = renderer.default_camera
    assert camera.lens == json.loads((REPO_ROOT / "assets" / "config" / "profile.json").read_text())["camera"]["lens"]["projection"]
    assert len(camera.fisheye_coefficients) == 4
    assert 0.0 < camera.fisheye_field_of_view_degrees <= 360.0


def test_unknown_aov_names_the_offender(renderer: Renderer) -> None:
    with pytest.raises(ValueError, match="nonsense"):
        renderer.render(aovs=("nonsense",), width=8, height=8)


def test_invalid_arguments_are_rejected(renderer: Renderer) -> None:
    with pytest.raises(ValueError):
        renderer.render(aovs=(), width=8, height=8)
    with pytest.raises(ValueError):
        renderer.render(aovs=("beauty",), width=8, height=8, samples=0)
    with pytest.raises(ValueError):
        renderer.render(aovs=("beauty",), width=0, height=8)


def test_unknown_scene_raises() -> None:
    with pytest.raises(RuntimeError):
        Renderer("scenes/does_not_exist.json")


@pytest.mark.parametrize(
    ("flags", "optics"),
    [((), {}), (("--bloom", "0.2", "--aberration", "0.02"), {"bloom_strength": 0.2, "aberration": 0.02})],
)
def test_beauty_matches_render_beauty_cli(renderer: Renderer, tmp_path: Path, flags: tuple, optics: dict) -> None:
    """Python's Beauty must be bit-identical to the CLI's, which is what proves the two paths are one renderer, optics included."""
    openexr = pytest.importorskip("OpenEXR", reason="reading the reference EXR needs the OpenEXR module")
    imath = pytest.importorskip("Imath")

    binary = Path(os.environ.get("PATHTRACER_RENDER_BEAUTY", REPO_ROOT / "build" / "render_beauty"))
    if not binary.exists():
        pytest.skip("render_beauty is not built")

    reference = tmp_path / "cli.exr"
    subprocess.run(
        [str(binary), "--scene", SCENE, "--width", "64", "--height", "64",
         "--passes", "8", "--seed", "1", "--aov", "beauty",
         # --out is mandatory; the PNG is display-encoded and discarded, the EXR is the linear reference.
         "--out", str(tmp_path / "cli.png"), "--out-exr", str(reference), *flags],
        check=True, capture_output=True,
    )

    exr = openexr.InputFile(str(reference))
    pixel = imath.PixelType(imath.PixelType.FLOAT)
    channels = [np.frombuffer(exr.channel(c, pixel), dtype=np.float32).reshape(64, 64) for c in ("R", "G", "B")]
    expected = np.stack(channels, axis=-1)

    actual = renderer.render(aovs=("beauty",), width=64, height=64, samples=8, seed=1, **optics)["beauty"]
    assert np.array_equal(actual, expected)


def test_omitted_samples_and_bloom_take_the_profile(renderer: Renderer) -> None:
    """Omitting samples and bloom_strength is exactly asking for profile.json's values; aberration has no profile key and is off."""
    size = {"aovs": ("beauty",), "width": 32, "height": 24}
    default = renderer.render(**size)["beauty"]
    profile = json.loads((REPO_ROOT / "assets" / "config" / "profile.json").read_text())
    assert renderer.default_samples == profile["pathTracer"]["samplesPerPixel"]
    assert renderer.default_bloom_strength == pytest.approx(profile["render"]["bloomStrength"])
    explicit = renderer.render(**size, samples=renderer.default_samples, bloom_strength=renderer.default_bloom_strength)["beauty"]
    assert np.array_equal(default, explicit)


def test_unbounded_profile_needs_explicit_samples(tmp_path: Path) -> None:
    """A progressive profile may accumulate forever, which no headless render can, so samples=None is refused by name."""
    assets = REPO_ROOT / "assets"
    for entry in assets.iterdir():
        if entry.name != "config":
            (tmp_path / entry.name).symlink_to(entry)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "sensor.json").symlink_to(assets / "config" / "sensor.json")
    profile = json.loads((assets / "config" / "profile.json").read_text())
    profile["pathTracer"]["samplesPerPixel"] = 0
    profile["render"]["renderMode"] = "progressive"
    (tmp_path / "config" / "profile.json").write_text(json.dumps(profile))
    with Renderer(SCENE, asset_root=str(tmp_path)) as unbounded:
        assert unbounded.default_samples == 0
        with pytest.raises(ValueError, match="samplesPerPixel is 0"):
            unbounded.render(aovs=("beauty",), width=8, height=8)
        assert unbounded.render(aovs=("beauty",), width=8, height=8, samples=4)["beauty"].shape == (8, 8, 3)


@pytest.mark.parametrize("lattice", PIXEL_LATTICES)
def test_display_optics_change_only_the_picture(renderer: Renderer, lattice: str) -> None:
    """Bloom and fringe act on beauty on either lattice; depth is data and keeps every float."""
    camera = dataclasses.replace(renderer.default_camera, pixel_lattice=lattice)
    kwargs = {"aovs": ("beauty", "depth"), "width": 32, "height": 24, "samples": 2, "camera": camera}
    off = renderer.render(**kwargs)
    on = renderer.render(**kwargs, bloom_strength=0.3, aberration=0.05)
    assert not np.array_equal(off["beauty"], on["beauty"])
    assert np.array_equal(off["depth"], on["depth"])


@pytest.mark.parametrize(
    ("optics", "lattice", "message"),
    [
        ({"bloom_strength": 1.1}, "square", "bloom strength .* lies outside"),
        ({"bloom_strength": math.nan}, "square", "bloom strength .* lies outside"),
        ({"aberration": 1.0}, "square", "aberration .* lies outside"),
        ({"aberration": -0.1}, "hexagonal", "aberration .* lies outside"),
        ({"bloom_strength": 1.1}, "hexagonal", "bloom strength .* lies outside"),
    ],
)
def test_invalid_display_optics_are_rejected_by_name(renderer: Renderer, optics: dict, lattice: str, message: str) -> None:
    """Bloom is a fraction in [0, 1] and the fringe lies in [0, 1), on either lattice."""
    camera = dataclasses.replace(renderer.default_camera, pixel_lattice=lattice)
    with pytest.raises(RuntimeError, match=message):
        renderer.render(aovs=("beauty",), width=8, height=8, camera=camera, **optics)


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    """The IEC 61966-2-1 inverse EOTF, the curve the OCIO "Un-tone-mapped" sRGB view reduces to."""
    return np.where(linear <= 0.0031308, linear * 12.92, (1.055 * linear ** (1 / 2.4)) - 0.055)


def test_display_encode_follows_the_srgb_curve() -> None:
    """The encode must be sRGB, not a gamma-2.2 approximation of it; dither moves the result by at most one count."""
    linear = np.array([[[0.0] * 3, [0.05] * 3, [0.2] * 3, [0.5] * 3, [1.0] * 3]], dtype=np.float32)
    encoded = display_encode(linear)[0, :, 0].astype(np.int16)
    expected = np.rint(_srgb_encode(linear[0, :, 0].astype(np.float64)) * 255.0).astype(np.int16)
    assert np.all(np.abs(encoded - expected) <= 1)
    # The endpoints are exact: clamping pins them either side of the dither.
    assert encoded[0] == 0 and encoded[-1] == 255


def test_display_encode_is_monotonic_and_clamps() -> None:
    ramp = np.linspace(-1.0, 3.0, 256, dtype=np.float32).repeat(3).reshape(1, 256, 3)
    encoded = display_encode(ramp)[0, :, 0].astype(np.int16)
    assert encoded[0] == 0 and encoded[-1] == 255
    # Dither is a function of pixel position with |d| < 1 code, so neighbouring codes invert by at most one.
    assert np.all(np.diff(encoded) >= -1)


def test_display_encode_exposure_is_a_stop() -> None:
    """exposure_ev must be stops, so +1 EV doubles the linear value before the curve rather than the encoded one."""
    linear = np.full((1, 1, 3), 0.1, dtype=np.float32)
    assert abs(int(display_encode(linear, exposure_ev=1.0)[0, 0, 0]) -
               int(display_encode(np.full((1, 1, 3), 0.2, dtype=np.float32))[0, 0, 0])) <= 1


def test_display_encode_exposes_by_the_camera(renderer: Renderer) -> None:
    """ISO 12232 SOS: the luminance whose image-plane exposure is 10/S lx s, at the camera's own gain, encodes as code 118."""
    camera = renderer.default_camera
    q = math.pi / 4.0 * 0.9 * 0.98 * math.cos(math.pi / 18.0) ** 4
    luminance = 10.0 * camera.aperture ** 2 / (q * camera.shutter_seconds * camera.iso)
    encoded = display_encode(np.full((64, 64, 3), luminance, dtype=np.float32), camera=camera)
    assert abs(float(encoded.mean()) - 118.0) <= 0.5


def test_display_encode_camera_is_the_gain_times_the_trim(renderer: Renderer) -> None:
    """camera= and exposure_ev compose as one product, render_beauty's, so the bytes match a pre-multiplied raw encode."""
    camera = renderer.default_camera
    linear = np.linspace(0.0, 400.0, 48, dtype=np.float32).repeat(3).reshape(4, 12, 3)
    premultiplied = linear * np.float32(np.float32(camera.exposure_gain) * np.float32(2.0))
    assert np.array_equal(display_encode(linear, camera=camera, exposure_ev=1.0), display_encode(premultiplied))


def test_display_encode_rejects_an_invalid_camera_by_name(renderer: Renderer) -> None:
    camera = dataclasses.replace(renderer.default_camera, aperture=math.nan)
    with pytest.raises(RuntimeError, match="aperture"):
        display_encode(np.zeros((2, 2, 3), dtype=np.float32), camera=camera)


def test_meter_and_camera_agree_at_sunny_16(renderer: Renderer) -> None:
    """f/16, 1/128 s, ISO 100 is EV100 15, which a K = 12.5 spot meter reads off 4096 cd/m^2."""
    sunny = dataclasses.replace(renderer.default_camera, aperture=16.0, shutter_seconds=1.0 / 128.0, iso=100.0)
    assert sunny.ev100 == 15.0 and meter_ev100(4096.0) == 15.0
    assert meter_ev100(0.125) == 0.0


def test_display_encode_without_transform_is_linear() -> None:
    linear = np.full((1, 1, 3), 0.5, dtype=np.float32)
    assert abs(int(display_encode(linear, display_transform=False)[0, 0, 0]) - 128) <= 1


def test_display_encode_rejects_wrong_shape() -> None:
    for bad in (np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4, 2), dtype=np.float32)):
        with pytest.raises(ValueError):
            display_encode(bad)


def test_display_encode_accepts_a_non_contiguous_view() -> None:
    """A caller slicing an RGBA buffer passes a strided view; the encode must copy rather than read past it."""
    padded = np.zeros((2, 3, 4), dtype=np.float32)
    padded[..., :3] = 0.5
    assert display_encode(padded[..., :3]).shape == (2, 3, 3)


def test_show_sky_blackens_only_the_background(renderer: Renderer) -> None:
    """showSky gates the primary ray's own miss, so it must change the background and nothing the camera hits."""
    # Outside the box looking at it through a square pinhole, whatever the profile authors.
    camera = dataclasses.replace(renderer.default_camera, position=(0.0, 0.0, 6.0), rotation=(0.0, 0.0, 0.0), lens="rectilinear",
                                 pixel_lattice="square")
    common = {"aovs": ("beauty",), "width": 64, "height": 36, "samples": 2, "seed": 1, "camera": camera}
    sky = renderer.render(**common, show_sky=True)["beauty"]
    without = renderer.render(**common, show_sky=False)["beauty"]

    corner = (slice(0, 6), slice(0, 6))
    assert sky[corner].max() > 0.0
    assert np.array_equal(without[corner], np.zeros_like(without[corner]))
    # The Cornell box fills the centre, and it is lit by the environment in both renders.
    interior = (slice(12, 24), slice(24, 40))
    assert np.array_equal(sky[interior], without[interior])


def test_show_sky_default_hides_the_sky(renderer: Renderer) -> None:
    """The default hides the sky, as the viewer does, so headless and viewer agree on one background."""
    common = {"aovs": ("beauty",), "width": 32, "height": 18, "samples": 2, "seed": 1}
    assert np.array_equal(renderer.render(**common)["beauty"],
                          renderer.render(**common, show_sky=False)["beauty"])


def test_optional_flags_reject_a_non_tristate(renderer: Renderer) -> None:
    """The ABI rejects anything outside {PT_DEFAULT, 0, 1} rather than coercing it to true."""
    for field in ("show_sky", "env_light_enabled"):
        with pytest.raises(RuntimeError, match=field):
            renderer.render(aovs=("beauty",), width=8, height=8, samples=1, **{field: 2})


def test_env_light_enabled_changes_the_lighting(renderer: Renderer) -> None:
    """Unlike show_sky, removing the environment from the light set must change what the camera hits."""
    common = {"aovs": ("beauty",), "width": 64, "height": 36, "samples": 2, "seed": 1, "show_sky": False}
    lit = renderer.render(**common, env_light_enabled=True)["beauty"]
    unlit = renderer.render(**common, env_light_enabled=False)["beauty"]
    assert not np.array_equal(lit, unlit)
    assert unlit.mean() < lit.mean()


def test_default_pose_reads_back_the_authored_scene(renderer: Renderer) -> None:
    """cornell.json authors its root unrotated and its one ceiling light at [-90, 0, 0]."""
    assert renderer.default_root_rotation == (0.0, 0.0, 0.0)
    assert renderer.default_light_rotations == ((-90.0, 0.0, 0.0),)


def test_explicit_authored_pose_matches_none(renderer: Renderer) -> None:
    """Passing the authored pose is the same request as passing None, so it must render the same floats."""
    common = {"aovs": ("beauty", "depth"), "width": 32, "height": 18, "samples": 2, "seed": 1}
    implicit = renderer.render(**common)
    explicit = renderer.render(
        **common, root_rotation=renderer.default_root_rotation, light_rotations=renderer.default_light_rotations
    )
    for name in common["aovs"]:
        assert np.array_equal(implicit[name], explicit[name])


def test_pose_changes_the_image_and_returning_restores_it(renderer: Renderer) -> None:
    """A root or light pose rebuilds the scene; going back must reproduce the authored render bit for bit."""
    common = {"aovs": ("beauty", "depth"), "width": 32, "height": 18, "samples": 2, "seed": 1}
    authored = renderer.render(**common)
    turned_root = renderer.render(**common, root_rotation=(0.0, 30.0, 0.0))
    turned_light = renderer.render(**common, light_rotations=((-60.0, 0.0, 0.0),))
    restored = renderer.render(**common)
    assert not np.array_equal(authored["depth"], turned_root["depth"])
    assert not np.array_equal(authored["beauty"], turned_light["beauty"])
    for name in common["aovs"]:
        assert np.array_equal(authored[name], restored[name])


def test_env_rotation_turns_the_sky(renderer: Renderer) -> None:
    """Half a turn about Y moves the background, which the camera sees only through the environment."""
    common = {"aovs": ("beauty",), "width": 32, "height": 18, "samples": 1, "seed": 1, "show_sky": True}
    assert not np.array_equal(renderer.render(**common)["beauty"], renderer.render(**common, env_rotation=(0, 180, 0))["beauty"])


def test_light_temperature_scales_each_channel_of_the_light(renderer: Renderer) -> None:
    """With the environment off, paths do not depend on the light's colour, so a temperature rescales each channel uniformly.

    The per-channel ratio to the authored render recovers the light's new colour: warmer is redder, and returning to
    None restores the authored floats exactly.
    """
    common = {"aovs": ("beauty",), "width": 32, "height": 18, "samples": 2, "seed": 1, "env_light_enabled": False}
    authored = renderer.render(**common)["beauty"].astype(np.float64)
    lit = authored.min(axis=-1) > 0.0
    profile = json.loads((REPO_ROOT / "assets" / "config" / "profile.json").read_text())
    # Both renders round the same float sums: samples x (maxBounces + 1) terms per pixel, each half an ulp, bound the gap.
    tolerance = common["samples"] * (profile["pathTracer"]["maxBounces"] + 1) * np.finfo(np.float32).eps
    red_over_blue = []
    for kelvin in (2856.0, 6504.0):
        ratio = renderer.render(**common, light_temperatures=(kelvin,))["beauty"][lit] / authored[lit]
        spread = (ratio.max(axis=0) - ratio.min(axis=0)) / ratio.mean(axis=0)
        assert np.all(spread <= tolerance), f"{kelvin} K: per-channel spread {spread} exceeds {tolerance:.2e}"
        red_over_blue.append(ratio.mean(axis=0)[0] / ratio.mean(axis=0)[2])
    assert red_over_blue[0] > red_over_blue[1]
    assert np.array_equal(renderer.render(**common)["beauty"], authored.astype(np.float32))


@pytest.mark.parametrize(
    ("kelvin", "message"),
    [
        ((2856.0, 2856.0), "light_temperatures must have shape"),
        ((0.0,), "not positive or lies outside the scene gamut"),
        ((math.inf,), "not positive or lies outside the scene gamut"),
        ((500.0,), "not positive or lies outside the scene gamut"),
    ],
)
def test_invalid_light_temperature_is_rejected_by_name(renderer: Renderer, kelvin: tuple, message: str) -> None:
    """A wrong count is caught before the ABI, which reads raw floats; an unrepresentable temperature by the renderer."""
    with pytest.raises((ValueError, RuntimeError), match=message):
        renderer.render(aovs=("beauty",), width=8, height=8, light_temperatures=kelvin)


@pytest.mark.parametrize(
    ("pose", "message"),
    [
        ({"root_rotation": (0.0, 1.0)}, "root_rotation must have shape"),
        ({"light_rotations": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}, "light_rotations must have shape"),
        ({"env_rotation": (0.0, 0.0, 0.0, 0.0)}, "env_rotation must have shape"),
        ({"root_rotation": (0.0, math.nan, 0.0)}, "root rotation is not finite"),
        ({"light_rotations": ((math.inf, 0.0, 0.0),)}, "a light rotation is not finite"),
        ({"env_rotation": (0.0, 0.0, math.nan)}, "environment rotation is not finite"),
    ],
)
def test_invalid_pose_is_rejected_by_name(renderer: Renderer, pose: dict, message: str) -> None:
    """A wrong shape is caught before the ABI, which reads raw triples; a non-finite angle by the renderer, by name."""
    with pytest.raises((ValueError, RuntimeError), match=message):
        renderer.render(aovs=("depth",), width=8, height=8, **pose)
