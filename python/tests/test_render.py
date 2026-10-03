"""Contract tests for the Python binding.

The strongest check here is ``test_beauty_matches_render_beauty_cli``: the renderer is deterministic by construction
(``render_beauty --assert-deterministic`` gates it), so Python's Beauty must be EXACTLY equal to the CLI's EXR at the
same scene, resolution, seed and pass count. Any tolerance there would hide the bugs worth catching.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from pathtracer import AOVS, LENS_PROJECTIONS, Renderer, aov_channels, aov_needs_samples, display_encode

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
    assert len(AOVS) == 32
    names = ("beauty", "depth", "lookahead", "normal", "sobel", "luminance", "gabor", "HSV", "DoG", "colourOpponent", "motionVector", "SNR")
    for name in names:
        assert name in AOVS


def test_aov_names_are_exact() -> None:
    assert aov_channels("bounceCount") == 1
    for name in ("bounce-count", "Bounce Count", "BounceCount"):
        with pytest.raises(ValueError, match="unknown AOV"):
            aov_channels(name)


def test_shapes_and_dtypes_match_the_declared_channels(renderer: Renderer) -> None:
    names = ("beauty", "depth", "normal", "UV")
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
    frame = renderer.render(aovs=("depth", "alpha"), width=64, height=48)
    depth, alpha = frame["depth"][..., 0], frame["alpha"][..., 0]
    assert np.isfinite(depth).all()
    hit = alpha > 0.0
    assert hit.any(), "the default camera should see geometry"
    assert (depth[hit] > 0.0).all()
    # The Cornell box encloses the camera, so every primary ray hits something well inside the far clip.
    assert depth[hit].max() < renderer.default_camera.far_clip


def test_lookahead_is_depth_on_the_profile_horizon(renderer: Renderer) -> None:
    """The lane's whole contract: clamp(1 - Z/lookaheadDistance, 0, 1) against the Depth lane of the same render."""
    horizon = json.loads((REPO_ROOT / "assets" / "config" / "profile.json").read_text())["pathTracer"]["lookaheadDistance"]
    frame = renderer.render(aovs=("lookahead", "depth", "alpha"), width=64, height=48)
    lookahead, depth, alpha = frame["lookahead"][..., 0], frame["depth"][..., 0], frame["alpha"][..., 0]
    hit = alpha > 0.0
    assert hit.any(), "the default camera should see geometry"
    expected = np.clip(1.0 - depth[hit] / horizon, 0.0, 1.0)
    assert np.allclose(lookahead[hit], expected, rtol=0.0, atol=1e-6)
    # A primary miss reads 0, the same value geometry beyond the horizon reads -- alpha is what separates the two.
    assert (lookahead[~hit] == 0.0).all() if (~hit).any() else True
    assert ((lookahead >= 0.0) & (lookahead <= 1.0)).all()


def test_normals_are_unit_length_where_geometry_was_hit(renderer: Renderer) -> None:
    frame = renderer.render(aovs=("normal", "alpha"), width=64, height=48)
    hit = frame["alpha"][..., 0] > 0.0
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
    for name in AOVS:
        array = renderer.render(aovs=(name,), width=32, height=32, samples=2)[name]
        assert array.shape == (32, 32, aov_channels(name))
        assert np.isfinite(array).all(), f"{name} produced a non-finite value"


def test_multi_aov_request_matches_single_requests(renderer: Renderer) -> None:
    names = ("beauty", "depth", "normal", "sobel")
    size = {"width": 32, "height": 32, "samples": 4, "seed": 5}
    together = renderer.render(aovs=names, **size)
    for name in names:
        alone = renderer.render(aovs=(name,), **size)[name]
        assert np.array_equal(alone, together[name]), f"{name} changed when requested alongside others"


def test_camera_override_changes_the_image(renderer: Renderer) -> None:
    default = renderer.default_camera
    moved = dataclasses.replace(default, rotation_degrees=(0.0, 25.0, 0.0))
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
    previous = dataclasses.replace(default, rotation_degrees=(0.0, -0.5, 0.0))
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
        frame = renderer.render(aovs=("depth", "worldPos", "alpha"), camera=camera, width=48, height=32)
        hit = frame["alpha"][..., 0] > 0.0
        assert hit.any(), f"{lens}: the camera should see geometry"
        distance = np.linalg.norm(frame["worldPos"][hit] - np.asarray(camera.position, dtype=np.float32), axis=-1)
        assert np.allclose(frame["depth"][..., 0][hit], distance, rtol=1e-5, atol=0.0), lens


def test_omnidirectional_pixels_are_the_documented_longitude_and_latitude(renderer: Renderer) -> None:
    """Each hit's direction from the eye lands at its own pixel centre under Camera's lat-long formula, so the docstring is the code."""
    # Pitched, yawed and rolled at once, so the frame is the documented Rz @ Ry @ Rx and not merely the identity.
    camera = dataclasses.replace(renderer.default_camera, lens="omnidirectional", rotation_degrees=(20.0, 30.0, 40.0))
    width, height = 64, 32
    frame = renderer.render(aovs=("worldPos", "alpha"), camera=camera, width=width, height=height)
    hit = frame["alpha"][..., 0] > 0.0
    assert hit.any(), "the lat-long should see the scene"
    rotation = _rotation_xyz(camera.rotation_degrees)
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


def test_unknown_lens_is_rejected_before_the_abi(renderer: Renderer) -> None:
    with pytest.raises(ValueError, match="pinhole"):
        renderer.render(aovs=("beauty",), camera=dataclasses.replace(renderer.default_camera, lens="pinhole"),
                        width=8, height=8, samples=1)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"near_clip": 0.0}, "clips"),
        ({"far_clip": 0.01}, "clips"),
        ({"rotation_degrees": (0.0, math.nan, 0.0)}, "rotation"),
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
    assert camera.lens == "rectilinear"
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


def test_beauty_matches_render_beauty_cli(renderer: Renderer, tmp_path: Path) -> None:
    """Python's Beauty must be bit-identical to the CLI's, which is what proves the two paths are one renderer."""
    openexr = pytest.importorskip("OpenEXR", reason="reading the reference EXR needs the OpenEXR module")
    imath = pytest.importorskip("Imath")

    binary = REPO_ROOT / "build" / "render_beauty"
    if not binary.exists():
        pytest.skip("render_beauty is not built")

    reference = tmp_path / "cli.exr"
    subprocess.run(
        [str(binary), "--scene", SCENE, "--width", "64", "--height", "64",
         "--passes", "8", "--seed", "1", "--aov", "beauty",
         # --out is mandatory; the PNG is display-encoded and discarded, the EXR is the linear reference.
         "--out", str(tmp_path / "cli.png"), "--out-exr", str(reference)],
        check=True, capture_output=True,
    )

    exr = openexr.InputFile(str(reference))
    pixel = imath.PixelType(imath.PixelType.FLOAT)
    channels = [np.frombuffer(exr.channel(c, pixel), dtype=np.float32).reshape(64, 64) for c in ("R", "G", "B")]
    expected = np.stack(channels, axis=-1)

    actual = renderer.render(aovs=("beauty",), width=64, height=64, samples=8, seed=1)["beauty"]
    assert np.array_equal(actual, expected)


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
    # Dither is a pure function of uv, so it perturbs neighbours independently; monotonicity holds to one count.
    assert np.all(np.diff(encoded) >= -1)


def test_display_encode_exposure_is_a_stop() -> None:
    """exposure_ev must be stops, so +1 EV doubles the linear value before the curve rather than the encoded one."""
    linear = np.full((1, 1, 3), 0.1, dtype=np.float32)
    assert abs(int(display_encode(linear, exposure_ev=1.0)[0, 0, 0]) -
               int(display_encode(np.full((1, 1, 3), 0.2, dtype=np.float32))[0, 0, 0])) <= 1


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
    common = {"aovs": ("beauty",), "width": 64, "height": 36, "samples": 2, "seed": 1}
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
