"""Headless access to the PATHTRACER renderer, returning AOVs as numpy arrays.

    >>> from pathtracer import Renderer
    >>> renderer = Renderer("scenes/cornell.json")
    >>> frame = renderer.render(width=256, height=256, samples=64, aovs=("beauty", "depth", "normal"))
    >>> frame["beauty"].shape
    (256, 256, 3)

Values are scene-referred linear, unclamped, with no display transform applied: what a model should train on, not
what a monitor should show. Pass one through ``display_encode`` for a picture that matches the viewer, or use
``render_beauty --out`` to write one from the CLI.

Arrays are C-contiguous float32, so ``torch.from_numpy(frame["beauty"])`` shares memory with no copy, leaving one
explicit ``.to(device)``.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from . import _ffi

__all__ = ["AOVS", "Camera", "Renderer", "aov_channels", "aov_needs_samples", "display_encode"]

_LIB = _ffi.load_library()

#: Every AOV the renderer can produce, in its own order. Read from the library, so Python cannot hold a stale copy.
AOVS: tuple[str, ...] = tuple(
    _LIB.pt_aov_name(index).decode() for index in range(_LIB.pt_aov_count())
)


def _aov_id(name: str) -> int:
    # int() rather than a cast: ctypes types every foreign return as Any, and _ffi's restype is what makes this an int.
    identifier = int(_LIB.pt_aov_id(name.encode()))
    if identifier < 0:
        raise ValueError(f"unknown AOV {name!r}; known AOVs are: {', '.join(AOVS)}")
    return identifier


def aov_channels(name: str) -> int:
    """Channels this AOV carries: 1 for a depth or filter response, 2 for UV and motion, 3 for radiance and vectors."""
    return int(_LIB.pt_aov_channels(_aov_id(name)))


def aov_needs_samples(name: str) -> bool:
    """Whether ``samples`` affects this AOV.

    False for the 15 primary-hit AOVs one pixel-centre camera ray each resolves; those are essentially free and
    converge immediately, so raising ``samples`` for them only wastes time.
    """
    return bool(_LIB.pt_aov_needs_samples(_aov_id(name)))


def display_encode(
    image: np.ndarray, *, exposure_ev: float = 0.0, display_transform: bool = True
) -> np.ndarray:
    """Scene-referred linear RGB to display-referred 8-bit sRGB, the viewer's own chain.

    Applies exposure, the OCIO display transform the window and ``render_beauty`` both use, triangular-PDF dither,
    then quantises. Takes ``(height, width, 3)`` float32 and returns ``(height, width, 3)`` uint8, ready for
    ``imshow`` or a PNG write.

    Pass ``display_transform=False`` for a data AOV such as depth or normal, whose raw range is not
    display-referred; exposure and the quantise still apply.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"display_encode takes (height, width, 3), got {image.shape}")
    height, width = image.shape[0], image.shape[1]
    if height < 1 or width < 1:
        raise ValueError(f"resolution must be positive, got {width}x{height}")
    # A caller may pass a slice or a non-float32 AOV, and the ABI reads raw float32; this is the one copy that costs.
    source = np.ascontiguousarray(image, dtype=np.float32)
    out = np.empty((height, width, 3), dtype=np.uint8)

    error = _ffi.make_error_buffer()
    if _LIB.pt_display_encode(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        width,
        height,
        exposure_ev,
        1 if display_transform else 0,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        error,
        len(error),
    ) != 0:
        raise RuntimeError(error.value.decode())
    return out


# Index-parallel with PT_LENS_* in _ffi.py and LensProjection in scene/lens.h: the projection names Camera.lens accepts.
LENS_PROJECTIONS = ("rectilinear", "fisheye_polynomial")


@dataclass(frozen=True)
class Camera:
    """Pose, lens and exposure.

    ``aperture``, ``shutter_seconds`` and ``iso`` set the photographic exposure value only. Neither projection
    produces depth of field, and ``aperture`` is not a lens radius.

    ``lens`` selects the projection: ``"rectilinear"`` is the pinhole, ``"fisheye_polynomial"`` is
    Kannala & Brandt's ``r(theta) = focal_length_mm * (theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9)``,
    whose coefficients are an OpenCV ``fisheye`` / COLMAP ``OPENCV_FISHEYE`` calibration's ``k1..k4`` unscaled. Only the
    radial geometry transfers: one focal length means ``fx == fy``, and the principal point is the sensor centre, so a
    calibrated camera's pixel grid is not reproduced. ``fisheye_field_of_view_degrees`` is the full angle across the
    image circle; samples outside the circle are black, and an ``r(theta)`` that is not provably monotone over it is
    rejected. The image circle's radius is ``focal_length_mm * theta_d(theta_max)``, so short focal lengths are what make the
    projection visible and a circle inside the gate leaves the corners black. Every AOV renders under either lens; G-buffer
    ``depth`` is the distance along the primary ray, the one depth a fisheye past 90 degrees still defines.

    Immutable, so an override is a ``dataclasses.replace`` of ``Renderer.default_camera`` rather than a mutation
    that could leak between renders.
    """

    position: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    film_back_mm: tuple[float, float]
    focal_length_mm: float
    near_clip: float
    far_clip: float
    aperture: float
    shutter_seconds: float
    iso: float
    lens: str = "rectilinear"
    fisheye_coefficients: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    fisheye_field_of_view_degrees: float = 180.0

    @classmethod
    def _from_struct(cls, struct: _ffi.PtCamera) -> Camera:
        return cls(
            position=(struct.position[0], struct.position[1], struct.position[2]),
            yaw_degrees=struct.yaw_degrees,
            pitch_degrees=struct.pitch_degrees,
            film_back_mm=(struct.film_back_mm[0], struct.film_back_mm[1]),
            focal_length_mm=struct.focal_length_mm,
            near_clip=struct.near_clip,
            far_clip=struct.far_clip,
            aperture=struct.aperture,
            shutter_seconds=struct.shutter_seconds,
            iso=struct.iso,
            lens=LENS_PROJECTIONS[struct.lens_projection],
            fisheye_coefficients=tuple(struct.fisheye_coefficients),
            fisheye_field_of_view_degrees=struct.fisheye_field_of_view_degrees,
        )

    def _to_struct(self) -> _ffi.PtCamera:
        struct = _ffi.PtCamera()
        struct.position = (ctypes.c_float * 3)(*self.position)
        struct.yaw_degrees = self.yaw_degrees
        struct.pitch_degrees = self.pitch_degrees
        struct.film_back_mm = (ctypes.c_float * 2)(*self.film_back_mm)
        struct.focal_length_mm = self.focal_length_mm
        struct.near_clip = self.near_clip
        struct.far_clip = self.far_clip
        struct.aperture = self.aperture
        struct.shutter_seconds = self.shutter_seconds
        struct.iso = self.iso
        # Raised here rather than at the ABI, so a typo never reaches a by-value struct field that only takes an int.
        if self.lens not in LENS_PROJECTIONS:
            raise ValueError(f"lens must be one of {LENS_PROJECTIONS}, got {self.lens!r}")
        struct.lens_projection = LENS_PROJECTIONS.index(self.lens)
        struct.fisheye_coefficients = (ctypes.c_float * 4)(*self.fisheye_coefficients)
        struct.fisheye_field_of_view_degrees = self.fisheye_field_of_view_degrees
        return struct


class Renderer:
    """One loaded scene, rendered as many times as you like.

    Loading builds the BVH, the environment-map CDFs and the thread pool, so construct once and call ``render``
    repeatedly rather than reconstructing per frame.
    """

    def __init__(self, scene_path: str, asset_root: str | None = None) -> None:
        error = _ffi.make_error_buffer()
        handle = _LIB.pt_renderer_open(
            asset_root.encode() if asset_root is not None else None,
            scene_path.encode(),
            error,
            len(error),
        )
        if not handle:
            raise RuntimeError(error.value.decode())
        self._handle = handle

    def close(self) -> None:
        """Releases the scene, its BVH and its thread pool. Idempotent."""
        handle, self._handle = getattr(self, "_handle", None), None
        if handle:
            _LIB.pt_renderer_close(handle)

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def default_camera(self) -> Camera:
        """profile.json's authored camera, the starting point for an override."""
        struct = _ffi.PtCamera()
        _LIB.pt_renderer_default_camera(self._handle, ctypes.byref(struct))
        return Camera._from_struct(struct)

    @property
    def default_resolution(self) -> tuple[int, int]:
        return (_LIB.pt_renderer_default_width(self._handle), _LIB.pt_renderer_default_height(self._handle))

    def render(
        self,
        *,
        aovs: Iterable[str] = ("beauty",),
        width: int | None = None,
        height: int | None = None,
        samples: int = 1,
        seed: int = 1,
        camera: Camera | None = None,
        previous_camera: Camera | None = None,
        show_sky: bool | None = None,
        env_light_enabled: bool | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Renders the requested AOVs and returns them keyed by the names given.

        Each producer runs at most once per call, so asking for several AOVs together costs far less than asking for
        them separately: the 10 path-traced lanes share one sample set, the 15 G-buffer lanes share one set of
        pixel-centre rays, and the Beauty filters share the one accumulated Beauty.

        ``samples`` is the number of one-sample passes averaged. The sampler's scramble is fixed by ``seed`` and its
        sequence index advances per pass, so the same arguments always reproduce the same floats exactly.

        ``show_sky`` decides whether a camera ray that hits nothing returns environment radiance. It gates the
        primary miss only -- indirect bounces and next-event estimation sample the environment either way -- so
        turning it off blackens the background without unlighting the scene. ``None`` keeps the headless default of
        showing it; the viewer's own default is off.

        ``env_light_enabled`` decides whether the environment is in the light set at all, which does change the
        lighting. ``None`` keeps the scene's authored ``environment.lightEnabled``.

        ``previous_camera`` is the view ``motionVector`` measures from: ``(dx, dy)`` in pixels of this frame, the
        displacement ``x_now - x_previous`` of the point seen at each pixel centre, zero where the previous view has no
        image of it. ``None`` is ``camera`` itself, so the motion is exactly zero.

        Returns arrays of shape ``(height, width, channels)``, float32, row 0 at the top.
        """
        names = tuple(aovs)
        if not names:
            raise ValueError("at least one AOV is required")
        if samples < 1:
            raise ValueError(f"samples must be at least 1, got {samples}")

        default_width, default_height = self.default_resolution
        width = default_width if width is None else width
        height = default_height if height is None else height
        if width < 1 or height < 1:
            raise ValueError(f"resolution must be positive, got {width}x{height}")

        identifiers = [_aov_id(name) for name in names]
        # numpy owns every byte, so nothing is freed across the ABI and the arrays outlive the call without a copy.
        buffers = [
            np.empty((height, width, _LIB.pt_aov_channels(identifier)), dtype=np.float32)
            for identifier in identifiers
        ]
        pointers = (ctypes.POINTER(ctypes.c_float) * len(buffers))(
            *(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for buffer in buffers)
        )

        request = _ffi.PtRenderRequest()
        request.camera = (camera if camera is not None else self.default_camera)._to_struct()
        if previous_camera is not None:
            request.previous_camera = ctypes.pointer(previous_camera._to_struct())
        request.width = width
        request.height = height
        request.samples = samples
        request.seed = seed
        request.aovs = (ctypes.c_int * len(identifiers))(*identifiers)
        request.aov_count = len(identifiers)
        request.show_sky = _ffi.PT_DEFAULT if show_sky is None else int(show_sky)
        request.env_light_enabled = _ffi.PT_DEFAULT if env_light_enabled is None else int(env_light_enabled)

        error = _ffi.make_error_buffer()
        if _LIB.pt_render(self._handle, ctypes.byref(request), pointers, error, len(error)) != 0:
            raise RuntimeError(error.value.decode())
        return dict(zip(names, buffers, strict=True))
