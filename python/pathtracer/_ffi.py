"""ctypes binding to libpathtracer_c.

Kept separate from the public API so ``__init__`` reads as the interface and this file as the plumbing.

``ctypes.CDLL`` rather than ``PyDLL``: CDLL releases the GIL for the duration of every foreign call, which is what
lets a render that takes milliseconds to seconds run without blocking other Python threads. ``PyDLL`` would hold it.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_ERROR_CAPACITY = 512

#: Mirrors PT_DEFAULT: the tri-state sentinel asking an optional request field to keep its default.
PT_DEFAULT = -1

# PT_ABI_VERSION in pathtracer_c.h. PtCamera and PtRenderRequest cross the boundary by value, so a layout drift must fail loudly.
PT_ABI_VERSION = 4

# PtCamera.lens_projection, PT_LENS_* in pathtracer_c.h. Index-parallel with LENS_PROJECTIONS in __init__.py.
PT_LENS_RECTILINEAR = 0
PT_LENS_FISHEYE_POLYNOMIAL = 1
PT_LENS_OMNIDIRECTIONAL = 2


class PtCamera(ctypes.Structure):
    """Mirrors ``PtCamera`` in include/pathtracer/api/pathtracer_c.h, field for field and in order."""

    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("yaw_degrees", ctypes.c_float),
        ("pitch_degrees", ctypes.c_float),
        ("film_back_mm", ctypes.c_float * 2),
        ("focal_length_mm", ctypes.c_float),
        ("near_clip", ctypes.c_float),
        ("far_clip", ctypes.c_float),
        ("aperture", ctypes.c_float),
        ("shutter_seconds", ctypes.c_float),
        ("iso", ctypes.c_float),
        ("lens_projection", ctypes.c_int),
        ("fisheye_coefficients", ctypes.c_float * 4),
        ("fisheye_field_of_view_degrees", ctypes.c_float),
    ]


class PtRenderRequest(ctypes.Structure):
    """Mirrors ``PtRenderRequest``."""

    _fields_ = [
        ("camera", PtCamera),
        ("previous_camera", ctypes.POINTER(PtCamera)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("samples", ctypes.c_int),
        ("seed", ctypes.c_uint),
        ("aovs", ctypes.POINTER(ctypes.c_int)),
        ("aov_count", ctypes.c_int),
        ("show_sky", ctypes.c_int),
        ("env_light_enabled", ctypes.c_int),
    ]


def _candidate_paths() -> list[Path]:
    """Where to look for the shared library, most explicit first.

    The environment variable wins so an out-of-tree or multi-config build can be pointed at without reinstalling;
    otherwise the default is the in-tree ``build/`` directory the project's own CMake preset writes to.
    """
    override = os.environ.get("PATHTRACER_LIB")
    if override:
        return [Path(override)]
    repo_root = Path(__file__).resolve().parents[2]
    return [repo_root / "build" / "libpathtracer_c.dylib", repo_root / "build" / "libpathtracer_c.so"]


def load_library() -> ctypes.CDLL:
    """Loads libpathtracer_c and declares every signature.

    Declaring argtypes/restype is not optional hygiene here: without them ctypes defaults every argument and return
    to ``int``, which silently truncates the 64-bit pointers this ABI passes and corrupts memory rather than failing.
    """
    searched = _candidate_paths()
    for path in searched:
        if path.exists():
            library = ctypes.CDLL(str(path))
            break
    else:
        joined = "\n  ".join(str(path) for path in searched)
        raise FileNotFoundError(
            f"libpathtracer_c not found. Build it with `cmake --build build --target pathtracer_c`, "
            f"or set PATHTRACER_LIB. Searched:\n  {joined}"
        )

    library.pt_renderer_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    library.pt_renderer_open.restype = ctypes.c_void_p
    library.pt_renderer_close.argtypes = [ctypes.c_void_p]
    library.pt_renderer_close.restype = None

    library.pt_aov_count.argtypes = []
    library.pt_aov_count.restype = ctypes.c_int
    library.pt_aov_name.argtypes = [ctypes.c_int]
    library.pt_aov_name.restype = ctypes.c_char_p
    library.pt_aov_id.argtypes = [ctypes.c_char_p]
    library.pt_aov_id.restype = ctypes.c_int
    library.pt_aov_channels.argtypes = [ctypes.c_int]
    library.pt_aov_channels.restype = ctypes.c_int
    library.pt_aov_needs_samples.argtypes = [ctypes.c_int]
    library.pt_aov_needs_samples.restype = ctypes.c_int

    library.pt_renderer_default_camera.argtypes = [ctypes.c_void_p, ctypes.POINTER(PtCamera)]
    library.pt_renderer_default_camera.restype = None
    library.pt_renderer_default_width.argtypes = [ctypes.c_void_p]
    library.pt_renderer_default_width.restype = ctypes.c_int
    library.pt_renderer_default_height.argtypes = [ctypes.c_void_p]
    library.pt_renderer_default_height.restype = ctypes.c_int

    library.pt_render.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PtRenderRequest),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    library.pt_render.restype = ctypes.c_int

    library.pt_display_encode.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    library.pt_display_encode.restype = ctypes.c_int

    library.pt_abi_version.argtypes = []
    library.pt_abi_version.restype = ctypes.c_int
    # Checked at load, the only point before a by-value struct is passed: a mismatched layout would otherwise read as garbage floats.
    actual = library.pt_abi_version()
    if actual != PT_ABI_VERSION:
        raise RuntimeError(f"libpathtracer ABI version {actual} does not match this package's {PT_ABI_VERSION}")
    return library


def make_error_buffer() -> ctypes.Array[ctypes.c_char]:
    return ctypes.create_string_buffer(_ERROR_CAPACITY)
