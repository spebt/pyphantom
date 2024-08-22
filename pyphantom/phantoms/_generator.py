"""
_generator.py
=============

Functions for generating different types of phantoms, which
are artificial objects or patterns used for testing, calibrating, and tuning
imaging systems.

The following phantom types can be generated:

- Hotrod phantom: A phantom with a specific pattern of circular objects (rods)
  arranged in a radial pattern.
- Dot phantom: A phantom with a single dot at the center.
- Disk phantom: A phantom with a single disk at the center.

The module also includes a function to generate a phantom based on a type
specified as a command line argument.

Functions:
----------

- _derenzo_phantom(shape=(100, 100)): Generates a hotrod phantom with the
  specified shape.
- _dot_phantom(shape=(100, 100), xy=None): Generates a dot phantom with the
  specified shape and dot position.
- _disk_phantom(shape=(100, 100), xy=None, r=None): Generates a disk phantom
  with the specified shape, disk position, and radius.
- get_phantom(ptype, shape: tuple[int] = (100, 100), xy=None, r=None):
  Generates a phantom based on the type specified as a command line argument.

"""

__all__ = ["get_phantom"]

from ._helper import (
    put_disk_at_xy,
    put_dot_at_xy,
    get_hot_rod_xy,
    shift_xylist,
    transform_xylist,
)

from ._phantom import _phantom as phantom

import numpy as np
import math


def _derenzo_phantom(shape=(100, 100)):
    shape = np.array(shape)
    img = np.zeros(shape)
    mask = np.zeros(shape)
    radii = np.ceil(np.array([1, 2, 3, 4, 5, 6]) * shape[0] / 100)
    pitches = 3 * radii
    R_min = int(12 * shape[0] / 100)
    R_max = int(35 * shape[0] / 100)
    Nlayers = (R_max - R_min) / pitches
    Nlayers = np.ceil(Nlayers)
    print(Nlayers)
    shift_Rs = np.ceil(np.array([10, 12, 14, 20, 20, 20]) * shape[0] / 100)
    for idx in range(0, 6):
        angle_rad = idx * math.pi / 3
        section_xylist = get_hot_rod_xy(int(Nlayers[idx]), shape * 0.5, pitches[idx])
        section_xylist = np.asarray(section_xylist)
        shift_x = math.cos(math.pi / 6) * shift_Rs[idx]
        shift_y = math.sin(math.pi / 6) * shift_Rs[idx]
        section_xylist = shift_xylist(np.array([shift_x, shift_y]), section_xylist)
        section_xylist = transform_xylist(angle_rad, shape * 0.5, section_xylist)

        for dot_xy in section_xylist:
            put_dot_at_xy(img, dot_xy, 10)
            put_dot_at_xy(mask, dot_xy, 10)

            # put_disk_at_xy(img, dot_xy, radii[idx], 10, ratio=0.7)
            # put_disk_at_xy(mask, dot_xy, radii[idx], idx + 1, ratio=1)
    return phantom("derenzo", img, mask)


def _dot_phantom(shape=(100, 100), xy=None):
    img = np.zeros(shape)
    mask = np.zeros(shape)
    if xy is None:
        xy = (int(shape[0] * 0.5), int(shape[0] * 0.5))
    img[xy[0], xy[1]] = 10
    mask[xy[0], xy[1]] = 1
    return phantom("dot", img, mask)


def _disk_phantom(shape=(100, 100), xy=None, r=None):
    img = np.zeros(shape)
    mask = np.zeros(shape)
    if xy is None:
        xy = (int(shape[0] * 0.5), int(shape[0] * 0.5))
    if r is None:
        r = int(shape[0] * 0.25)
    put_disk_at_xy(img, xy, r, 10, ratio=0.7)
    put_disk_at_xy(mask, xy, r, 1, ratio=1)
    return phantom("disk", img, mask)


def get_phantom(
    ptype: str,
    shape: tuple[int, int] = (100, 100),
    xy: tuple[int, int] = (50, 50),
    r: int = 10,
):
    print(f"Produce {ptype} phantom, with dimension {shape[0]}x{shape[1]}")
    # call the corresponding phantom generator based on the phantom types
    # available hotrod, contrast, dot, disk

    if ptype == "derenzo":
        return _derenzo_phantom(shape)
    elif ptype == "contrast":
        print("contrast, WIP...")
    elif ptype == "dot":
        return _dot_phantom(shape, xy)
    elif ptype == "disk":
        return _disk_phantom(shape, xy, r)
