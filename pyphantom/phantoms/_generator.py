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
    put_dot_at_xys,
    get_derenzo_section_xy,
)

from ._phantom import _phantom as phantom

import numpy as np
import math


def _derenzo_phantom(shape=(100, 100)):
    hw = shape[0] // 2
    hh = shape[1] // 2
    img = np.zeros(shape)
    mask = np.zeros(shape)
    sr_base = shape[0] // 10
    # pitches = np.array([2, 3, 4, 5, 6, 7])
    radii = np.array([1, 2, 3, 4, 5, 5])
    sr = radii + sr_base
    pitches = np.max((radii * 1.5, np.full(6, 2)), axis=0).astype(int)
    rmax = np.full(6, hw * 0.95)
    nlayers = np.ceil((rmax - sr - radii * 2) / (pitches * 2)).astype(int)
    print(f"nlayers: {nlayers}")
    sw = sr * math.cos(0.5236)
    sh = sr * math.sin(0.5236)
    base_xys = [
        (hw, hh + sr[0]),
        (hw - sw[1], hh + sh[1]),
        (hw - sw[2], hh - sh[2]),
        (hw, hh - sr[3]),
        (hw + sw[4], hh - sh[4]),
        (hw + sw[5], hh + sh[5]),
    ]
    for sid in range(0, 6):
        xyarray = get_derenzo_section_xy(
            base_xy=base_xys[sid],
            pitch_half=pitches[sid],
            N_layer=nlayers[sid],
            sId=sid,
        )
        for xy in xyarray:
            put_disk_at_xy(img, xy, radii[sid], 10, ratio=0.7)
            put_disk_at_xy(mask, xy, radii[sid], sid + 1, ratio=1)

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
