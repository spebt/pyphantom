"""
_helper.py

Helper functions for phantom generation.
These functions include:

- put_disk_at_xy: Places a disk at a given x, y coordinate in an image.
- get_hot_rod_xy: Calculates the x, y coordinates for a hot rod.
- shift_xylist: Shifts a list of x, y coordinates by a given amount.
- transform_xylist: Transforms a list of x, y coordinates by a given angle and image dimensions.
- get_args_parsed: Parses command line arguments for the phantom generation program.
- get_phantomType: Determines the type of phantom based on provided arguments.

For more detailed information on each function, refer to their individual docstrings.

:author: Fang Han
:contact: fhan0904@gmail.com
"""

__all__ = [
    "put_disk_at_xy",
    "put_dot_at_xys",  # Added "put_dot_at
    "get_hot_rod_xy",
    "get_derenzo_section_xy",
    "get_args_parsed",
]

import numpy
import math


def put_disk_at_xy(img, xy, radius, value, ratio):
    import skimage

    xx, yy = skimage.draw.circle_perimeter(int(xy[0]), int(xy[1]), int(radius))
    img[xx, yy] = value * ratio
    xx, yy = skimage.draw.disk((int(xy[0]), int(xy[1])), int(radius))
    img[xx, yy] = value


def put_dot_at_xys(img: numpy.ndarray, xys: numpy.ndarray, value: int):
    img[xys[:, 0], xys[:, 1]] = value


def get_hot_rod_xy(Nlayer, origin, pitch):
    xy_list = []
    for ilayer in range(0, Nlayer):
        ilayer_y = origin[1] + pitch * math.sin(math.pi / 3) * ilayer
        ilayer_x_start = origin[0] + pitch * 0.5 * ilayer
        for idot in range(0, Nlayer - ilayer):
            idot_x = int(ilayer_x_start + pitch * idot)
            idot_y = int(ilayer_y)
            xy_list.append((idot_x, idot_y))
    return xy_list


def get_derenzo_section_xy(
    base_xy: tuple[int, int] = (0, 0),
    pitch_half: int = 2,
    N_layer: int = 3,
    sId: int = 0,
):
    pitch_short = int(pitch_half * math.sqrt(3))
    coeff_short = numpy.array([0])
    coeff_long = numpy.array([0])
    for id in range(1, N_layer):
        coeff_short = numpy.concatenate((coeff_short, numpy.full(id + 1, id)))
        coeff_long = numpy.concatenate((coeff_long, numpy.arange(-id, id + 1, 2)))
    uu = coeff_short * pitch_short
    vv = coeff_long * pitch_half
    if sId == 0:
        xx = vv + base_xy[0]
        yy = uu + base_xy[1]
    elif sId == 1:
        xx = vv + base_xy[0] - pitch_half * (N_layer - 1)
        yy = -uu + base_xy[1] + pitch_short * (N_layer - 1)
    elif sId == 2:
        xx = vv + base_xy[0] - pitch_half * (N_layer - 1)
        yy = uu + base_xy[1] - pitch_short * (N_layer - 1)
    elif sId == 3:
        xx = vv + base_xy[0]
        yy = base_xy[1] - uu
    elif sId == 4:
        xx = vv + base_xy[0] + pitch_half * (N_layer - 1)
        yy = uu + base_xy[1] - pitch_short * (N_layer - 1)
    elif sId == 5:
        xx = vv + base_xy[0] + pitch_half * (N_layer - 1)
        yy = -uu + base_xy[1] + pitch_short * (N_layer - 1)
    else:
        raise ValueError("Invalid section ID")
    return numpy.array([xx, yy]).T


def get_args_parsed(args, pname="generate"):
    import argparse

    parser = argparse.ArgumentParser(
        prog=pname,
        description="Python package for digital phantom generation",
        epilog="For more information, visit https:\\\\spebt.github.io\\pyphantom",
    )
    available_phantoms = ["hotrod", "Derenzo", "derenzo", "contrast", "dot", "disk"]
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Phantom type",
        required=True,
        choices=available_phantoms,
    )
    parser.add_argument(
        "-o", "--outdir", type=str, help="Output directory", default="output"
    )
    return parser.parse_args(args)


def get_phantomType(args: dict):
    if len(args) != 2:
        raise ValueError("Need 1 phantom type as argument")
    alllist = ["hotrod", "Derenzo", "derenzo", "contrast", "dot", "disk"]
    hotrod_list = ["hotrod", "Derenzo", "derenzo"]
    if args[1] in alllist:
        if args[1] in hotrod_list:
            return "hotrod (Derenzo)"
        else:
            return args[1]
    else:
        raise ValueError("Unknown phantom type")
