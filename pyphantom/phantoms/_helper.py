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
    "put_dot_at_xy",  # Added "put_dot_at
    "get_hot_rod_xy",
    "shift_xylist",
    "transform_xylist",
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


def put_dot_at_xy(img, xy, value):
    img[xy[0], xy[1]] = value


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


def shift_xylist(shifts, xylist):
    newlist = []
    for dot_xy in xylist:
        new_x = dot_xy[0] + shifts[0]
        new_y = dot_xy[1] + shifts[1]
        newlist.append([int(new_x), int(new_y)])
    return numpy.array(newlist)


def transform_xylist(angrad, imgDims, xylist):
    newlist = []
    for dot_xy in xylist:
        temp_x = dot_xy[0] - imgDims[0] * 0.5
        temp_y = dot_xy[1] - imgDims[1] * 0.5
        new_x = temp_x * math.cos(angrad) - temp_y * math.sin(angrad)
        new_y = temp_x * math.sin(angrad) + temp_y * math.cos(angrad)
        new_x = new_x + imgDims[0] * 0.5
        new_y = new_y + imgDims[1] * 0.5
        newlist.append((int(new_x), int(new_y)))
    return newlist


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
