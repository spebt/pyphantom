import numpy as np
import skimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
import os


def put_disk_at_xy(img, xy, radius, value, ratio):
    xx, yy = skimage.draw.circle_perimeter(int(xy[0]), int(xy[1]), radius)
    img[xx, yy] = value * ratio
    xx, yy = skimage.draw.disk((int(xy[0]), int(xy[1])), radius)
    img[xx, yy] = value


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
    return np.array(newlist)


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


def generate_hotrod_phantom():
    imgN_x_ = 180
    imgN_y_ = 180
    img = np.zeros((imgN_x_, imgN_y_))
    mask = np.zeros((imgN_x_, imgN_y_))
    radii = np.array([2, 3, 4, 6, 7, 9])
    pitches = 4 * radii
    R_min = 20
    R_max = 90
    Nlayers = (R_max - R_min) / pitches
    # print(pitches)
    # print(Nlayers)
    Nlayers = np.ceil(Nlayers)
    # print(Nlayers)
    shift_Rs = np.array([16, 20, 18, 30, 25, 36])
    for idx in range(0, 6):
        angle_rad = idx * math.pi / 3
        section_xylist = get_hot_rod_xy(
            int(Nlayers[idx]), (imgN_x_ / 2, imgN_y_ / 2), pitches[idx]
        )
        section_xylist = np.asarray(section_xylist)
        shift_x = math.cos(math.pi / 6) * shift_Rs[idx]
        shift_y = math.sin(math.pi / 6) * shift_Rs[idx]
        section_xylist = shift_xylist(np.array([shift_x, shift_y]), section_xylist)
        section_xylist = transform_xylist(angle_rad, (imgN_x_, imgN_y_), section_xylist)
        for dot_xy in section_xylist:
            put_disk_at_xy(img, dot_xy, radii[idx], 10, ratio=0.7)
            put_disk_at_xy(mask, dot_xy, radii[idx], idx + 1, ratio=1)

    fig, ax = plt.subplots(figsize=(11, 10), dpi=100)
    norm = plt.Normalize(0, 10)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "orange"])
    cbar = fig.colorbar(
        ax.imshow(
            img.T, origin="lower", cmap=mpl.colormaps["gray"], norm=norm, aspect="equal"
        ),
        pad=0.01,
    )

    for idx in range(0, 6):
        ax.plot(
            [90, math.cos(idx * math.pi / 3) * R_max + 90],
            [90, math.sin(idx * math.pi / 3) * R_max + 90],
            color="w",
            ls=":",
        )
    fig.tight_layout()
    # Output
    outDir = "output/"
    if os.path.exists(outDir) != True:
        try:
            os.mkdir(outDir)
        except Exception as err:
            print(err)
            exit(2)
    outImgName = "hotrod_phantom_plot_%dx%d.png" % (imgN_x_, imgN_y_)
    outNpzName = "hotrod_phantom_data_%dx%d.npz" % (imgN_x_, imgN_y_)
    print("Save phantom plot to", outDir + outImgName)
    fig.savefig(outDir + outImgName, dpi=100)
    print("Save phantom data to", outDir + outNpzName)
    np.savez_compressed(outDir + outNpzName, phantom=img, mask=mask)


def generate_dot_phantom():
    imgN_x_ = 180
    imgN_y_ = 180
    img = np.zeros((imgN_x_, imgN_y_))
    mask = np.zeros((imgN_x_, imgN_y_))
    img[89, 89] = 10
    mask[89, 89] = 1
    fig, ax = plt.subplots(figsize=(11, 10), dpi=100)
    norm = plt.Normalize(0, 10)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "orange"])
    cbar = fig.colorbar(
        ax.imshow(
            img.T, origin="lower", cmap=mpl.colormaps["gray"], norm=norm, aspect="equal"
        ),
        pad=0.01,
    )
    outDir = "output/"
    if os.path.exists(outDir) != True:
        try:
            os.mkdir(outDir)
        except Exception as err:
            print(err)
            exit(2)
    outImgName = "dot_phantom_plot_%dx%d.png" % (imgN_x_, imgN_y_)
    outNpzName = "dot_phantom_data_%dx%d.npz" % (imgN_x_, imgN_y_)
    print("Save phantom plot to", outDir + outImgName)
    fig.savefig(outDir + outImgName, dpi=100)
    print("Save phantom data to", outDir + outNpzName)
    np.savez_compressed(outDir + outNpzName, phantom=img, mask=mask)


def get_phantomType(args):
    # print(len(args), args[0])
    if len(args) != 2:
        raise ValueError("Need 1 phantom type as argument")
    alllist = ["hotrod", "Derenzo", "derenzo", "contrast", "dot"]
    hotrod_list = ["hotrod", "Derenzo", "derenzo"]
    if args[1] in alllist:
        if args[1] in hotrod_list:
            return "hotrod (Derenzo)"
        else:
            return args[1]
    else:
        raise ValueError("Unknown phantom type")


# Main stuff
phanType = ""
try:
    phanType = get_phantomType(sys.argv)
except Exception as err:
    print("Error:", err)
    exit(1)
print("Produce", phanType, "phantom.")
match phanType:
    case "hotrod (Derenzo)":
        # print('hotrod')
        generate_hotrod_phantom()
    case "contrast":
        print("contrast, WIP...")
    case "dot":
        generate_dot_phantom()
exit(0)
