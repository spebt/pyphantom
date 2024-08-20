import numpy as np
import math
import skimage


class phantom:
    def __init__(self, name, img: np.ndarray, mask: np.ndarray) -> None:
        self.name = name
        self.image = img
        self.mask = mask

    def __str__(self) -> str:
        return f"{self.name} {self.image.shape}"


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


def generate_hotrod_phantom(shape=(100, 100)):
    img = np.zeros(shape)
    mask = np.zeros(shape)
    radii = np.array([2, 3, 4, 6, 7, 9]) * shape[0] / 180
    pitches = 4 * radii
    R_min = 20 * shape[0] / 180
    R_max = 90 * shape[0] / 180
    Nlayers = (R_max - R_min) / pitches
    Nlayers = np.ceil(Nlayers)
    shift_Rs = np.array([16, 20, 18, 30, 25, 36]) * shape[0] / 180
    for idx in range(0, 6):
        angle_rad = idx * math.pi / 3
        section_xylist = get_hot_rod_xy(int(Nlayers[idx]), shape * 0.5, pitches[idx])
        section_xylist = np.asarray(section_xylist)
        shift_x = math.cos(math.pi / 6) * shift_Rs[idx]
        shift_y = math.sin(math.pi / 6) * shift_Rs[idx]
        section_xylist = shift_xylist(np.array([shift_x, shift_y]), section_xylist)
        section_xylist = transform_xylist(angle_rad, shape * 0.5, section_xylist)
        for dot_xy in section_xylist:
            put_disk_at_xy(img, dot_xy, radii[idx], 10, ratio=0.7)
            put_disk_at_xy(mask, dot_xy, radii[idx], idx + 1, ratio=1)


def generate_dot_phantom(shape=(100, 100)):
    img = np.zeros(shape)
    mask = np.zeros(shape)
    img[89, 89] = 10
    mask[89, 89] = 1


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
