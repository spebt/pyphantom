if __name__ == "__main__":
    import os
    import sys

    from matplotlib import pyplot as plt
    from torch import arange, cat, cos
    from torch import load as torch_load
    from torch import sin, tensor
    from typing import Dict, List

    in_filename = sys.argv[1]

    # Check if the file exists
    if not os.path.isfile(in_filename):
        print(f"File {in_filename} does not exist.")
        sys.exit(1)

    # Read the file
    try:
        data = torch_load(in_filename, map_location="cpu")
    except Exception as e:
        print(f"Error reading file {in_filename}: {e}")

    radii = tensor(data["Metadata"]["rods radii in mm"])  # mm

    angles = arange(0, 2 * 3.14159, 2 * 3.14159 / 6).unsqueeze(-1) + 3.14159 / 6
    anno_radii = tensor([30.5, 28.75, 28.75, 30.25, 30.25, 31.75])  # mm

    anno_xy = cat(
        (
            cos(angles) * anno_radii.view(-1, 1),
            sin(angles) * anno_radii.view(-1, 1),
        ),
        -1,
    )
    anno_text = [f"{r*2:.2f} mm" for r in radii]

    phantom = data["Phantom tensor"]
    fov_size_in_mm: List[float] = data["Metadata"]["size in mm"]  # mm
    fov_n_pixels = data["Metadata"]["n pixels"]  # px
    fov_mm_per_pixel: List[float] = data["Metadata"]["mm per pixel"]  # mm/px
    fov_size_in_mm: List[float] = data["Metadata"]["size in mm"]  # mm

    plt.close("all")
    fig, ax = plt.subplots(dpi=150, figsize=(12, 12), layout="constrained")
    ax.imshow(
        phantom.T,
        cmap="gray",
        extent=(
            -fov_size_in_mm[0] / 2,
            fov_size_in_mm[0] / 2,
            -fov_size_in_mm[1] / 2,
            fov_size_in_mm[1] / 2,
        ),
        origin="lower",
        aspect="equal",
        interpolation="none",
    )
    # for sector_centers_mm in sectors_centers_mm:
    #     ax.plot(
    #         sector_centers_mm[:, 0],
    #         sector_centers_mm[:, 1],
    #         "o",
    #         markersize=2,
    #     )
    xticks = ax.set_xticks(
        arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4),
        # labels=arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4).tolist(),
    )
    yticks = ax.set_yticks(
        arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4),
        # labels=arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4).tolist(),
    )
    ax.set_xlabel("x (mm)", fontsize=12)

    ax.set_ylabel("y (mm)", fontsize=12)

    for i, rad in enumerate(angles):
        ax.annotate(
            anno_text[i],
            xy=(0, 0),
            xytext=tuple(anno_xy[i].tolist()),
            textcoords="data",
            fontsize=20,
            ha="center",
            va="center",
            color="w",
        )
    fig_title = f"Hot Rods Phantom, {fov_size_in_mm[0]} mm X {fov_size_in_mm[1]} mm"
    fig_title += f", {fov_n_pixels[0]} px X {fov_n_pixels[1]} px"
    fig.suptitle(fig_title, fontsize=20)
    fig.savefig(
        f"hot_rods_phantom_{fov_size_in_mm[0]}_mm_x_{fov_size_in_mm[1]}_mm.png",
        dpi=150,
    )

