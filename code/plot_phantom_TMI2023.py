if __name__ == "__main__":
    import os, sys
    from matplotlib import pyplot as plt
    from torch import arange, cat, cos
    from torch import load as torch_load
    from torch import sin, tensor, Tensor
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
        print(f"Error loading file {in_filename}: {e}")
        sys.exit(1)

    phantom_tensor: Tensor = data["Phantom tensor"]

    fov_size_in_mm: List[float] = data["Metadata"]["size in mm"]  # mm
    fov_n_pixels = data["Metadata"]["n pixels"]  # px
    fov_mm_per_pixel: List[float] = data["Metadata"]["mm per pixel"]  # mm/px
    fov_size_in_mm: List[float] = data["Metadata"]["size in mm"]  # mm

    fig, ax = plt.subplots(
        1, 1, figsize=(16, 16), dpi=100, layout="constrained"
    )
    ax.imshow(
        phantom_tensor.T,
        cmap="gray",
        origin="lower",
        interpolation="none",
        extent=(
            -fov_size_in_mm[0] / 2,
            fov_size_in_mm[0] / 2,
            -fov_size_in_mm[1] / 2,
            fov_size_in_mm[1] / 2,
        ),
    )

    xticks = ax.set_xticks(
        arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 1),
        # labels=arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4).tolist(),
    )
    yticks = ax.set_yticks(
        arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 1),
        # labels=arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4).tolist(),
    )
    ax.set_xlabel("x (mm)", fontsize=12)

    ax.set_ylabel("y (mm)", fontsize=12)
    ax.autoscale()
    fig.savefig(
        in_filename.replace(".pt", ".png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close("all")

    # The interpolated phantom plot

    fig, ax = plt.subplots(
        1, 1, figsize=(16, 16), dpi=100, layout="constrained"
    )
    ax.imshow(
        phantom_tensor.T,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
        extent=(
            -fov_size_in_mm[0] / 2,
            fov_size_in_mm[0] / 2,
            -fov_size_in_mm[1] / 2,
            fov_size_in_mm[1] / 2,
        ),
    )

    xticks = ax.set_xticks(
        arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 1),
        # labels=arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4).tolist(),
    )
    yticks = ax.set_yticks(
        arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 1),
        # labels=arange(-fov_size_in_mm[0] / 2, fov_size_in_mm[0] / 2 + 1, 4).tolist(),
    )
    ax.set_xlabel("x (mm)", fontsize=12)

    ax.set_ylabel("y (mm)", fontsize=12)
    ax.autoscale()
    fig.savefig(
        in_filename.replace(".pt", "_interpolated.png"),
        dpi=200,
    )