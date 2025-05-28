if __name__ == "__main__":
    from _disk_shape import fov_tensor_dict, hot_rods_add_sector
    from matplotlib import pyplot as plt
    from torch import arange, cat, cos
    from torch import float32 as torch_float32
    from torch import sin, tensor, zeros, save as torch_save

    fov_size_in_mm = (64.0, 64.0)  # mm
    fov_px_size_in_mm = (0.125, 0.125)  # mm
    fov_n_pxs = (tensor(fov_size_in_mm) / tensor(fov_px_size_in_mm)).int().tolist()

    fov_dict = fov_tensor_dict(fov_n_pxs, fov_size_in_mm)

    phantom = zeros(fov_n_pxs, dtype=torch_float32)
    pxs_per_mm = int(1.0 / fov_dict["mm per pixel"][0])

    radii = tensor([0.5, 0.625, 0.75, 1.0, 1.25, 1.5])  # mm
    shifts = tensor(
        [[4, 0.0], [4.5, 0.0], [4, 0.0], [5, 0.0], [5, 0.0], [6, 0.0]]
    )  # x, y shift in mm
    n_x_layers = tensor([14, 11, 9, 7, 6, 5])  # Number of layers in the x-direction

    angles = arange(0, 2 * 3.14159, 2 * 3.14159 / 6).unsqueeze(-1) + 3.14159 / 6

    sectors_centers_mm = []
    for i in range(shifts.shape[0]):
        transform = cat((angles[i : i + 1], shifts[i : i + 1]), dim=1)
        sector_centers_mm, sector_centers_px = hot_rods_add_sector(
            phantom, int(n_x_layers[i].item()), radii[i].item(), transform, fov_dict
        )
        sectors_centers_mm.append(sector_centers_mm)

    anno_radii = tensor([30.5, 28.75, 28.75, 30.25, 30.25, 31.75])  # mm

    anno_xy = cat(
        (cos(angles) * anno_radii.view(-1, 1), sin(angles) * anno_radii.view(-1, 1)), -1
    )
    anno_text = [f"{r*2:.2f} mm" for r in radii]

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
    fig_title = f"Hot Rods Phantom, {fov_dict["size in mm"][0].item()} mm X {fov_dict["size in mm"][1].item()} mm"
    fig_title += f", {int(fov_dict["n pixels"][0].item())} px X {int(fov_dict["n pixels"][1].item())} px"
    fig.suptitle(fig_title, fontsize=20)
    fig.savefig(
        f"hot_rods_phantom_{fov_dict["size in mm"][0].item()}_mm_x_{fov_dict["size in mm"][1].item()}_mm.png",
        dpi=150,
    )

    out_dict = {
        "Description": "Hot Rods Phantom",
        "Metadata": {
            "size in mm": fov_dict["size in mm"].tolist(),
            "mm per pixel": fov_dict["mm per pixel"].tolist(),
            "n pixels": fov_dict["n pixels"].tolist(),
            "center coordinates in mm": fov_dict["center coordinates in mm"].tolist(),
            "rods radii in mm": radii.tolist(),
        },
        "Phantom tensor": phantom,
        "Phantom shape": phantom.shape,
        "Phantom dtype": phantom.dtype,
    }
    torch_save(
        out_dict,
        f"hot_rods_phantom_{fov_dict["size in mm"][0].item()}_mm_x_{fov_dict["size in mm"][1].item()}_mm.pt",
    )
