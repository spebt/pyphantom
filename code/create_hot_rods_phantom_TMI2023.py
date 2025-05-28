if __name__ == "__main__":
    from torch import arange, cat
    from torch import float32 as torch_float32
    from torch import save as torch_save
    from torch import tensor, zeros, pi

    from _disk_shape import fov_tensor_dict, hot_rods_add_sector

    fov_px_size_in_mm = (0.015, 0.015)  # mm

    fov_n_pxs = (668, 668)

    # size of the field of view in mm
    fov_size_in_mm = (
        tensor(fov_n_pxs) * tensor(fov_px_size_in_mm)
    ).tolist()  # mm

    fov_dict = fov_tensor_dict(fov_n_pxs, fov_size_in_mm)

    phantom = zeros(fov_n_pxs, dtype=torch_float32)
    pxs_per_mm = int(1.0 / fov_dict["mm per pixel"][0])

    radii = tensor([0.02, 0.025, 0.03, 0.04, 0.05, 0.06])  # mm
    shifts = tensor(
        [
            [0.35, 0.0],
            [0.35, 0.0],
            [0.35, 0.0],
            [0.35, 0.0],
            [0.30, 0.0],
            [0.45, 0.0],
        ]
    )  # x, y shift in mm
    n_x_layers = tensor(
        [38, 32, 26, 20, 17, 13]
    )  # Number of layers in the x-direction

    angles = arange(0, 2 * pi, 2 * pi / 6).unsqueeze(-1) + pi / 6

    sectors_centers_mm = []
    for i in range(shifts.shape[0]):
        transform = cat((angles[i : i + 1], shifts[i : i + 1]), dim=1)
        sector_centers_mm, sector_centers_px = hot_rods_add_sector(
            phantom,
            int(n_x_layers[i].item()),
            radii[i].item(),
            transform,
            fov_dict,
        )
        sectors_centers_mm.append(sector_centers_mm)

    out_dict = {
        "Description": "Hot-rod Phantom TMI 2023",
        "Metadata": {
            "size in mm": fov_dict["size in mm"].tolist(),
            "mm per pixel": fov_dict["mm per pixel"].tolist(),
            "n pixels": fov_dict["n pixels"].tolist(),
            "center coordinates in mm": fov_dict[
                "center coordinates in mm"
            ].tolist(),
            "rods radii in mm": radii.tolist(),
        },
        "Phantom tensor": phantom,
        "Phantom shape": phantom.shape,
        "Phantom dtype": phantom.dtype,
    }
    torch_save(
        out_dict,
        f"hot_rods_phantom_{fov_dict['size in mm'][0].item():.2f}_mm_x_{fov_dict['size in mm'][1].item():.2f}_mm.pt",
    )
