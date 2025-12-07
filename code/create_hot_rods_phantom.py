
if __name__ == "__main__":

    from torch import arange, cat
    from torch import float32 as torch_float32
    from torch import save as torch_save
    from torch import tensor, zeros

    from _disk_shape import fov_tensor_dict, hot_rods_add_sector

    fov_size_in_mm = (70.0, 70.0)  # mm
    fov_px_size_in_mm = (0.25, 0.25)  # mm
    fov_n_pxs = (
        (tensor(fov_size_in_mm) / tensor(fov_px_size_in_mm)).int().tolist()
    )

    fov_dict = fov_tensor_dict(fov_n_pxs, fov_size_in_mm)

    phantom = zeros(fov_n_pxs, dtype=torch_float32)
    pxs_per_mm = int(1.0 / fov_dict["mm per pixel"][0])

    radii = tensor([0.5, 0.625, 0.75, 1.0, 1.25, 1.5])  # mm
    shifts = tensor(
        [[3.5, 0.0], [4.0, 0.0], [4.5, 0.0], [5.0, 0.0], [5.5, 0.0], [6.0, 0.0]]
    )
    n_x_layers = tensor(
        [14, 11, 9, 7, 6, 5]
    )  # Number of layers in the x-direction

    angles = arange(0, 2 * 3.14159, 2 * 3.14159 / 6).unsqueeze(-1) + 3.14159 / 6

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
        "Description": "Hot Rods Phantom densely",
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
        f"hot_rods_phantom_{fov_dict['size in mm'][0].item()}_mm_x_{fov_dict['size in mm'][1].item()}_mm.pt",
    )
