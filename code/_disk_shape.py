from typing import Dict, Sequence

from torch import Tensor, arange, cat
from torch import float32 as torch_float32
from torch import meshgrid, norm, ones, sqrt, stack, tensor, zeros, bucketize
from torch.nn.functional import conv2d

from _geometry_2d_transform import transform_to_positions_2d_batch


def hot_rods_sector_centers(
    n_x_layers: int,
    radius: float,
    transform: Tensor,
    fov_dict: Dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Get the centers of the sectors in a hexagonal grid.

    Parameters
    ----------

    n_x_layers : int
        Number of layers in the x-direction.

    radius : float
        Radius of the rods in millimeters.

    transform : torch.Tensor
        Transformation to apply to the sector centers.
        This should be a 2D tensor of shape (n, 3) including translation and rotation.

    fov_dict : Dict[str, Tensor]
        Dictionary containing the field of view information, including 'n pixels'.
        'size in mm', and 'center coordinates in mm'.


    Returns
    -------

    sector_centers_mm : torch.Tensor
        Centers of the sectors in millimeters.

    sector_centers_px : torch.Tensor
        Centers of the sectors in pixels.
    """

    sector_center_grid = stack(
        [
            tensor([i for j in range(0, n_x_layers) for i in [j] * (j + 1)])
            * sqrt(tensor([3])),
            tensor(
                [
                    ia + ib * 2
                    for j in range(0, n_x_layers)
                    for ia, ib in zip([j] * (j + 1), range(-j, j + 1, 1))
                ]
            ),
        ],
        dim=1,
    )

    # sector_centers_px = sector_center_grid * radius_px * 2 + fov_dict["n pixels"] * 0.5
    sector_centers_mm = sector_center_grid * radius * 2
    shift_before = transform[:, 1:]
    sector_centers_mm = sector_centers_mm + shift_before.squeeze()
    transform = cat((transform[:, :1], tensor([0.0, 0.0]).unsqueeze(0)), dim=1)
    sector_centers_mm = transform_to_positions_2d_batch(
        transform,
        sector_centers_mm,
    ).view(-1, 2)
    boundaries = (
        arange(-int(fov_dict["n pixels"][0] / 2), int(fov_dict["n pixels"][0] / 2))
        * fov_dict["mm per pixel"][0]
        + fov_dict["center coordinates in mm"][0]
    )
    sector_centers_px = bucketize(sector_centers_mm, boundaries)
    return sector_centers_mm, sector_centers_px


def hot_rods_add_sector(
    phantom: Tensor,
    n_x_layers: int,
    radius: float,
    transform: Tensor,
    fov_dict: Dict[str, Tensor],
):
    """
    Add a sector to the phantom.

    Parameters
    ----------

    phantom : torch.Tensor
        The phantom to which the sector will be added.

    n_x_layers : int
        Number of layers in the x-direction.

    radius : float
        Radius of the rods in millimeters.

    transform : torch.Tensor
        Transformation to apply to the sector centers.
        This should be a 2D tensor of shape (n, 3) including translation and rotation.

    fov_dict : Dict[str, Tensor]
        Dictionary containing the field of view parameters.
        It should include 'mm per pixel' and 'fov size'.

    Returns
    -------

    sector_centers_mm : torch.Tensor
        Centers of the sectors in millimeters.

    sector_centers_px : torch.Tensor
        Centers of the sectors in pixels.
    """
    pxs_per_mm = int(1.0 / fov_dict["mm per pixel"][0])
    radius_px = int(radius * pxs_per_mm)

    sector_centers_mm, sector_centers_px = hot_rods_sector_centers(
        n_x_layers, radius, transform, fov_dict
    )
    disk = single_disk(radius_px)
    for center_px in sector_centers_px:
        # print(int(center_px[0] - radius_px), int(center_px[0] + radius_px))
        phantom[
            int(center_px[0] - radius_px) : int(center_px[0] + radius_px),
            int(center_px[1] - radius_px) : int(center_px[1] + radius_px),
        ] += disk
    return sector_centers_mm, sector_centers_px

def fov_tensor_dict(
    n_pixels: Sequence[int] = (512, 512),
    size_in_mm: Sequence[float] = (128, 128),
    center_coordinates: Sequence[float] = (0.0, 0.0),
    n_subdivisions: Sequence[int] = (1, 1),
) -> dict:
    """
    Create a dictionary with the FOV information.
    """
    fov_dict = {
        "n pixels": tensor(n_pixels),
        "size in mm": tensor(size_in_mm),
        "center coordinates in mm": tensor(center_coordinates),
    }
    fov_dict["mm per pixel"] = fov_dict["size in mm"] / fov_dict["n pixels"]
    fov_dict["n subdivisions"] = tensor(n_subdivisions)
    return fov_dict


def high_res_disk_mask(radius_px: int) -> Tensor:
    """
    Create a high-resolution disk mask.
    """
    epsilon = 1e-6
    mask = zeros((radius_px * 2, radius_px * 2), dtype=torch_float32)
    pixel_grid = stack(
        meshgrid(
            arange(radius_px * 2 + 1),
            arange(radius_px * 2 + 1),
            indexing="ij",
        ),
        dim=-1,
    )
    pixel_centroids = (pixel_grid[:-1, :-1].float() + 0.5) - radius_px
    pixel_four_corners = (
        stack(
            (
                pixel_grid[:-1:, :-1],
                pixel_grid[1:, :-1],
                pixel_grid[1:, 1:],
                pixel_grid[:-1, 1:],
            ),
            dim=-2,
        ).view(-1, 4, 2)
        - radius_px
    )
    pixel_five_points = cat((pixel_four_corners, pixel_centroids.view(-1, 1, 2)), dim=1)

    distance_from_center = norm(pixel_five_points, dim=-1)
    mask.view(-1)[distance_from_center[:, 4] < radius_px + epsilon] = 1.0
    return mask


def down_sample_disk_mask(mask: Tensor, downsampling_factor: int) -> Tensor:
    """
    Down sample a disk mask by a given factor.
    """
    return conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        ones(
            (1, 1, downsampling_factor, downsampling_factor),
            dtype=torch_float32,
        )
        / (downsampling_factor**2),
        padding=0,
        stride=downsampling_factor,
    ).squeeze()


def single_disk(radius_px: int, factor: int = 16) -> Tensor:
    """
    Create a single disk mask with a given radius.
    """

    mask = high_res_disk_mask(radius_px * factor)
    return down_sample_disk_mask(mask, factor)
