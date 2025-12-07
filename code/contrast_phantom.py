# contrast_phantom.py
from typing import Dict, List, Sequence, Tuple, Optional
import math

import torch
from torch import Tensor
from torch import float32 as torch_float32
from torch import int64 as torch_int64
from torch.nn.functional import conv2d

# Reuse your utilities
from _disk_shape import fov_tensor_dict, single_disk, hot_rods_sector_centers

# -----------------------------
# Helpers: mm<->px, safe slicing
# -----------------------------
def mm_to_px_scalar(mm: float, mm_per_px: float) -> int:
    return int(round(mm / mm_per_px))

#def mm_to_px_vec2(v_mm: Tensor, mm_per_px_xy: Tensor) -> Tensor:
    # returns int64 pixel indices
#    return torch.round(v_mm / mm_per_px_xy).to(torch_int64)
def mm_to_px_vec2(v_mm: Tensor, mm_per_px_xy: Tensor, n_pixels_xy: Tensor) -> Tensor:
    """
    Convert mm coords (centered at (0,0)) to pixel indices in [0..N-1]^2.
    """
    return torch.round(
        v_mm / mm_per_px_xy + n_pixels_xy.to(v_mm.dtype) * 0.5
    ).to(torch_int64)

def safe_add_patch(img: torch.Tensor, patch: torch.Tensor, cx: int, cy: int) -> None:
    """
    Add 'patch' (ph x pw) into 'img' (H x W) centered at (cx, cy).
    Uses EXCLUSIVE end indices to avoid off-by-one with even-sized patches.
    Handles boundary clipping.
    """
    H, W = img.shape
    ph, pw = patch.shape

    # top-left of patch if perfectly centered on (cx, cy)
    x0 = cx - ph // 2
    y0 = cy - pw // 2
    x1 = x0 + ph   # exclusive
    y1 = y0 + pw   # exclusive

    # clip to image bounds (exclusive ends)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(H, x1)
    sy1 = min(W, y1)

    # nothing to draw
    if sx0 >= sx1 or sy0 >= sy1:
        return

    # corresponding patch region
    px0 = sx0 - x0
    py0 = sy0 - y0
    px1 = px0 + (sx1 - sx0)
    py1 = py0 + (sy1 - sy0)

    img[sx0:sx1, sy0:sy1] += patch[px0:px1, py0:py1]



# ------------------------------------------
# ROI builders: lesion (edge-excluded) & bg
# ------------------------------------------
def make_circular_roi_mask(
    shape_hw: Sequence[int],
    center_xy_px: Tuple[int, int],
    radius_px: int,
    edge_exclude_px: int = 0,
) -> Tensor:
    """
    Binary mask for a circular ROI. If edge_exclude_px>0, shrink radius accordingly.
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    cx, cy = int(center_xy_px[0]), int(center_xy_px[1])
    r = max(radius_px - edge_exclude_px, 0)

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch_float32),
        torch.arange(W, dtype=torch_float32),
        indexing="ij",
    )
    dist2 = (xx - cy) ** 2 + (yy - cx) ** 2
    mask = (dist2 <= (r * r)).to(torch_float32)
    return mask


def make_background_mask(
    shape_hw: Sequence[int],
    lesion_masks: List[Tensor],
    border_exclude_px: int = 2,
) -> Tensor:
    """
    Background ROI: all voxels except lesions (optionally dilated by 1 px to be safe)
    and except a border band 'border_exclude_px'.
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    bg = torch.ones((H, W), dtype=torch_float32)

    # exclude lesions (+1 px dilate)
    if lesion_masks:
        kernel = torch.ones((1, 1, 3, 3), dtype=torch_float32)
        accum = torch.zeros_like(bg).unsqueeze(0).unsqueeze(0)
        for m in lesion_masks:
            accum += m.unsqueeze(0).unsqueeze(0)
        dil = conv2d(accum, kernel, padding=1).squeeze()
        bg[dil > 0.0] = 0.0

    # exclude border
    if border_exclude_px > 0:
        bg[:border_exclude_px, :] = 0.0
        bg[-border_exclude_px:, :] = 0.0
        bg[:, :border_exclude_px] = 0.0
        bg[:, -border_exclude_px:] = 0.0

    return bg


# ------------------------------------------
# Core: build a contrast phantom (grid mode)
# ------------------------------------------
def build_contrast_phantom_grid(
    fov_dict: Dict[str, Tensor],
    diameters_mm: Sequence[float],              # columns
    contrast_ratios: Sequence[float],           # rows; ratio = C_lesion / C_bg
    start_xy_mm: Tuple[float, float],           # top-left lesion center in mm (x,y)
    spacing_mm: Tuple[float, float],            # delta between centers (dx, dy) in mm
    bg_value: float = 1.0,                      # warm background baseline
    edge_exclude_px: int = 1,                   # exclude 1-px ring for hot ROI
    border_exclude_px: int = 3,                 # exclude near edges for bg ROI
    supersample_factor: int = 16,               # anti-alias factor for disks
) -> Dict:
    """
    Returns:
      {
        "phantom": (H,W) float32
        "lesions": [
            {"center_px": (x,y), "radius_px": r, "diameter_mm": d, "contrast": rC, "mask_hot": HxW},
            ...
        ]
        "mask_bg": HxW
        "metadata": {...}
      }
    """
    H, W = map(int, fov_dict["n pixels"].tolist())
    mm_per_px = fov_dict["mm per pixel"]          # (mx, my)
    mm_per_px_xy = mm_per_px.clone()
    n_pixels_xy  = fov_dict["n pixels"].clone()
    phantom = torch.full((H, W), float(bg_value), dtype=torch_float32)

    # cache disks by radius_px
    disk_cache: Dict[int, Tensor] = {}

    lesions_info: List[Dict] = []

    # grid iteration: rows = contrasts, cols = diameters
    for r_idx, cr in enumerate(contrast_ratios):
        for c_idx, d_mm in enumerate(diameters_mm):
            radius_mm = d_mm * 0.5

            center_mm_x = start_xy_mm[0] + c_idx * spacing_mm[0]
            center_mm_y = start_xy_mm[1] + r_idx * spacing_mm[1]
            center_mm = torch.tensor([center_mm_x, center_mm_y], dtype=torch_float32)

            #center_px = mm_to_px_vec2(center_mm, mm_per_px_xy)
            center_px = mm_to_px_vec2(center_mm, mm_per_px_xy, n_pixels_xy)
            cx, cy = int(center_px[0].item()), int(center_px[1].item())

            radius_px = max(1, mm_to_px_scalar(radius_mm, float(mm_per_px_xy[0])))

            if radius_px not in disk_cache:
                disk_cache[radius_px] = single_disk(radius_px, factor=supersample_factor)

            lesion_value = float(cr) * float(bg_value)  # C_lesion = r * C_bg
            patch = disk_cache[radius_px] * (lesion_value - bg_value)  # delta on top of bg

            safe_add_patch(phantom, patch, cx, cy)

            # lesion ROI (edge excluded)
            mask_hot = make_circular_roi_mask((H, W), (cx, cy), radius_px, edge_exclude_px)

            lesions_info.append(
                {
                    "center_px": (cx, cy),
                    "radius_px": radius_px,
                    "diameter_mm": float(d_mm),
                    "contrast_ratio": float(cr),
                    "mask_hot": mask_hot,
                }
            )

    # background ROI (exclude lesions + edge)
    mask_bg = make_background_mask((H, W), [li["mask_hot"] for li in lesions_info], border_exclude_px)

    return {
        "phantom": phantom,
        "lesions": lesions_info,
        "mask_bg": mask_bg,
        "metadata": {
            "layout": "grid",
            "diameters_mm": list(map(float, diameters_mm)),
            "contrast_ratios": list(map(float, contrast_ratios)),
            "start_xy_mm": tuple(map(float, start_xy_mm)),
            "spacing_mm": tuple(map(float, spacing_mm)),
            "bg_value": float(bg_value),
            "mm_per_pixel": fov_dict["mm per pixel"].tolist(),
            "size_in_mm": fov_dict["size in mm"].tolist(),
            "n_pixels": fov_dict["n pixels"].tolist(),
        },
    }


# -------------------------------------------------------
# Alternative: sector/hex layout with per-sector contrast
# -------------------------------------------------------
def build_contrast_phantom_sectors(
    fov_dict: Dict[str, Tensor],
    sector_params: Sequence[Dict],
    bg_value: float = 1.0,
    edge_exclude_px: int = 1,
    border_exclude_px: int = 3,
    supersample_factor: int = 16,
) -> Dict:
    """
    sector_params: list of dicts, each like
      {
        "n_x_layers": 7,
        "diameter_mm": 6.0,
        "contrast_ratio": 5.0,            # C_lesion / C_bg
        "transform": [theta, tx, ty],     # radians, mm, mm (same convention you used)
      }
    """
    H, W = map(int, fov_dict["n pixels"].tolist())
    mm_per_px = fov_dict["mm per pixel"]
    mm_per_px_xy = mm_per_px.clone()
    n_pixels_xy  = fov_dict["n pixels"].clone()

    phantom = torch.full((H, W), float(bg_value), dtype=torch_float32)
    lesions_info: List[Dict] = []
    disk_cache: Dict[int, Tensor] = {}

    for sp in sector_params:
        n_layers = int(sp["n_x_layers"])
        d_mm = float(sp["diameter_mm"])
        r_mm = 0.5 * d_mm
        cr = float(sp["contrast_ratio"])
        transform = torch.tensor(sp["transform"], dtype=torch_float32).unsqueeze(0)  # (1,3)

        # get centers (reuses your current transform semantics)
        centers_mm, centers_px_bucket = hot_rods_sector_centers(
            n_layers, r_mm, transform, fov_dict
        )

        # compute pixel centers via mm directly (more robust than bucket edges)
        #centers_px = mm_to_px_vec2(centers_mm, mm_per_px_xy)
        centers_px = mm_to_px_vec2(centers_mm, mm_per_px_xy, n_pixels_xy)

        radius_px = max(1, mm_to_px_scalar(r_mm, float(mm_per_px_xy[0])))
        if radius_px not in disk_cache:
            disk_cache[radius_px] = single_disk(radius_px, factor=supersample_factor)

        delta_value = (cr * bg_value) - bg_value
        patch = disk_cache[radius_px] * float(delta_value)

        for k in range(centers_px.shape[0]):
            cx, cy = int(centers_px[k, 0].item()), int(centers_px[k, 1].item())
            safe_add_patch(phantom, patch, cx, cy)

            mask_hot = make_circular_roi_mask((H, W), (cx, cy), radius_px, edge_exclude_px)
            lesions_info.append(
                {
                    "center_px": (cx, cy),
                    "radius_px": radius_px,
                    "diameter_mm": d_mm,
                    "contrast_ratio": cr,
                    "mask_hot": mask_hot,
                }
            )

    mask_bg = make_background_mask((H, W), [li["mask_hot"] for li in lesions_info], border_exclude_px)

    return {
        "phantom": phantom,
        "lesions": lesions_info,
        "mask_bg": mask_bg,
        "metadata": {
            "layout": "sectors_hex",
            "sector_params": sector_params,
            "bg_value": float(bg_value),
            "mm_per_pixel": fov_dict["mm per pixel"].tolist(),
            "size_in_mm": fov_dict["size in mm"].tolist(),
            "n_pixels": fov_dict["n pixels"].tolist(),
        },
    }


# --------------------------------
# CNR computation helper (Eq. 8)
# --------------------------------
def compute_cnr(
    img: Tensor,                          # reconstructed image or this phantom
    lesions: List[Dict],                  # from build_* return
    mask_bg: Tensor,                      # from build_* return
) -> List[Tuple[int, float]]:
    """
    Returns list of (k, CNR_k) with k = lesion index.
    Eq. (8): CNR_k = (C_k / C_bg - 1) / (SD_bg / C_bg)
    """
    bg_vals = img[mask_bg > 0.5]
    Cbg = float(bg_vals.mean().item()) if bg_vals.numel() else float("nan")
    SDbg = float(bg_vals.std(unbiased=True).item()) if bg_vals.numel() > 1 else float("nan")

    out = []
    for k, li in enumerate(lesions):
        hk = img[li["mask_hot"] > 0.5]
        Ck = float(hk.mean().item()) if hk.numel() else float("nan")
        cnr_k = ((Ck / Cbg) - 1.0) / (SDbg / Cbg) if (not math.isnan(Cbg) and SDbg > 0.0) else float("nan")
        out.append((k, cnr_k))
    return out
