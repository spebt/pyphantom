# create_contrast_phantom.py
import argparse
import math
import torch
from torch import save as torch_save

from _disk_shape import fov_tensor_dict
from contrast_phantom import build_contrast_phantom_grid, compute_cnr


# ---------- helpers ----------
def autofit_spacing_to_circle(diameters_mm, n_cols, n_rows, R_mm, margin_mm=1.0) -> float:
    """
    Largest equal spacing (mm) for a centered n_rows x n_cols grid so that ALL
    circles with per-column diameters 'diameters_mm' lie inside a circle of radius R_mm.
    """
    r_max = max(d / 2.0 for d in diameters_mm)
    denom = math.sqrt((n_cols - 1) ** 2 + (n_rows - 1) ** 2)
    if denom == 0:
        return 0.0
    d = 2.0 * (R_mm - margin_mm - r_max) / denom
    return max(0.0, d)


def grid_half_extents(n_cols, n_rows, dx, dy):
    """Return (half_width_mm, half_height_mm) for a centered grid with spacing dx, dy."""
    return ( (n_cols - 1) * dx / 2.0, (n_rows - 1) * abs(dy) / 2.0 )


def check_all_inside(diams, ncols, nrows, start_xy, spacing_xy, R, margin) -> bool:
    """Verify every circle fits: hypot(x,y) + r <= R - margin."""
    sx, sy = start_xy
    dx, dy = spacing_xy
    dy = -abs(dy)  # rows go top->bottom
    for i in range(nrows):
        y = sy + i * dy
        for j in range(ncols):
            x = sx + j * dx
            r = diams[j] * 0.5
            if math.hypot(x, y) + r > R - margin + 1e-6:
                return False
    return True
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Create a contrast phantom (.pt) with grid layout.")
    # FOV
    p.add_argument("--size-mm", type=float, nargs=2, default=(70.0, 70.0), help="FOV size mm: Sx Sy")
    p.add_argument("--px-mm",   type=float, nargs=2, default=(0.25, 0.25), help="Pixel size mm: dx dy")

    # Grid (rows = contrast ratios top->bottom, cols = lesion diameters)
    p.add_argument("--diameters-mm",  type=float, nargs="+", default=[3,4,5,6,7], help="Lesion diameters per column (mm)")
    p.add_argument("--contrast-rows", type=float, nargs="+", default=[25,20,15,10,5], help="C_lesion/C_bg per row (top→bottom)")

    # Placement controls
    p.add_argument("--spacing-mm", type=float, nargs=2, default=(9.5, 9.5), help="Grid spacing (dx dy) mm")
    p.add_argument("--center-grid", action="store_true", help="Center the grid in the FOV")
    p.add_argument("--start-mm",   type=float, nargs=2, default=(0.0, 0.0), help="Top-left center (x,y) mm if not centering")

    # Auto-fit into circle
    p.add_argument("--fit-to-circle", action="store_true", help="Auto center + choose spacing to fit inside circle")
    p.add_argument("--body-radius-mm", type=float, default=None, help="Circle radius mm (default 0.45 * min(size_mm))")
    p.add_argument("--margin-mm", type=float, default=1.0, help="Safety margin inside circle (mm) when fitting")

    # Values & ROIs
    p.add_argument("--bg-value", type=float, default=1.0, help="Background baseline value")
    p.add_argument("--edge-exclude-px",   type=int, default=1, help="Exclude px from lesion edge for hot ROI")
    p.add_argument("--border-exclude-px", type=int, default=3, help="Exclude border band from background ROI")
    p.add_argument("--ss-factor", type=int, default=16, help="Supersampling factor for anti-aliased disks")

    # Output
    p.add_argument("--out-prefix", type=str, default="contrast_phantom", help="Output filename prefix")
    return p.parse_args()


def main():
    args = parse_args()

    # --- FOV dict (same convention as your hot-rods scripts) ---
    n_px_x = int(round(args.size_mm[0] / args.px_mm[0]))
    n_px_y = int(round(args.size_mm[1] / args.px_mm[1]))
    fov = fov_tensor_dict(
        n_pixels=(n_px_x, n_px_y),
        size_in_mm=(args.size_mm[0], args.size_mm[1]),
        center_coordinates=(0.0, 0.0),
    )

    n_cols = len(args.diameters_mm)
    n_rows = len(args.contrast_rows)

    # --- placement: either fit-to-circle, or center/start with given spacing ---
    if args.fit_to_circle:
        R = args.body_radius_mm if args.body_radius_mm is not None else 0.45 * min(args.size_mm)
        d = autofit_spacing_to_circle(args.diameters_mm, n_cols, n_rows, R_mm=R, margin_mm=args.margin_mm)
        dx = dy = d
        # center grid, rows top->bottom (negative dy when used)
        half_w, half_h = grid_half_extents(n_cols, n_rows, dx, dy)
        start_xy_mm = (-half_w, +half_h)
        spacing_mm = (dx, -abs(dy))
        assert check_all_inside(args.diameters_mm, n_cols, n_rows, start_xy_mm, spacing_mm, R, args.margin_mm), \
            "Auto-fit failed—reduce diameters or margin."
    else:
        dx, dy = args.spacing_mm
        if args.center_grid:
            half_w, half_h = grid_half_extents(n_cols, n_rows, dx, dy)
            start_xy_mm = (-half_w, +half_h)
        else:
            start_xy_mm = (args.start_mm[0], args.start_mm[1])
        spacing_mm = (dx, -abs(dy))  # enforce top->bottom rows

    # --- build phantom ---
    pack = build_contrast_phantom_grid(
        fov_dict=fov,
        diameters_mm=args.diameters_mm,
        contrast_ratios=args.contrast_rows,
        start_xy_mm=start_xy_mm,
        spacing_mm=spacing_mm,
        bg_value=args.bg_value,
        edge_exclude_px=args.edge_exclude_px,
        border_exclude_px=args.border_exclude_px,
        supersample_factor=args.ss_factor,
    )

    phantom = pack["phantom"]
    lesions = pack["lesions"]
    mask_bg = pack["mask_bg"]

    # quick CNR check on phantom values
    cnr_on_phantom = compute_cnr(phantom, lesions, mask_bg)

    out_dict = {
        "Description": "Contrast phantom (grid layout: rows=contrast top→bottom, cols=diameter)",
        "Metadata": {
            "size in mm": fov["size in mm"].tolist(),
            "mm per pixel": fov["mm per pixel"].tolist(),
            "n pixels": fov["n pixels"].tolist(),
            "center coordinates in mm": fov["center coordinates in mm"].tolist(),
            "diameters in mm (columns)": list(map(float, args.diameters_mm)),
            "contrast ratios (rows top→bottom)": list(map(float, args.contrast_rows)),
            "grid start xy in mm": list(map(float, start_xy_mm)),
            "grid spacing in mm": list(map(float, spacing_mm)),
            "bg value": float(args.bg_value),
            "edge exclude px (hot ROI)": int(args.edge_exclude_px),
            "border exclude px (bg ROI)": int(args.border_exclude_px),
            "fit_to_circle": bool(args.fit_to_circle),
            "body_radius_mm": float(args.body_radius_mm) if args.body_radius_mm is not None else None,
            "margin_mm": float(args.margin_mm),
        },
        "Phantom tensor": phantom,
        "Lesions": lesions,
        "Background mask": mask_bg,
        "Phantom shape": phantom.shape,
        "Phantom dtype": phantom.dtype,
        "CNR (on phantom values)": cnr_on_phantom,
    }

    pt_name = f"{args.out_prefix}_{int(args.size_mm[0])}x{int(args.size_mm[1])}mm_{n_px_x}x{n_px_y}px.pt"
    torch_save(out_dict, pt_name)
    print(f"Saved: {pt_name}")


if __name__ == "__main__":
    main()
