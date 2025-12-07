# plot_ring_hotrods_phantom.py
import os, sys
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_ring_hotrods_phantom.py <phantom.pt> [--show-ring]")
        sys.exit(1)

    in_filename = sys.argv[1]
    assert os.path.isfile(in_filename), f"File not found: {in_filename}"
    show_ring = ("--show-ring" in sys.argv)

    data = torch.load(in_filename, map_location="cpu")
    img = data["Phantom tensor"].float()
    meta = data["Metadata"]

    size_mm = meta["size_in_mm"]
    n_pixels = meta["n_pixels"]
    mm_per_px = meta["mm_per_pixel"]

    vmin = float(meta.get("bg_counts", 5.0))
    vmax = float(meta.get("rod_counts", 20.0))
    norm = PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax)

    plt.close("all")
    fig, ax = plt.subplots(dpi=180, figsize=(8, 8))
    fig.patch.set_facecolor("#0f0606")
    ax.set_facecolor("#0f0606")

    ax.imshow(
        img.T.numpy(),
        cmap="inferno",
        norm=norm,
        extent=(-size_mm[0]/2, size_mm[0]/2, -size_mm[1]/2, size_mm[1]/2),
        origin="lower",
        aspect="equal",
        interpolation="none",
    )

    if show_ring:
        Rin = meta["ring_inner_radius_mm"]
        Rout = meta["ring_outer_radius_mm"]
        ax.add_patch(Circle((0,0), Rout, fill=False, ec="red", lw=1.5))
        ax.add_patch(Circle((0,0), Rin,  fill=False, ec="red", lw=1.0))

    ax.set_title(f"Ring Phantom ({meta['n_rods']} Rods @ {meta['rod_diameter_mm']:.1f} mm Diameter)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    out_png = os.path.splitext(os.path.basename(in_filename))[0] + ".png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    print(f"Saved: {out_png}")
