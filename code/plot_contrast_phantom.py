# plot_contrast_phantom.py
import os, sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

def circular_mask_xy(shape_hw, radius_px, center=None):
    H, W = shape_hw
    cx, cy = ((H - 1) / 2.0, (W - 1) / 2.0) if center is None else center
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rr2 = (yy - cx) ** 2 + (xx - cy) ** 2
    return rr2 <= (radius_px ** 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_contrast_phantom.py <phantom.pt> [--body-radius-mm 31.5] [--no-body-mask] [--show-centers]")
        sys.exit(1)

    in_filename = sys.argv[1]
    assert os.path.isfile(in_filename), f"File not found: {in_filename}"

    body_radius_mm = None
    use_body_mask = True
    show_centers = ("--show-centers" in sys.argv)
    if "--body-radius-mm" in sys.argv:
        i = sys.argv.index("--body-radius-mm")
        body_radius_mm = float(sys.argv[i + 1])
    if "--no-body-mask" in sys.argv:
        use_body_mask = False

    data = torch.load(in_filename, map_location="cpu")
    phantom = data["Phantom tensor"].float()
    meta = data["Metadata"]

    size_mm = meta["size in mm"]
    n_pixels = meta["n pixels"]
    mm_per_px = meta["mm per pixel"]
    bg_val = float(meta.get("bg value", 1.0))

    vmax = float(phantom.max().item())
    vmin = max(0.0, bg_val * 0.8)
    norm = PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax)

    img_np = phantom.numpy()
    if use_body_mask:
        if body_radius_mm is None:
            body_radius_mm = 0.45 * min(size_mm)
        radius_px = int(round(body_radius_mm / mm_per_px[0]))  # assume square pixels
        mask = circular_mask_xy(phantom.shape, radius_px)
        img_np = img_np.copy()
        img_np[~mask] = vmin

    plt.close("all")
    fig, ax = plt.subplots(dpi=180, figsize=(8, 8))
    fig.patch.set_facecolor("#0f0606")
    ax.set_facecolor("#0f0606")

    ax.imshow(
        img_np.T,
        cmap="inferno",          # warm reddish map
        norm=norm,               # vmin/vmax embedded in norm
        extent=(-size_mm[0]/2, size_mm[0]/2, -size_mm[1]/2, size_mm[1]/2),
        origin="lower",
        aspect="equal",
        interpolation="none",
    )

    if show_centers:
        for li in data.get("Lesions", []):
            (cx, cy) = li["center_px"]
            ax.plot(
                (cx - n_pixels[0] / 2) * mm_per_px[0],
                (cy - n_pixels[1] / 2) * mm_per_px[1],
                "wo", ms=2, alpha=0.7,
            )

    ax.set_xlim(-size_mm[0]/2, size_mm[0]/2)
    ax.set_ylim(-size_mm[1]/2, size_mm[1]/2)
    ax.axis("off")
    out_png = os.path.splitext(os.path.basename(in_filename))[0] + ".png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    print(f"Saved: {out_png}")
