__all__ = ["plot_phantom"]

from phantom import phantom


def check_outdir(outdir: str):
    import os

    if not os.path.exists(outdir):
        try:
            os.mkdir(outdir)
        except Exception as err:
            print(err)
            raise err


def plot_phantom(phantom: phantom, fname):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(11, 10), dpi=100)
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    fig.colorbar(
        ax.imshow(
            img.T, origin="lower", cmap=mpl.colormaps["gray"], norm=norm, aspect="equal"
        ),
        pad=0.01,
    )

    fig.tight_layout()
    # Output
    outDir = "output/"
    outImgName = "hotrod_phantom_plot_%dx%d.png" % (phantom.shape[0], phantom.shape[1])
    print("Save phantom plot to", outDir + fname)
    fig.savefig(outDir + fname, dpi=100)


def save_phantom_npz(phantom, mask, fname):
    import numpy as np

    print("Save phantom data to", outDir + outNpzName)
    np.savez_compressed(outDir + outNpzName, phantom=img, mask=mask)
    outNpzName = "hotrod_phantom_data_%dx%d.npz" % (phantom.shape[0], phantom.shape[1])
