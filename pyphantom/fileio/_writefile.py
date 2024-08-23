"""
writefile.py
============

Write to a file

:author: Fang Han
:date: 2024-08-07
:contact: fhan0904@gmail.com

Provide the following functions:

- save_phantom_npz(phantom, fname)
  Save the phantom data and mask to a compressed numpy npz file

- save_phantom_png(phantom, fname)
  Plot the phantom image and mask onto a png file

Read more in the documentation at
https://spebt.github.io/pyphantom/

"""

__all__ = ["save_phantom_npz", "save_phantom_png", "save_phantom_all"]

from pyphantom.phantoms import phantom
from ._utils import check_outdir


def save_phantom_png(phantom: phantom, fname: str):
    from matplotlib.pyplot import subplots
    from matplotlib.colors import Normalize
    from matplotlib import colormaps

    fig, axs = subplots(1, 2, figsize=(22, 10), dpi=100)
    norm = Normalize(vmin=0, vmax=10)
    imshow_0 = axs[0].imshow(
        phantom.image.T,
        origin="lower",
        cmap=colormaps["gray"],
        norm=norm,
        aspect="equal",
    )
    axs[1].imshow(
        phantom.mask.T,
        origin="lower",
        cmap=colormaps["gray"],
        norm=norm,
        aspect="equal",
    )
    axs[0].set_title(f"{phantom.typename} phantom image")
    axs[1].set_title(f"{phantom.typename} phantom mask")
    fig.colorbar(imshow_0, ax=axs[:], shrink=0.6)
    # Output
    fig.savefig(fname, dpi=100)


def save_phantom_npz(phantom: phantom, fname: str) -> None:
    from numpy import savez_compressed

    mydict = {
        "phantom type": phantom.typename,
        "phantom": phantom.image,
        "mask": phantom.mask,
        "shape": phantom.shape,
    }
    savez_compressed(fname, **mydict)


def save_phantom_all(phantom, outdir) -> None:
    try:
        check_outdir(outdir)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    import os

    dim_str = f"{phantom.shape[0]}x{phantom.shape[1]}"
    file_extentions = ["npz", "png"]
    file_datatypes = ["data", "plot"]
    for ftype, fext in zip(file_datatypes, file_extentions):
        outf_fname = f"{phantom.typename}_phantom_{ftype}_{dim_str}.{fext}"
        outf_fullpath = os.path.join(outdir, outf_fname)
        print(f"Save phantom {ftype} to {outf_fullpath}")
        globals()[f"save_phantom_{fext}"](phantom, outf_fullpath)
