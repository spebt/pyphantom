__all__ = ["save_phantom_npz", "save_phantom_png", "save_phantom_all"]

from phantom import phantom


def check_outdir(outdir: str):
    import os.path.exists as exists

    if not exists(outdir):
        raise FileNotFoundError("Output directory does not exist: %s" % outdir)
    else:
        return SUCCESS


def save_phantom_png(phantom: phantom, fname: str):
    import matplotlib.pyplot.subplots as subplots
    import matplotlib.color.Normalize as Normalize

    fig, axs = subplots(1, 2, figsize=(22, 10), dpi=100)
    norm = Normalize(vmin=0, vmax=10)
    imshow_0 = axs[0].imshow(
        phantom.image.T,
        origin="lower",
        cmap=mpl.colormaps["gray"],
        norm=norm,
        aspect="equal",
    )
    axs[1].imshow(
        phantom.mask.T,
        origin="lower",
        cmap=mpl.colormaps["gray"],
        norm=norm,
        aspect="equal",
    )
    axs[0].set_title(f"{phantom.typename} phantom image")
    axs[1].set_title(f"{phantom.typename} phantom mask")
    fig.colorbar(imshow_0, ax=axs[:], shrink=0.6)
    # Output
    fig.savefig(fname, dpi=100)


def save_phantom_npz(phantom: phantom, fname: str) -> None:
    import numpy.savez_compressed as savez_compressed

    mydict = {
        "phantom type": phantom.typename,
        "phantom": phantom.image,
        "mask": phantom.mask,
    }
    savez_compressed(fname, mydict)


def save_phantom_all(phantom, outdir) -> None:
    import os.path.join

    try:
        check_outdir(outdir)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    dim_str = f"{phantom.shape[0]}x{phantom.shape[1]}"
    file_extentions = ["npz", "png"]
    file_datatypes = ["data", "plot"]
    for ftype, fext in zip(file_datatypes, file_extentions):
        outf_fname = f"{phantom.typename}_phantom_{ftype}_{dim_str}.{fext}"
        outf_fullpath = os.path.join(outdir, outf_fname)
        if ftype == "npz":
            print("Save phantom data to", outf_fullpath)
            save_phantom_npz(phantom, outf_fullpath)
        elif ftype == "png":
            print("Save phantom plot to", outf_fullpath)
