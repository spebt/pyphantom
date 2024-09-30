"""
_utils
======

Utility functions for the package.
"""

__all__ = ["check_outdir"]


def check_outdir(outdir: str):
    import os

    if not os.path.exists(outdir):
        raise FileNotFoundError(
            f'Output directory "{outdir}" does not exist. Create it first.'
        )
