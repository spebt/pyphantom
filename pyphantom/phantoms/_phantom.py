"""
_phantom.py
===========

This module defines the `phantom` class which represents a phantom object used in imaging systems.

Classes:
--------
    phantom
        Represents a phantom object used in imaging systems.


:author: Fang Han
:contact: fhan0904@gmail.com

"""

__all__ = ["_phantom"]


class _phantom:
    import numpy

    def __init__(self, typename, img: numpy.ndarray, mask: numpy.ndarray) -> None:
        assert img.shape == mask.shape, "Image and mask must have the same shape"
        self.typename = typename
        self.image = img
        self.mask = mask
        self.shape = img.shape

    def __str__(self) -> str:
        return f"{self.typename} {self.shape}"
