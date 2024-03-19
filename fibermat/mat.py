#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧵 Mat(erial)
-------------

Classes
-------
Mat :
    A class to describe a material made up of a set of random straight fibers.

"""

import numpy as np
import pandas as pd


class Mat(pd.DataFrame):
    """
    A class to describe a material made up of a set of random straight fibers.
    It defines:
        - the geometry of the fibers
        - the material properties
        - the initial configuration (positions and orientations)

    Parameters
    ----------
    n : int
        Number of fibers. Default is 0.
    length : float
        Fiber length (mm). Default is 25 mm.
    width : float
        Fiber width (mm). Default is 1 mm.
    thickness : float
        Fiber thickness (mm). Default is 1 mm.
    size : float
        Box dimensions (mm). Default is 50 mm.
    theta : float
        In plane angle (rad). Default is π rad.
    psi : float
        Out-of-plane angle (rad). Default is 0 rad.
    shear : float
        Shear modulus (MPa). Default is 1 MPa.
    tensile : float
        Tensile modulus (MPa). Default is ∞ MPa.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    **kwargs :
        Additional keyword arguments ignored by the function.

    Attributes
    ----------
    attrs : dictionary
        Global attributes:
            - n : int, Number of fibers.
            - size : float, Box dimensions (mm).
    index : pandas.Index
        Fiber label.
    l : pandas.Series
        Fiber length (mm).
    b : pandas.Series
        Fiber width (mm).
    h : pandas.Series
        Fiber thickness (mm).
    x : pandas.Series
        Fiber position: X-coordinate (mm).
    y : pandas.Series
        Fiber position: Y-coordinate (mm).
    z : pandas.Series
        Fiber position: Z-coordinate (mm).
    u : pandas.Series
        Fiber orientation: X-component.
    v : pandas.Series
        Fiber orientation: Y-component.
    w : pandas.Series
        Fiber orientation: Z-component.
    G : pandas.Series
        Shear modulus (MPa).
    E : pandas.Series
        Tensile modulus (MPa).

    Methods
    -------
    init()
        Generate a set of random straight fibers.
    check()
        Check that `Mat` object is defined correctly.

    Properties
    ----------
    dimensions : pandas.DataFrame
        Fiber dimensions. Size: (N x 3).
    positions : pandas.DataFrame
        Fiber positions. Size: (N x 3).
    orientations : pandas.DataFrame
        Fiber orientations. Size: (N x 3).

    Examples
    --------
    ```python
        >>> # Generate a set of fibers
        >>> mat = Mat(**inputs)
        >>> # Get fiber data
        >>> dimensions = mat[[*"lbh"]]
        >>> positions = mat[[*"xyz"]]
        >>> orientations = mat[[*"uvw"]]

    ```

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the `Mat` object.

        Parameters
        ----------
        *args :
            Additional positional arguments passed to the constructor.
        **kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        Mat.init :
            Generate a set of random straight fibers.

        """
        if len(args) and isinstance(args[0], pd.DataFrame):
            # Initialize the DataFrame from argument
            super().__init__(*args, **kwargs)
            # Copy global attributes from argument
            self.attrs = args[0].attrs

        else:
            # Initialize the DataFrame from parameters
            self.__init__(Mat.init(*args, **kwargs))

        # Check `Mat` object
        Mat.check(self)

    # ~~~ Constructor ~~~ #

    @staticmethod
    def init(n=0, length=25., width=1., thickness=1., size=50.,
             theta=np.pi, psi=0., shear=1., tensile=np.inf, seed=0, **kwargs):
        """
        Generate a set of random straight fibers.

        Parameters
        ----------
        n : int
            Number of fibers. Default is 0.
        length : float
            Fiber length (mm). Default is 25 mm.
        width : float
            Fiber width (mm). Default is 1 mm.
        thickness : float
            Fiber thickness (mm). Default is 1 mm.
        size : float
            Box dimensions (mm). Default is 50 mm.
        theta : float
            In plane angle (rad). Default is π rad.
        psi : float
            Out-of-plane angle (rad). Default is 0 rad.
        shear : float
            Shear modulus (MPa). Default is 1 MPa.
        tensile : float
            Tensile modulus (MPa). Default is ∞ MPa.
        seed : int, optional
            Random seed for reproducibility. Default is 0.
        **kwargs :
            Additional keyword arguments ignored by the function.

        Returns
        -------
        mat : pandas.DataFrame
            Initialized `Mat` object.

        """
        # Random generation
        np.random.seed(seed)  # : random seed for reproducibility

        # Fiber dimensions
        l = np.full(n, length)  # : fiber length (mm)
        b = np.full(n, width)  # : fiber width (mm)
        h = np.full(n, thickness)  # : fiber thickness (mm)
        # Fiber position
        x = np.random.uniform(-0.5, 0.5, n) * size  # : X-coordinate (mm)
        y = np.random.uniform(-0.5, 0.5, n) * size  # : Y-coordinate (mm)
        z = np.random.uniform(-0.5, 0.5, n) * size  # : Z-coordinate (mm)
        z = np.sort(z)
        # Angles in radians
        p = np.random.uniform(-0.5, 0.5, n) * theta  # : in plane angle (rad)
        q = np.random.uniform(-0.5, 0.5, n) * psi  # : out-of-plane angle(rad)
        # Fiber orientation
        u = np.cos(p) * np.cos(q)  # : fiber orientation: X-component
        v = np.sin(p) * np.cos(q)  # : fiber orientation: Y-component
        w = np.sin(q)  # : fiber orientation: Z-component
        # Mechanical properties
        G = np.full(n, shear)  # : shear modulus (MPa)
        E = np.full(n, tensile)  # : tensile modulus (MPa)

        # Initialize mat DataFrame
        mat = pd.DataFrame(
            data=np.c_[l, b, h, x, y, z, u, v, w, G, E],
            columns=[*"lbhxyzuvwGE"]
        )

        # Set attributes
        mat.attrs["n"] = n
        mat.attrs["size"] = size

        # Return the `Mat` object
        return mat

    # ~~~ Public methods ~~~ #

    @staticmethod
    def check(mat=None):
        """
        Check that `Mat` object is defined correctly.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a `Mat` object.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of `Mat` object.
        AttributeError
            If any attributes are missing from the dictionary `mat.attrs`.
        IndexError
            if row indices are incorrectly defined:
                - Row indices are not unique in [0, ..., n-1] where n is the number of fibers.
                - Fiber labels are not sorted.
        ValueError
            If any of the following conditions are not met:
                - Dimensions are not positive.
                - Positions are not within a box of size specified in `mat.attrs`.
                - Orientation vectors do not have unit lengths.
                - Material properties are not positive.

        Returns
        -------
        mat : pandas.DataFrame
            Validated `Mat` object.

        Notes
        -----
        If `mat` is None, it returns an empty `Mat` object.

        """
        if mat is None:
            mat = Mat()

        # Keys
        try:
            mat[[*"lbhxyzuvwGE"]]
        except KeyError as e:
            raise KeyError(e)

        # Attributes
        if not ("n" in mat.attrs.keys()):
            raise AttributeError("'n' is not in attribute dictionary.")
        if not ("size" in mat.attrs.keys()):
            raise AttributeError("'size' is not in attribute dictionary.")

        # Indices
        if len(mat) != mat.attrs["n"]:
            raise ValueError("Attribute `n: {}` does not correspond to"
                             " the number of fibers ({})."
                             .format(mat.attrs["n"], len(mat)))
        if (len(np.unique(mat.index)) != len(mat)
                or not np.all(np.unique(mat.index) == np.arange(len(mat)))):
            raise IndexError("Row indices must be unique in [0,..., {}]."
                             .format(len(mat) - 1))
        if not np.all(mat.index == np.arange(len(mat))):
            raise IndexError("Fiber labels must be sorted.")

        # Data
        if not np.all(mat[[*"lbh"]] > 0):
            raise ValueError("Dimensions must be positive.")
        if not np.all((-0.5 * mat.attrs["size"] <= mat[[*"xy"]])
                      & (mat[[*"xy"]] <= 0.5 * mat.attrs["size"])):
            raise ValueError("Positions must be in a box of size {}"
                             " (between {} and {})."
                             .format(mat.attrs["size"],
                                     -0.5 * mat.attrs["size"],
                                     0.5 * mat.attrs["size"]))
        if not np.allclose(np.linalg.norm(mat[[*"uvw"]], axis=1), 1):
            raise ValueError("Orientation vectors must have unit lengths.")
        if not np.all(mat[[*"GE"]] > 0):
            raise ValueError("Material properties must be positive.")

        # Return the `Mat` object
        return mat


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    # Generate a set of fibers
    mat = Mat(10)

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    if len(mat):
        # Draw fibers
        for i in tqdm(range(len(mat))):
            fiber = mat.iloc[i]
            A = fiber[[*"xyz"]].values - 0.5 * fiber.l * fiber[[*"uvw"]].values
            B = fiber[[*"xyz"]].values + 0.5 * fiber.l * fiber[[*"uvw"]].values
            plt.plot(*np.c_[A, B])
        ax.set_xlim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
        ax.set_ylim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
    plt.show()
