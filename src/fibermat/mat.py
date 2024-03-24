#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


class Mat(pd.DataFrame):
    """
    A class inherited from pandas.DataFrame_ to **describe a fibrous material** made up of a set of random fibers. It defines:

        - the geometry of the straight fibers.
        - the initial configuration (positions and orientations).
        - the material properties.

    Parameters
    ----------
    n : int, optional
        Number of fibers. Default is 0.
    length : float, optional
        Fiber length (mm). Default is 25 mm.
    width : float, optional
        Fiber width (mm). Default is 1 mm.
    thickness : float, optional
        Fiber thickness (mm). Default is 1 mm.
    size : float, optional
        Box dimensions (mm). Default is 50 mm.
    theta : float, optional
        In plane angle (rad). Default is π rad.
    psi : float, optional
        Out-of-plane angle (rad). Default is 0 rad.
    shear : float, optional
        Shear modulus (MPa). Default is 1 MPa.
    tensile : float, optional
        Tensile modulus (MPa). Default is ∞ MPa.
    seed : int, optional
        Random seed for reproducibility. Default is 0.

    .. NOTE::
        The constructor calls :meth:`init` method if the object is instantiated with parameters. Otherwise, initialization is performed with the pandas.DataFrame_ constructor.

    .. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    :Use:

        >>> # Generate a set of fibers
        >>> mat = Mat(100)
        >>> mat
               l    b    h          x          y          z         u         v    w    G    E
        0   25.0  1.0  1.0   2.440675   8.890827 -24.338157  0.289366  0.957218 -0.0  1.0  inf
        1   25.0  1.0  1.0  10.759468 -11.499601 -24.178519  0.651721  0.758459  0.0  1.0  inf
        2   25.0  1.0  1.0   5.138169  11.759701 -23.967450  0.865730 -0.500512 -0.0  1.0  inf
        3   25.0  1.0  1.0   2.244159  23.109427 -23.766064  0.252040 -0.967717  0.0  1.0  inf
        4   25.0  1.0  1.0  -3.817260 -12.562343 -23.716864  0.957840 -0.287303  0.0  1.0  inf
        ..   ...  ...  ...        ...        ...        ...       ...       ...  ...  ...  ...
        95  25.0  1.0  1.0 -15.840432  -0.477060  23.096819  0.128503  0.991709 -0.0  1.0  inf
        96  25.0  1.0  1.0   4.325647 -13.629269  23.645974  0.898537 -0.438898 -0.0  1.0  inf
        97  25.0  1.0  1.0 -23.994623 -12.282176  23.874757  0.900374 -0.435117 -0.0  1.0  inf
        98  25.0  1.0  1.0  16.447001 -22.098542  24.091469  0.051275 -0.998685 -0.0  1.0  inf
        99  25.0  1.0  1.0 -24.765226  -3.279169  24.516947  0.549633 -0.835406  0.0  1.0  inf
        <BLANKLINE>
        [100 rows x 11 columns]

    Data
    ----
    + index : pandas.Index
        Fiber label. Each label refers to a unique fiber.
    + Fiber dimensions:
        - l : pandas.Series
            Fiber length (mm). By default, 25 mm long fibers are used.
        - b : pandas.Series
            Fiber width (mm). By default, 1 mm wide fibers are used.
        - h : pandas.Series
            Fiber thickness (mm). By default, 1 mm thick fibers are used.
    + Fiber position:
        - x : pandas.Series
            X-coordinate (mm). By default, positions are randomly distributed with a uniform distribution between the positions X: [-25 mm ; 25 mm].
        - y : pandas.Series
            Y-coordinate (mm). By default, positions are randomly distributed with a uniform distribution between the positions Y: [-25 mm ; 25 mm].
        - z : pandas.Series
            Z-coordinate (mm). By default, positions are randomly distributed with a uniform distribution between the positions Z: [-25 mm ; 25 mm].
    + Fiber orientation:
        - u : pandas.Series
            X-component. By default, orientations are randomly distributed with a uniform distribution in the half-unit circle θ: [-π / 2, π / 2].
        - v : pandas.Series
            Y-component. By default, orientations are randomly distributed with a uniform distribution in the half-unit circle θ: [-π / 2, π / 2].
        - w : pandas.Series
            Z-component. By default, orientations are in the plane, so the Z-component is 0.
    + Material properties:
        - G : pandas.Series
            Shear modulus (MPa). By default, shear modulus is 1 MPa.
        - E : pandas.Series
            Tensile modulus (MPa). By default, tensile modulus is infinite, which corresponds to fibers that are infinitely rigid in tension and bending.

    ----

    Attributes
    ----------
    :attr:`attrs` :
        Global attributes of DataFrame.

    Methods
    -------
    :meth:`init` :
        Generate a set of random straight fibers.
    :meth:`check` :
        Check that a :class:`Mat` object is defined correctly.

    ----

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the :class:`Mat` object.

        Parameters
        ----------
        args :
            Additional positional arguments passed to the constructor.

        Other Parameters
        ----------------
        kwargs :
            Additional keyword arguments passed to the constructor.

        See Also
        --------
        `Mat.init`.

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
        n : int, optional
            Number of fibers. Default is 0.
        length : float, optional
            Fiber length (mm). Default is 25 mm.
        width : float, optional
            Fiber width (mm). Default is 1 mm.
        thickness : float, optional
            Fiber thickness (mm). Default is 1 mm.
        size : float, optional
            Box dimensions (mm). Default is 50 mm.
        theta : float, optional
            In plane angle (rad). Default is π rad.
        psi : float, optional
            Out-of-plane angle (rad). Default is 0 rad.
        shear : float, optional
            Shear modulus (MPa). Default is 1 MPa.
        tensile : float, optional
            Tensile modulus (MPa). Default is ∞ MPa.
        seed : int, optional
            Random seed for reproducibility. Default is 0.

        Returns
        -------
        mat : pandas.DataFrame
            Initialized :class:`Mat` object.

        Other Parameters
        ----------------
        kwargs :
            Additional keyword arguments ignored by the function.

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

    # ~~~ Public properties ~~~ #

    @property
    def attrs(self):
        """
        Global attributes of DataFrame:
            - n : int
                Number of fibers. By default, it is empty (n = 0).
            - size : float
                Box dimensions (mm). By default, the domain is a 50 mm square cube.

        """
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = attrs

    # ~~~ Public methods ~~~ #

    @staticmethod
    def check(mat=None):
        """
        Check that a :class:`Mat` object is defined correctly.

        This method is automatically called by functions that use a :class:`Mat` object as input.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a :class:`Mat` object.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of :class:`Mat` object.
        AttributeError
            If any attributes are missing from the dictionary :attr:`attrs`.
        IndexError
            If row indices are incorrectly defined:
                - Row indices are not unique in [0, ..., n-1] where n is the number of fibers.
                - Fiber labels are not sorted.
        ValueError
            If any of the following conditions are not met:
                - Dimensions are not positive.
                - Positions are not within a box of size specified in :attr:`attrs`.
                - Orientation vectors do not have unit lengths.
                - Material properties are not positive.

        Returns
        -------
        mat : pandas.DataFrame
            Validated :class:`Mat` object.

        .. TIP::
            - If `mat` is None, it returns an empty :class:`Mat` object.
            - If a "skip_check" flag is True in :attr:`attrs`, the check is passed.

        """
        if mat is None:
            mat = Mat()

        if "skip_check" in mat.attrs.keys() and mat.attrs["skip_check"]:
            # Return the `Mat` object
            return mat

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

    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(100)

    # Get fiber data
    dimensions = mat[[*"lbh"]]  # size: (n x 3)
    positions = mat[[*"xyz"]]  # size: (n x 3)
    orientations = mat[[*"uvw"]]  # size: (n x 3)

    # Check data
    Mat.check(mat)  # or `mat.check()`
    # -> returns `mat` if correct, otherwise it raises an error.

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    if len(mat):
        # Draw fibers
        for i in tqdm(range(len(mat))):
            # Get fiber data
            fiber = mat.iloc[i]
            # Calculate fiber end points
            A = fiber[[*"xyz"]].values - 0.5 * fiber.l * fiber[[*"uvw"]].values
            B = fiber[[*"xyz"]].values + 0.5 * fiber.l * fiber[[*"uvw"]].values
            plt.plot(*np.c_[A, B])
        # Set drawing box dimensions
        ax.set_xlim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
        ax.set_ylim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
    plt.show()
