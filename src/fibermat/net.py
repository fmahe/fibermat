#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

from fibermat import Mat


class Net(pd.DataFrame):
    r"""
    A class inherited from pandas.DataFrame_ to **build a fiber network**. It describes nodes and connections between fibers within a :class:`Mat` object:

        - **nodes** are defined as the nearest points between pairs of fibers.
        - **connections** link pairs of nodes to define relative positions between fibers.

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a :class:`Mat` object.

    .. NOTE::
        The constructor calls :meth:`init` method if the object is instantiated with parameters. Otherwise, initialization is performed with the pandas.DataFrame_ constructor.

    .. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    :Use:

        >>> # Generate a set of fibers
        >>> mat = Mat(100)
        >>> # Build the fiber network
        >>> net = Net(mat)
        >>> net
              A   B         sA         sB         xA         yA         zA         xB         yB         zB
        0     0   0  12.500000 -12.500000   6.057752  20.856058 -24.338157  -1.176401  -3.074404 -24.338157
        1     0   2   3.938063  -1.799582   3.580217  12.660413 -24.338157   3.580217  12.660413 -23.967450
        2     0   3   6.509881   8.253676   4.324414  15.122205 -24.338157   4.324414  15.122205 -23.766064
        3     0   5   0.269800  -7.165082   2.518746   9.149084 -24.338157   2.518746   9.149084 -21.802237
        4     0   6 -10.466114   6.264470  -0.587864  -1.127531 -24.338157  -0.587864  -1.127531 -21.637518
        ..   ..  ..        ...        ...        ...        ...        ...        ...        ...        ...
        862  95  95  12.500000 -12.500000 -14.234141  11.919304  23.096819 -17.446723 -12.873423  23.096819
        863  96  96  12.500000 -12.500000  15.557356 -19.115497  23.645974  -6.906063  -8.143040  23.645974
        864  97  97  12.500000 -12.500000 -12.739951 -17.721142  23.874757 -35.249295  -6.843209  23.874757
        865  98  98  12.500000 -12.500000  17.087939 -34.582099  24.091469  15.806064  -9.614985  24.091469
        866  99  99  12.500000 -12.500000 -17.894817 -13.721749  24.516947 -31.635635   7.163412  24.516947
        <BLANKLINE>
        [867 rows x 10 columns]

    Data
    ----
    + index : pandas.Index
        Connection label. Each label refers to a unique connection.
    + Pair of fibers:
        - A : pandas.Series
            First fiber label. It must satisfy `net.A` ≤ `net.B`.
        - B : pandas.Series
            Second fiber label. It must satisfy `net.A` ≤ `net.B`.
    + Curvilinear abscissa:
        - sA : pandas.Series
            Curvilinear abscissa of node along the first fiber (mm).
        - sB : pandas.Series
            Curvilinear abscissa of node along the second fiber (mm).
    + Relative node positions:
        - xA : pandas.Series
            X-coordinate of node along the first fiber (mm).
        - yA : pandas.Series
            Y-coordinate of node along the first fiber (mm).
        - zA : pandas.Series
            Z-coordinate of node along the first fiber (mm).
        - xB : pandas.Series
            X-coordinate of node along the second fiber (mm).
        - yB : pandas.Series
            Y-coordinate of node along the second fiber (mm).
        - zB : pandas.Series
            Z-coordinate of node along the second fiber (mm).

    ----

    Attributes
    ----------
    :attr:`attrs` :
        Global attributes of DataFrame.

    Methods
    -------
    :meth:`init` :
        Build a fiber network.
    :meth:`check` :
        Check that a :class:`Net` object is defined correctly.

    ----

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the :class:`Net` object.

        Parameters
        ----------
        args :
            Additional positional arguments passed to the constructor.
        kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        `Net.init`.

        """
        if (len(args) and isinstance(args[0], pd.DataFrame)
                and not isinstance(args[0], Mat)):
            # Initialize the DataFrame from argument
            super().__init__(*args, **kwargs)
            # Copy global attributes from argument
            self.attrs = args[0].attrs

        else:
            # Initialize the DataFrame from parameters
            self.__init__(Net.init(*args, **kwargs))

        # Check `Net` object
        Net.check(self)

    # ~~~ Constructor ~~~ #

    @staticmethod
    def init(mat=None, periodic=True, pairs=None, **kwargs):
        """
        Build a fiber network.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a :class:`Mat` object.

        Returns
        -------
        net : pandas.DataFrame
            Initialized :class:`Net` object.

        Other Parameters
        ----------------
        periodic : bool, optional
            If True, duplicate fibers for periodicity. Default is True.
        pairs : numpy.ndarray, optional
            Pairs of fiber indices used to find nearest points. Size: (m x 2).
        kwargs :
            Additional keyword arguments ignored by the function.

        """
        # Optional
        mat = Mat(mat)

        # Periodic boundary conditions (optional)
        if periodic:
            X = Y = mat.attrs["size"]
            # Duplicate fibers along boundaries
            l = mat.l.max()
            x1, x2 = -0.5 * X, 0.5 * X
            y1, y2 = -0.5 * Y, 0.5 * Y
            mask_x = mat.x < x1 + l
            mask_y = mat.y < y1 + l
            mask_xy = (mat.x > x2 - l) & (mat.y > y2 - l)
            # Left/right sides
            mat_x = mat[mask_x].copy()
            mat_x["inner"] = False
            mat_x["x"] += mat.attrs["size"]
            # Back/front sides
            mat_y = mat[mask_y].copy()
            mat_y["inner"] = False
            mat_y["y"] += mat.attrs["size"]
            # Corners
            mat_xy = mat[mask_xy].copy()
            mat_xy["inner"] = False
            mat_xy["x"] -= mat.attrs["size"]
            mat_xy["y"] -= mat.attrs["size"]
            # Merge repeated mats
            mat_ = pd.concat([mat, mat_x, mat_y, mat_xy])
        else:
            mat_ = mat

        # Pairwise indices
        if pairs is None:
            pairs = np.c_[np.triu_indices(len(mat_))]

        # Pairwise distances (optional)
        A = mat_[[*"xy"]].values[pairs][:, 0, :]
        B = mat_[[*"xy"]].values[pairs][:, 1, :]
        distances = np.linalg.norm(A - B, axis=1)
        # Spherical neighborhoods
        lengths = mat_[["l"]].values[pairs]
        diameters = np.mean(lengths, axis=1).ravel()
        pairs = pairs[distances < diameters]

        # Get material data
        L = mat_[[*"l"]].values[pairs]
        X = mat_[[*"xyz"]].values[pairs]
        U = mat_[[*"uvw"]].values[pairs]
        Xt = np.swapaxes(X, 1, 2)
        Ut = np.swapaxes(U, 1, 2)
        r = np.diff(Xt, axis=2)

        # Define normal projective system
        UUt = U @ Ut
        Ur = U @ r
        R = 0.5 * L
        mask = (np.linalg.det(UUt) != 0)
        # Solve normal equations: 𝕌ᵀ·R⃗ = r⃗ ⟺ 𝕌·𝕌ᵀ·R⃗ = 𝕌·r⃗
        R[mask] = np.linalg.solve(UUt[mask], Ur[mask])

        # Prepare node data
        pairs = mat_.index.values[pairs]
        abscissa = R * np.array([[1], [-1]])
        # points = X + abscissa * U

        # Remove nodes that do not belong to finite lines
        radius = 0.5 * mat_.l.values[pairs].reshape(-1, 2, 1)
        mask = np.all(np.abs(abscissa) < radius, axis=1).ravel()
        mask |= np.equal(*pairs.T)  # for fiber end points
        pairs = pairs[mask]
        # points = points[mask]
        abscissa = abscissa[mask]

        # Sort and remove repeated nodes (for periodicity)
        indices = np.argsort(pairs, axis=1)
        pairs = np.take_along_axis(pairs, indices, axis=1)
        abscissa = np.take_along_axis(abscissa[..., 0], indices, axis=1)
        df = (pd.DataFrame(np.c_[pairs, abscissa])
              .sort_values(by=[0, 1])
              .drop_duplicates([0, 1]))
        pairs = df[[0, 1]].values.astype(int)
        abscissa = df[[2, 3]].values.reshape(-1, 2, 1)
        points = (mat_[[*"xyz"]].values[pairs]
                  + abscissa * mat_[[*"uvw"]].values[pairs])

        # Initialize net DataFrame
        net = pd.DataFrame(
            data=np.c_[(pairs,
                        abscissa.reshape(-1, 2),
                        points.reshape(-1, 6))],
            columns=["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"]
        )
        # Convert type to int
        net[[*"AB"]] = net[[*"AB"]].astype(int)

        # Set attributes
        net.attrs = mat.attrs
        net.attrs["periodic"] = periodic

        # Return the `Net` object
        return net

    # ~~~ Public properties ~~~ #

    @property
    def attrs(self):
        """
        Global attributes of DataFrame:
            - n : int
                Number of fibers. By default, it is empty (n = 0).
            - size : float
                Box dimensions (mm). By default, the domain is a 50 mm square cube.
            - periodic : bool
                Boundary periodicity. By default, the domain is periodic.

        """
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = attrs

    # ~~~ Public methods ~~~ #

    @staticmethod
    def check(net=None):
        """
        Check that a :class:`Net` object is defined correctly.

        This method is automatically called by functions that use a :class:`Net` object as input.

        Parameters
        ----------
        net : pandas.DataFrame, optional
            Fiber network represented by a :class:`Net` object.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of :class:`Net` object.
        AttributeError
            If any attributes are missing from the dictionary :attr:`attrs`.
        IndexError
            If row indices are incorrectly defined:
                - Row indices are not unique in [0, ..., m-1] where m is the number of connections.
                - Connection labels are not sorted.
        TypeError
            If labels are not integers.
        ValueError
            If any of the following conditions are not met:
                - Fiber labels are incorrect.
                - There are duplicate connections.
                - Fiber labels are not ordered.

        Returns
        -------
        net : pandas.DataFrame
            Validated :class:`Net` object.

        .. TIP::
            - If `net` is None, it returns an empty :class:`Net` object.
            - If a "skip_check" flag is True in :attr:`attrs`, the check is passed.

        """
        if net is None:
            net = Net()

        if "skip_check" in net.attrs.keys() and net.attrs["skip_check"]:
            # Return the `Net` object
            return net

        # Keys
        try:
            net[["A", "B", "sA", "sB", "xA", "yA", "zA", "xB", "yB", "zB"]]
        except KeyError as e:
            raise KeyError(e)

        # Attributes
        if not ("n" in net.attrs.keys()):
            raise AttributeError("'n' is not in attribute dictionary.")
        if not ("size" in net.attrs.keys()):
            raise AttributeError("'size' is not in attribute dictionary.")
        if not ("periodic" in net.attrs.keys()):
            raise AttributeError("'periodic' is not in attribute dictionary.")

        # Indices
        if not np.all(np.unique(net.index) == np.arange(len(net))):
            raise IndexError("Row indices must be unique in [0,..., {}]."
                             .format(len(net) - 1))
        if not np.all(net.index == np.arange(len(net))):
            raise IndexError("Connection labels must be sorted.")

        # Types
        if net[[*"AB"]].values.dtype != int:
            raise TypeError("Fiber labels are not integers.")

        # Data
        if len(net) and not (0 <= net[[*"AB"]].values.min()
                             and net[[*"AB"]].values.max() < net.attrs["n"]):
            raise ValueError("Fiber labels must be in [0,..., {}]."
                             .format(net.attrs["n"] - 1))
        if (len(net) != len(
                net[["A", "B", "sA", "sB"]].drop_duplicates(
                    ignore_index=True))):
            raise ValueError("Connections must be unique.")
        if not np.all(net.A <= net.B):
            raise ValueError("Pairs of fiber labels must be ordered: A ≤ B.")

        # Return the `Net` object
        return net


class Stack(Net):
    """
    A class inherited from :class:`Net` to **stack a set of fibers**. It solves the *linear programming system*:

    .. MATH::
        \min_{z} (-\mathbf{f} \cdot \mathbf{z}) \quad s.t. \quad \mathbb{C} \, \mathbf{z} \leq \mathbf{H} \quad and \quad \mathbf{z} \geq \mathbf{h} / 2
    .. MATH::
        with \quad \mathbf{f} = -\mathbf{m} \, g \quad and \quad \mathbf{h} > 0

    where:
        - :math:`\mathbf{f}` is the vector of fiber weights (with :math:`\mathbf{m}` fiber masses, :math:`g`: gravity).
        - :math:`\mathbf{z}` is the unknown vector of fiber vertical positions.
        - :math:`\mathbf{h}` is the vector of fiber thicknesses.
        - :math:`\mathbb{C}` is the matrix of inequality constraints that positions must satisfy to prevent the fibers from crossing each other.
        - :math:`-\mathbf{H}` corresponds to the minimum distances between the pairs of fibers.

    *Non-penetration conditions* between two fibers give the expressions of rows of :math:`\mathbb{C}` and :math:`\mathbf{H}`:

    .. MATH::
        z_B - z_A \geq (h_A + h_B) \, / \, 2 \quad \Leftrightarrow \quad z_A - z_B \leq - (h_A + h_B) \, / \, 2

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a :class:`Mat` object.
    net : pandas.DataFrame, optional
        Fiber network represented by a :class:`Net` object.

    .. NOTE::
        The constructor calls :meth:`init` method if the object is instantiated with parameters. Otherwise, initialization is performed with the pandas.DataFrame_ constructor.

    .. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

        :Use:

            >>> # Generate a set of fibers
            >>> mat = Mat(100)
            >>> # Build the fiber network
            >>> net = Net(mat)
            >>> # Stack fibers
            >>> stack = Stack(mat, net)
            >>> stack
                  A   B         sA         sB         xA         yA    zA         xB         yB    zB
            0     0   0  12.500000 -12.500000   6.057752  20.856058   0.5  -1.176401  -3.074404   0.5
            1     0   2   3.938063  -1.799582   3.580217  12.660413   0.5   3.580217  12.660413   1.5
            2     0   3   6.509881   8.253676   4.324414  15.122205   0.5   4.324414  15.122205   2.5
            3     0   5   0.269800  -7.165082   2.518746   9.149084   0.5   2.518746   9.149084   1.5
            4     0   6 -10.466114   6.264470  -0.587864  -1.127531   0.5  -0.587864  -1.127531   1.5
            ..   ..  ..        ...        ...        ...        ...   ...        ...        ...   ...
            862  95  95  12.500000 -12.500000 -14.234141  11.919304  27.5 -17.446723 -12.873423  27.5
            863  96  96  12.500000 -12.500000  15.557356 -19.115497  27.5  -6.906063  -8.143040  27.5
            864  97  97  12.500000 -12.500000 -12.739951 -17.721142  27.5 -35.249295  -6.843209  27.5
            865  98  98  12.500000 -12.500000  17.087939 -34.582099  27.5  15.806064  -9.614985  27.5
            866  99  99  12.500000 -12.500000 -17.894817 -13.721749  26.5 -31.635635   7.163412  26.5
            <BLANKLINE>
            [867 rows x 10 columns]

    Data
    ----
    + index : pandas.Index
        Connection label. Each label refers to a unique connection.
    + Pair of fibers:
        - A : pandas.Series
            First fiber label. It must satisfy `net.A` ≤ `net.B`.
        - B : pandas.Series
            Second fiber label. It must satisfy `net.A` ≤ `net.B`.
    + Curvilinear abscissa:
        - sA : pandas.Series
            Curvilinear abscissa of node along the first fiber (mm).
        - sB : pandas.Series
            Curvilinear abscissa of node along the second fiber (mm).
    + Relative node positions:
        - xA : pandas.Series
            X-coordinate of node along the first fiber (mm).
        - yA : pandas.Series
            Y-coordinate of node along the first fiber (mm).
        - zA : pandas.Series
            Z-coordinate of node along the first fiber (mm).
        - xB : pandas.Series
            X-coordinate of node along the second fiber (mm).
        - yB : pandas.Series
            Y-coordinate of node along the second fiber (mm).
        - zB : pandas.Series
            Z-coordinate of node along the second fiber (mm).

    ----

    Attributes
    ----------
    :attr:`attrs` :
        Global attributes of DataFrame.

    Methods
    -------
    :meth:`init` :
        Stack fibers by gravity.
    :meth:`check` :
        Check that a :class:`Stack` object is defined correctly.
    :meth:`solve` :
        Solve the stacking problem.
    :meth:`constraint` :
        Assemble the linear system.

    ----

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the :class:`Stack` object.

        Parameters
        ----------
        args :
            Additional positional arguments passed to the constructor.
        kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        `Stack.init`.

        """
        if (len(args) and isinstance(args[0], pd.DataFrame)
                and not isinstance(args[0], Mat)):
            # Initialize the DataFrame from argument
            super().__init__(*args, **kwargs)
            # Copy global attributes from argument
            self.attrs = args[0].attrs

        else:
            # Initialize the DataFrame from parameters
            self.__init__(Stack.init(*args, **kwargs))

        # Check `Stack` object
        Stack.check(self)

    # ~~~ Constructor ~~~ #

    @staticmethod
    def init(mat=None, net=None, threshold=None, **kwargs):
        """
        Stack fibers by gravity.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a :class:`Mat` object.
        net : pandas.DataFrame, optional
            Fiber network represented by a :class:`Net` object.

        Returns
        -------
        stack : pandas.DataFrame
            Initialized :class:`Stack` object.

        Other Parameters
        ----------------
        threshold : float, optional
            Threshold distance value for proximity detection (mm).
        kwargs :
            Additional keyword arguments ignored by the function.

        .. WARNING::
            :class:`Mat` object is modified during execution.

        """
        # Optional
        mat = Mat.check(mat)
        net = Net.check(net)

        # Solve the stacking problem
        linsol = Stack.solve(mat, net)

        if linsol:
            # Update DataFrames
            mat.z = linsol.x
            net = Net(mat, periodic=net.attrs["periodic"])

            # Remove nodes based on threshold distances between nodes
            mask = np.zeros(len(net))
            # : 0 if removed node
            # : 1 if kept node
            # : 2 if end node
            if threshold is not None:
                mask[np.abs(net.zB.values - net.zA.values) <= threshold] = 1
                mask[net.A == net.B] = 2
                net = net[mask > 0]
                net.reset_index(drop=True, inplace=True)

        # Initialize stack DataFrame
        stack = Stack(net)

        # Return the `Stack` object
        return stack

    # ~~~ Public properties ~~~ #

    @property
    def attrs(self):
        """
        Global attributes of DataFrame:
            - n : int
                Number of fibers. By default, it is empty (n = 0).
            - size : float
                Box dimensions (mm). By default, the domain is a 50 mm square cube.
            - periodic : bool
                Boundary periodicity. By default, the domain is periodic.

        """
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        self._attrs = attrs

    # ~~~ Public methods ~~~ #

    @staticmethod
    def check(stack=None):
        """
        Check that a :class:`Stack` object is defined correctly.

        This method is automatically called by functions that use a :class:`Stack` object as input.

        Parameters
        ----------
        stack : pandas.DataFrame, optional
            Fiber stack represented by a :class:`Stack` object.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of :class:`Stack` object.
        AttributeError
            If any attributes are missing from the dictionary :attr:`attrs`.
        IndexError
            If row indices are incorrectly defined:
                - Row indices are not unique in [0, ..., m-1] where m is the number of connections.
                - Connection labels are not sorted.
        TypeError
            If labels are not integers.
        ValueError
            If any of the following conditions are not met:
                - Fiber labels are incorrect.
                - There are duplicate connections.
                - Fiber labels are not ordered.

        Returns
        -------
        stack : pandas.DataFrame
            Validated :class:`Stack` object.

        .. TIP::
            - If `stack` is None, it returns an empty :class:`Stack` object.
            - If a "skip_check" flag is True in :attr:`attrs`, the check is passed.

        """
        if stack is None:
            stack = Stack()

        # Return the `Stack` object
        return Net.check(stack)

    @staticmethod
    def solve(mat=None, net=None):
        """
        Solve the stacking problem.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a :class:`Mat` object.
        net : pandas.DataFrame, optional
            Fiber network represented by a :class:`Net` object.

        Returns
        -------
        linsol : OptimizeResult
            Results of linear programming solver.

        .. SEEALSO::
            The solver is based on scipy.optimize.linprog_.

        .. _scipy.optimize.linprog: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

        """
        # Optional
        mat = Mat.check(mat)
        net = Net.check(net)

        # Assemble the linear programming system
        C, f, H, h = Stack.constraint(mat, net)

        if len(mat):
            # Linear programming solver
            bounds = np.c_[0.5 * h, np.full(len(h), np.inf)]
            linsol = sp.optimize.linprog(f, C, H, bounds=bounds)
        else:
            linsol = None

        return linsol

    @staticmethod
    def constraint(mat=None, net=None):
        """
        Assemble the linear system:

        .. MATH::
            \min_{z} (-\mathbf{f} \cdot \mathbf{z}) \quad s.t. \quad \mathbb{C} \, \mathbf{z} \leq \mathbf{H} \quad and \quad \mathbf{z} \geq \mathbf{h} / 2
        .. MATH::
            with \quad \mathbf{f} = -\mathbf{m} \, g \quad and \quad \mathbf{h} > 0

        Parameters
        ----------
        mat : pd.DataFrame, optional
            Set of fibers represented by a :class:`Mat` object.
        net : pd.DataFrame, optional
            Fiber network represented by a :class:`Net` object.

        Returns
        -------
            C : sparse matrix
                Constraint matrix.
            f : numpy.ndarray
                Force vector.
            H : numpy.ndarray
                Upper-bound vector.
            h : numpy.ndarray
                Thickness vector.

        """
        # Optional
        mat = Mat.check(mat)
        net = Net.check(net)

        # Get network data
        mask = net.A.values < net.B.values
        i = net.A[mask].values
        j = net.B[mask].values
        k = 1 * np.arange(len(i))
        O = i * 0  # : zero
        I = O + 1  # : one

        # Get material data
        h = mat.h.values

        # Create constraint data
        row = np.array([k, k]).ravel()
        col = np.array([i, j]).ravel()
        data = np.array([I, -I]).ravel()

        # Initialize ℂ matrix
        C = sp.sparse.coo_matrix((data, (row, col)),
                                 shape=(1 * len(net[mask]), 1 * len(mat)))

        # Initialize 𝒇 and 𝑯 vectors
        f = np.pi / 4 * mat[[*"lbh"]].prod(axis=1)  # : potential field
        H = np.zeros(C.shape[0])
        # X₂ - X₁ ≥ ½(h₁ + h₂) ⟺ X₁ - X₂ ≤ -½(h₁ + h₂)
        H -= 0.5 * (h[i] + h[j])

        return C, f, H, h


def _test_stack(n=100):
    """
    Test for the stacking algorithm.

    Parameters
    ----------
    n : int, optional
        Number of fibers. Default is 100.

    Raises
    ------
    ValueError
        If results are incorrect.

    Returns
    -------
    bool
        Returns True if the test was successful.

    Example
    -------
    >>> _test_stack()
    True

    """
    mat = Mat(n, thickness=0.1, psi=0, seed=None)
    mat.l = np.random.normal(mat.l.mean(), 0.04 * mat.l.mean(), len(mat))
    mat.h = np.random.normal(mat.h.mean(), 0.1 * mat.h.mean(), len(mat))
    net = Net(mat)
    Stack(mat, net)

    # Get material data
    h = mat.h.values
    z = 0.5 * h

    # Stack fibers
    for i, j in net[[*"AB"]][net.A < net.B].values:
        z[j] = max(z[i] + 0.5 * (h[i] + h[j]), z[j])

    if not np.all(mat.z == z):
        raise ValueError("Stacking algorithm error.")
    else:
        return True


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.interpolate import interp1d
    from tqdm import tqdm

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(10)
    # Build the fiber network
    net = Net(mat, periodic=True)
    # Stack fibers
    stack = Stack(mat, net)

    # Get the linear system
    C, f, H, h = Stack.constraint(mat, net)
    linsol = Stack.solve(mat, net)
    # Contact force
    force = linsol.ineqlin.marginals
    # Resulting force
    load = 0.5 * force @ np.abs(C) + 0.5 * force @ C

    # Check data
    Stack.check(stack)  # or `stack.check()`
    # -> returns `stack` if correct, otherwise it raises an error.

    # Normalize by fiber weight
    load /= np.pi / 4 * mat[[*"lbh"]].prod(axis=1).mean()
    # Get loaded nodes
    points = (stack[stack.A < stack.B][["xA", "yA", "zA", "xB", "yB", "zB"]]
              .values.reshape(-1, 2, 3))
    # Prepare color scale
    cmap = plt.cm.viridis
    color = interp1d([np.min(load), np.max(load)], [0, 1])

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
            plt.plot(*np.c_[A, B], c=cmap(color(load[i])))
    if len(points):
        # Draw contacts
        for point in tqdm(points[~np.isclose(force, 0)]):
            plt.plot(*point.T, '--ok', lw=1, mfc='none', ms=3, alpha=0.2)
    # Set drawing box dimensions
    ax.set_xlim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
    ax.set_ylim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
    # Add a color bar
    norm = plt.Normalize(vmin=np.min(load), vmax=np.max(load))
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(smap, ax=ax)
    cbar.set_label("Load / $mg$ ($N\,/\,N$)")
    plt.show()
