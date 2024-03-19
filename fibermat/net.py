#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕸️ Net(work)
------------

Classes
-------
Net :
    A class to build a fiber network.
Stack :
    A class to stack a set of fibers.

"""

import numpy as np
import pandas as pd
import scipy as sp

from fibermat import Mat


class Net(pd.DataFrame):
    """
    A class to build a fiber network.
    It describes nodes and connections within a `Mat` object.

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a `Mat` object.
    pairs : numpy.ndarray, optional
        Pairs of fiber indices used to find nearest points. Size: (M x 2).
    periodic : bool
        If True, duplicate fibers for periodicity. Default is True.

    Attributes
    ----------
    attrs : dictionary
        Global attributes:
            - n : int, Number of fibers.
            - size : float, Box dimensions (mm).
            - periodic : bool, Periodicity.
    index : pandas.Index
        Connection label.
    A : pandas.Series
        First fiber label.
    B : pandas.Series
        Second fiber label.
    sA : pandas.Series
        Curvilinear abscissa of node along the first fiber (mm).
    sB : pandas.Series
        Curvilinear abscissa of node along the second fiber (mm).
    xA : pandas.Series
        X-coordinate of node along the first fiber (mm).
    yA : pandas.Series
        Y-coordinate of node along the first fiber (mm).
    zA : pandas.Series
        Z-coordinate of node along the first fiber (mm).
    xB : pandas.Series
        X-coordinate of node along the second fiber (mm).
    yB : pandas.Series
        Y-coordinate of node along the second fiber (mm).
    zB : pandas.Series
        Z-coordinate of node along the second fiber (mm).

    Methods
    -------
    init()
        Build a fiber network.
    check()
        Check that `Net` object is defined correctly.

    Properties
    ----------
    pairs : numpy.ndarray
        Pairs of fiber labels representing connected nodes. Size: (M x 2).
    points : numpy.ndarray
        Node coordinates along each fiber. Size: (M x 2 x 3).
    abscissa : numpy.ndarray
        Curvilinear abscissa of nodes along each fiber. Size: (M x 2 x 1).

    Examples
    --------
    ```python
        >>> # Generate a set of fibers
        >>> mat = Mat(**inputs)
        >>> # Build the fiber network
        >>> net = Net(mat, **inputs)
        >>> # Get node data
        >>> pairs = net[[*"AB"]].values
        >>> abscissa = net[["sA", "sB"]].values.reshape(-1, 2, 1)
        >>> points = net[["xA", "yA", "zA", "xB", "yB", "zB"]].values.reshape(-1, 2, 3)

    ```

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the `Net` object.

        Parameters
        ----------
        *args :
            Additional positional arguments passed to the constructor.
        **kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        Net.init :
            Build a fiber network.

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
    def init(mat=None, pairs=None, periodic=True, **kwargs):
        """
        Build a fiber network.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a `Mat` object.
        pairs : numpy.ndarray, optional
            Pairs of fiber indices used to find nearest points. Size: (M x 2).
        periodic : bool
            If True, duplicate fibers for periodicity. Default is True.
        **kwargs
            Additional keyword arguments ignored by the function.

        Returns
        -------
        net : pandas.DataFrame
            Initialized `Net` object.

        """
        # Optional
        mat = Mat.check(mat)

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
        points = X + abscissa * U

        # Remove nodes that do not belong to finite lines
        radius = 0.5 * mat_.l.values[pairs].reshape(-1, 2, 1)
        mask = np.all(np.abs(abscissa) < radius, axis=1).ravel()
        mask |= np.equal(*pairs.T)  # for fiber end points
        pairs = pairs[mask]
        points = points[mask]
        abscissa = abscissa[mask]

        # Sort and remove repeated nodes (for periodicity)
        indices = np.argsort(pairs, axis=1)
        pairs = np.take_along_axis(pairs, indices, axis=1)
        abscissa = np.take_along_axis(abscissa[..., 0], indices, axis=1)
        df = (pd.DataFrame(np.c_[pairs, abscissa])
              .sort_values([0, 1])
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

    # ~~~ Public methods ~~~ #

    @staticmethod
    def check(net=None):
        """
        Check that `Net` object is defined correctly.

        Parameters
        ----------
        net : pandas.DataFrame, optional
            Fiber network represented by a `Net` object.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of `Net` object.
        AttributeError
            If any attributes are missing from the dictionary `net.attrs`.
        IndexError
            if row indices are incorrectly defined:
                - Row indices are not unique in [0, ..., n-1] where n is the number of connections.
                - Connection labels are not sorted.
        TypeError
            If fiber labels are not integers.
        ValueError
            If any of the following conditions are not met:
                - Fiber labels are incorrect.
                - There are duplicate connections.
                - Fiber labels are not ordered.

        Returns
        -------
        net : pandas.DataFrame
            Validated `Net` object.

        Notes
        -----
        If `net` is None, it returns an empty `Net` object.

        """
        if net is None:
            net = Net()

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
            raise IndexError("Row indices must be unique in [0,..., {}-1]."
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
    A class to stack a set of fibers.
    It solves the following linear programming system:

        min_{𝒛}(-𝒇·𝒛) s.t. ℂ·𝒛 ≤ 𝑯 and 𝒛 ≥ ½𝐡
            with 𝒇 = -𝐦g and 𝐡 > 0

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a `Mat` object.
    net : pandas.DataFrame, optional
        Fiber network represented by a `Net` object.
    threshold: float, optional
        Threshold distance value for proximity detection (mm).

    Attributes
    ----------
    attrs : dictionary
        Global attributes:
            - n : int, Number of fibers.
            - size : float, Box dimensions (mm).
            - periodic : bool, Periodicity.
    index : pandas.Index
        Connection label.
    A : pandas.Series
        First fiber label.
    B : pandas.Series
        Second fiber label.
    sA : pandas.Series
        Curvilinear abscissa of node along the first fiber (mm).
    sB : pandas.Series
        Curvilinear abscissa of node along the second fiber (mm).
    xA : pandas.Series
        X-coordinate of node along the first fiber (mm).
    yA : pandas.Series
        Y-coordinate of node along the first fiber (mm).
    zA : pandas.Series
        Z-coordinate of node along the first fiber (mm).
    xB : pandas.Series
        X-coordinate of node along the second fiber (mm).
    yB : pandas.Series
        Y-coordinate of node along the second fiber (mm).
    zB : pandas.Series
        Z-coordinate of node along the second fiber (mm).

    Methods
    -------
    init()
        Stack fibers under a gravity field.
    check()
        Check that `Stack` object is defined correctly.
    solve()
        Linear programming solver for the stacking problem.
    constraint()
        Assembly linear system to be minimized.

    Properties
    ----------
    force : numpy.ndarray
        Contact force. Size: (M).
    load : numpy.ndarray
        Resulting force. Size: (N).

    Examples
    --------
    ```python
        >>> # Generate a set of fibers
        >>> mat = Mat(**inputs)
        >>> # Build the fiber network
        >>> net = Net(mat, **inputs)
        >>> # Stack fibers
        >>> stack = Stack(mat, net, **inputs)
        >>> # Get the linear system
        >>> C, f, H, h = Stack.constraint(mat, net)
        >>> linsol = Stack.solve(mat, net)
        >>> # Contact force
        >>> force = linsol.ineqlin.marginals
        >>> # Resulting force
        >>> load = 0.5 * force @ np.abs(C) + 0.5 * force @ C


    ```

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the `Stack` object.

        Parameters
        ----------
        *args :
            Additional positional arguments passed to the constructor.
        **kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        Stack.init :
            Stack fibers under a gravity field.

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
        Stack fibers under a gravity field.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a `Mat` object.
        net : pandas.DataFrame, optional
            Fiber network represented by a `Net` object.
        threshold: float, optional
            Threshold distance value for proximity detection (mm).
        **kwargs :
            Additional keyword arguments ignored by the function.

        Returns
        -------
        stack : pandas.DataFrame
            Initialized `Stack` object.

        Notes
        -----
        `Mat` object is modified during execution.

        """
        # Optional
        mat = Mat.check(mat)
        net = Net.check(net)

        # Solve the stacking problem
        linsol = Stack.solve(mat, net)

        if linsol:
            # Update DataFrames
            mat["z"] = linsol.x
            net = Net(mat)

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

    # ~~~ Public methods ~~~ #

    @staticmethod
    def check(stack=None):
        """
        Check that `Stack` object is defined correctly.

        Parameters
        ----------
        stack : pandas.DataFrame, optional
            Fiber stack represented by a `Stack` object.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of `Stack` object.
        AttributeError
            If any attributes are missing from the dictionary `stack.attrs`.
        IndexError
            if row indices are incorrectly defined:
                - Row indices are not unique in [0, ..., n-1] where n is the number of connections.
                - Connection labels are not sorted.
        TypeError
            If fiber labels are not integers.
        ValueError
            If any of the following conditions are not met:
                - Fiber labels are incorrect.
                - There are duplicate connections.
                - Fiber labels are not ordered.

        Returns
        -------
        stack : pandas.DataFrame
            Validated `Stack` object.

        Notes
        -----
        If `stack` is None, it returns an empty `Stack` object.

        """
        if stack is None:
            stack = Stack()

        # Return the `Stack` object
        return Net.check(stack)

    @staticmethod
    def solve(mat=None, net=None):
        """
        Linear programming solver for the stacking problem.

        Parameters
        ----------
        mat : pandas.DataFrame, optional
            Set of fibers represented by a `Mat` object.
        net : pandas.DataFrame, optional
            Fiber network represented by a `Net` object.

        Returns
        -------
        linsol : OptimizeResult
            Results of linear programming solver.

        """
        # Optional
        mat = Mat.check(mat)
        net = Net.check(net)

        # Assembly linear programming system
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
        Assembly linear system to be minimized:

            min_{z}(-𝒇·z) s.t. ℂ·z ≤ 𝑯 and z ≥ ½𝐡
                with 𝒇 = -𝐦g and 𝐡 > 0

        Parameters
        ----------
        mat : Mat, optional
            Set of fibers represented by a `Mat` object.
        net : Net, optional
            Fiber network represented by a `Net` object.

        Returns
        -------
        tuple
            C : sparse matrix
                Constraint matrix.
            f : numpy.ndarray
                Force vector.
            H : numpy.ndarray
                Upper bound vector.
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
    n : int
        Number of fibers. Default is 100.

    Raises
    ------
    ValueError
        If results are incorrect.

    Returns
    -------
    bool
        Returns True if the test was successful.

    """
    mat = Mat(n, thickness=0.1, psi=0, seed=None)
    mat.l = np.random.normal(mat.l.mean(), 0.04 * mat.l.mean(), len(mat))
    mat.h = np.random.normal(mat.h.mean(), 0.1 * mat.h.mean(), len(mat))
    net = Net(mat)
    stack = Stack(mat, net)

    Mat.check(mat)
    Stack.check(stack)

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
    from matplotlib import pyplot as plt
    from scipy.interpolate import interp1d
    from tqdm import tqdm

    # Generate a set of fibers
    mat = Mat(10)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(mat, net, threshold=None)

    print(_test_stack())

    C, f, H, h = Stack.constraint(mat, net)
    linsol = Stack.solve(mat, net)

    # Contact force
    force = linsol.ineqlin.marginals
    # Normalize by fiber weight
    force /= np.pi / 4 * mat[[*"lbh"]].prod(axis=1).mean()
    # Resulting force
    load = 0.5 * force @ np.abs(C) + 0.5 * force @ C
    color = interp1d([np.min(load), np.max(load)], [0, 1])

    points = (stack[stack.A < stack.B][["xA", "yA", "zA", "xB", "yB", "zB"]]
              .values.reshape(-1, 2, 3))

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    # Draw fibers
    for i in tqdm(range(len(mat))):
        fiber = mat.iloc[i]
        A = fiber[[*"xyz"]].values - 0.5 * fiber.l * fiber[[*"uvw"]].values
        B = fiber[[*"xyz"]].values + 0.5 * fiber.l * fiber[[*"uvw"]].values
        plt.plot(*np.c_[A, B], c=plt.cm.viridis(color(load[i])))
    # Draw contacts
    for point in tqdm(points[~np.isclose(force, 0)]):
        plt.plot(*point.T, '--ok', lw=1, mfc='none', ms=3, alpha=0.2)
    ax.set_xlim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
    ax.set_ylim(-0.5 * mat.attrs["size"], 0.5 * mat.attrs["size"])
    # Color bar
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.min(load), vmax=np.max(load))
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(smap, ax=ax)
    cbar.set_label("Load / $mg$ ($N\,/\,N$)")
    plt.show()
