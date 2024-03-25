#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from tqdm import tqdm

from fibermat import Mat, Net


class Mesh(pd.DataFrame):
    """
    A class inherited from pandas.DataFrame_ to **represent a mesh structure** for a set of discontinuous fibers. It defines:

        - the **beam** elements (intra-fiber connections).
        - the **constraint** elements (inter-fiber connections).

    Parameters
    ----------
    net : pandas.DataFrame, optional
         Fiber network represented by a :class:`Net` object.

    Other Parameters
    ----------------
    kwargs :
        Additional keyword arguments ignored by the function.

    .. NOTE::
        The constructor calls :meth:`init` method if the object is instantiated with parameters. Otherwise, initialization is performed with the pandas.DataFrame_ constructor.

    .. _pandas.DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

    :Use:

        >>> # Generate a set of fibers
        >>> mat = Mat(100)
        >>> # Build the fiber network
        >>> net = Net(mat)
        >>> # Create the fiber mesh
        >>> mesh = Mesh(net)
        >>> mesh
              fiber          s          x          y          z  beam  constraint
        0         0 -12.500000  -1.176401  -3.074404 -24.338157     1           0
        1         0 -11.222534  -0.806746  -1.851590 -24.338157     2         752
        2         0 -10.466114  -0.587864  -1.127531 -24.338157     3         123
        3         0 -10.009779  -0.455816  -0.690719 -24.338157     4        1519
        4         0  -5.432013   0.868835   3.691203 -24.338157     5         706
        ...     ...        ...        ...        ...        ...   ...         ...
        1729     99   6.159453 -21.379789  -8.424815  24.516947  1730        1094
        1730     99   6.740970 -21.060169  -8.910618  24.516947  1731         157
        1731     99   7.284437 -20.761462  -9.364634  24.516947  1732        1294
        1732     99  11.270660 -18.570503 -12.694751  24.516947  1733        1585
        1733     99  12.500000 -17.894817 -13.721749  24.516947  1716        1733
        <BLANKLINE>
        [1734 rows x 7 columns]

    Data
    ----
    + index : pandas.Index
        Node label. Each label refers to a unique node.
    + Node:
        - fiber : pandas.Series
            Label of the fiber to which the node belongs
        - s : pandas.Series
            Node curvilinear abscissa along the fiber (mm).
    + Node position:
        - x : pandas.Series
            X-coordinate of the node (mm).
        - y : pandas.Series
            Y-coordinate of the node (mm).
        - z : pandas.Series
            Z-coordinate of the node (mm).
    + Elements:
        - constraint : pandas.Series
            Index of the connected node. It defines a constraint on the relative node positions.
        - beam : pandas.Series
            Index of the next node along the fiber. It defines a mechanical beam element.

    ----

    Attributes
    ----------
    :attr:`attrs` :
        Global attributes of DataFrame.

    Methods
    -------
    :meth:`init` :
        Create a discontinuous fiber mesh.
    :meth:`check` :
        Check that a :class:`Mesh` object is defined correctly.
    :meth:`isMesh` :
        Check that an object can be instantiated as a :class:`Mesh`.

    ----

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the :class:`Mesh` object.

        Parameters
        ----------
        args :
            Additional positional arguments passed to the constructor.
        kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        `Mesh.init`.

        """
        if (len(args) and isinstance(args[0], pd.DataFrame)
                and not Net.isNet(args[0])):
            # Initialize the DataFrame from argument
            super().__init__(*args, **kwargs)
            # Copy global attributes from argument
            self.attrs = args[0].attrs

        else:
            # Initialize the DataFrame from parameters
            self.__init__(Mesh.init(*args, **kwargs))

        # Check `Mesh` object
        Mesh.check(self)

    # ~~~ Constructor ~~~ #

    @staticmethod
    def init(net=None, **kwargs):
        """
        Create a discontinuous fiber mesh.

        Parameters
        ----------
        net : pandas.DataFrame, optional
             Fiber network represented by a :class:`Net` object.

        Returns
        -------
        mesh : pandas.DataFrame
            Initialized :class:`Mesh` object.

        Other Parameters
        ----------------
        kwargs :
            Additional keyword arguments ignored by the function.

        """
        # Optional
        net = Net(net)

        # Get network data
        pairs = net[[*"AB"]].values.ravel()
        abscissa = net[["sA", "sB"]].values.ravel()
        points = net[["xA", "yA", "zA", "xB", "yB", "zB"]].values.reshape(-1, 3)

        # Initialize mesh DataFrame
        mesh = pd.DataFrame(
            data=np.c_[pairs, abscissa, points],
            columns=["fiber", *"sxyz"]
        )
        mesh["beam"] = 0
        mesh["constraint"] = 0
        # Convert type to int
        mesh.fiber = mesh.fiber.astype(int)

        # Set attributes
        mesh.attrs = net.attrs

        if len(net):
            # Inter-fiber connections
            mesh.constraint = mesh.index.values.reshape(-1, 2)[:, ::-1].ravel()
            # Sort nodes along fibers
            mesh.sort_values(by=["fiber", "s"], inplace=True)
            # Intra-fiber elements
            mesh.beam = np.hstack(
                mesh.groupby("fiber")
                .apply(lambda x: np.roll(x.index, -1))
            )
            # Reorder nodes
            indices = np.argsort(mesh.index).astype(int)
            mesh.index = indices[mesh.index]
            mesh.beam = indices[mesh.beam]
            mesh.constraint = indices[mesh.constraint]
            # Correct end nodes
            mask = (mesh.fiber.values == mesh.fiber.loc[mesh.constraint].values)
            mesh.constraint.values[mask] = mesh.index[mask]

        # Return the `Mesh` object
        return mesh

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
    def check(mesh=None):
        """
        Check that a :class:`Mesh` object is defined correctly.

        This method is automatically called by functions that use a :class:`Mesh` object as input.

        Parameters
        ----------
        mesh : pandas.DataFrame, optional
            Fiber mesh represented by a :class:`Mesh` object.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of :class:`Mesh` object.
        AttributeError
            If any attributes are missing from the dictionary :attr:`attrs`.
        IndexError
            If row indices are incorrectly defined:
                - Row indices are not unique in [0,..., m-1] where m is the number of nodes.
                - Node labels are not sorted.
        TypeError
            If labels are not integers.
        ValueError
            If fibers, beams or constraints are ill-defined, improperly connected or not sorted.

        Returns
        -------
        mesh : pandas.DataFrame
            Validated :class:`Mesh` object.

        .. TIP::
            - If `mesh` is None, it returns an empty :class:`Mesh` object.
            - If a "skip_check" flag is True in :attr:`attrs`, the check is passed.

        """
        if mesh is None:
            mesh = Mesh()

        if "skip_check" in mesh.attrs.keys() and mesh.attrs["skip_check"]:
            warnings.warn("{}.attrs['skip_check'] is active."
                          " Delete it or set it to False.".format(mesh.__class__),
                          UserWarning)
            # Return the `Mesh` object
            return mesh

        # Keys
        try:
            mesh[["fiber", *"sxyz", "beam", "constraint"]]
        except KeyError as e:
            raise KeyError(e)

        # Attributes
        if not ("n" in mesh.attrs.keys()):
            raise AttributeError("'n' is not in attribute dictionary.")
        if not ("size" in mesh.attrs.keys()):
            raise AttributeError("'size' is not in attribute dictionary.")
        if not ("periodic" in mesh.attrs.keys()):
            raise AttributeError("'periodic' is not in attribute dictionary.")

        # Indices
        if not np.all(np.unique(mesh.index) == np.arange(len(mesh))):
            raise IndexError("Row indices must be unique in [0,..., {}]."
                             .format(len(mesh) - 1))
        if not np.all(mesh.index == np.arange(len(mesh))):
            raise IndexError("Node labels must be sorted.")

        # Types
        if mesh["fiber"].values.dtype != int:
            raise TypeError("Fiber labels are not integers.")
        if mesh[["beam", "constraint"]].values.dtype != int:
            raise TypeError("Node labels are not integers.")

        # Data
        if len(mesh):
            if not (0 <= mesh.fiber.min()
                    and mesh.fiber.max() < mesh.attrs["n"]):
                raise ValueError("Fiber labels must be in [0,..., {}]."
                                 .format(mesh.attrs["n"] - 1))
            if not (0 <= mesh.beam.min() and mesh.beam.max() < len(mesh)):
                raise ValueError("Beam labels must be in [0,..., {}]."
                                 .format(len(mesh) - 1))
            if not (0 <= mesh.constraint.min()
                    and mesh.constraint.max() < len(mesh)):
                raise ValueError("Constraint labels must be in [0,..., {}]."
                                 .format(len(mesh) - 1))
            if not np.all(np.sort(mesh.fiber) == mesh.fiber):
                raise ValueError("Fibers are not sorted.")
            if not np.all(mesh.sort_values(by=["fiber", "s"]).index
                          == mesh.index):
                raise ValueError("Nodes are not ordered along fibers.")
            if not np.all(mesh.beam == np.hstack(
                    mesh.groupby("fiber").apply(lambda x: np.roll(x.index, -1))
            )):
                raise ValueError("Beams are not sorted.")
            if not np.all(mesh.constraint[mesh.constraint].values == mesh.index):
                raise ValueError("Constraints are not reciprocal.")
            end1 = mesh.loc[(mesh.groupby("fiber")
                             .apply(lambda x: x.index[0]))]
            end2 = mesh.loc[(mesh.groupby("fiber")
                             .apply(lambda x: x.index[-1]))]
            if (not np.all(end1.constraint == end1.index)
                    or not np.all(end2.constraint == end2.index)):
                raise ValueError(
                    "End node constraints must be equal to node labels.")

        # Return the `Mesh` object
        return mesh

    @staticmethod
    def isMesh(obj):
        """
        Check that an object can be instantiated as a :class:`Mesh`.

        Parameters
        ----------
        obj : Any
            Object to be tested.

        Returns
        -------
        bool
            Returns the test answer indicating whether the object can be a instantiated as :class:`Mesh`.

        """
        try:
            Mesh.check(obj)
            return True
        except:
            return False


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(10)
    # Build the fiber network
    net = Net(mat, periodic=True)
    # Create the fiber mesh
    mesh = Mesh(net)

    # Check data
    Mesh.check(mesh)  # or `mesh.check()`
    # -> returns `mesh` if correct, otherwise it raises an error.

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    if len(mesh):
        # Draw elements
        for i, j, k in tqdm(zip(mesh.index, mesh.beam, mesh.constraint),
                            total=len(mesh)):
            # Get element data
            a, b, c = mesh.iloc[[i, j, k]][[*"xyz"]].values
            if mesh.iloc[i].s < mesh.iloc[j].s:
                # Draw intra-fiber connection
                plt.plot(*np.c_[a, b],
                         c=plt.cm.tab10(mesh.fiber.iloc[i] % 10))
            if mesh.iloc[i].z < mesh.iloc[k].z:
                # Draw inter-fiber connection
                plt.plot(*np.c_[a, c], '--ok',
                         lw=1, mfc='none', ms=3, alpha=0.2)
            if mesh.iloc[i].fiber == mesh.iloc[k].fiber:
                # Draw fiber end nodes
                plt.plot(*np.c_[a, c], '+k', ms=3, alpha=0.2)
    # Set drawing box dimensions
    ax.set_xlim(-0.5 * mesh.attrs["size"], 0.5 * mesh.attrs["size"])
    ax.set_ylim(-0.5 * mesh.attrs["size"], 0.5 * mesh.attrs["size"])
    plt.show()
