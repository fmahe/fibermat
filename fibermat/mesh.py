#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📐 Mesh
-------

Classes
-------
Mesh :
    A class representing a mesh structure for a set of discontinuous fibers.

"""

import numpy as np
import pandas as pd

from fibermat import Net


class Mesh(pd.DataFrame):
    """
    A class representing a mesh structure for a set of discontinuous fibers.
    It defines:
        - the intra-fiber elements (beam elements)
        - the inter-fiber connections (constraint elements)

    Parameters
    ----------
    net : pandas.DataFrame, optional
         Fiber network represented by a `Net` object.

    Attributes
    ----------
    attrs : dictionary
        Global attributes:
            - n : int, Number of fibers.
            - size : float, Box dimensions (mm).
            - periodic : bool, Periodicity.
    index : pandas.Index
        Node label.
    fiber : pandas.Series
        Fiber label.
    x : pandas.Series
        Node position: X-coordinate (mm).
    y : pandas.Series
        Node position: Y-coordinate (mm).
    z : pandas.Series
        Node position: Z-coordinate (mm).
    s : pandas.Series
        Node curvilinear abscissa (mm).
    constraint : pandas.Series
        Index of the connected node.
    beam : pandas.Series
        Index of the next node along the fiber.

    Methods
    -------
    init()
        Create a discontinuous fiber mesh.
    check()
        Check that `Mesh` object is defined correctly.

    Examples
    --------
    ```python
        >>> # Generate a set of fibers
        >>> mat = Mat(**inputs)
        >>> # Build the fiber network
        >>> net = Net(mat, **inputs)
        >>> # Create the fiber mesh
        >>> mesh = Mesh(net)
        >>> display(mesh.groupby(by="fiber").apply(lambda x: x, include_groups=False))

    ```

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the `Mesh` object.

        Parameters
        ----------
        *args :
            Additional positional arguments passed to the constructor.
        **kwargs :
            Additional keyword arguments passed to the constructor.

        See also
        --------
        Mesh.init :
            Create a discontinuous fiber mesh.

        """
        if (len(args) and isinstance(args[0], pd.DataFrame)
                and not isinstance(args[0], Net)):
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
    def init(net=None):
        """
        Create a discontinuous fiber mesh.

        Parameters
        ----------
        net : pandas.DataFrame, optional
             Fiber network represented by a `Net` object.

        Returns
        -------
        mesh : pandas.DataFrame
            Initialized `Mesh` object.

        """
        # Optional
        net = Net.check(net)

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
            mesh.beam = np.r_[*(
                mesh.groupby("fiber")
                .apply(lambda x: np.roll(x.index, -1), include_groups=False)
            )]
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

    # ~~~ Public methods ~~~ #

    @staticmethod
    def check(mesh=None):
        """
        Check that `Mesh` object is defined correctly.

        Parameters
        ----------
        mesh : pandas.DataFrame, optional
            Fiber mesh represented by a `Mesh` object. Default is None.

        Raises
        ------
        KeyError
            If any keys are missing from the columns of `Mesh` object.
        AttributeError
            If any attributes are missing from the dictionary `mesh.attrs`.
        IndexError
            if row indices are incorrectly defined:
                - Row indices are not unique in [0,..., n-1] where n is the number of nodes.
                - Node labels are not sorted.
        TypeError
            If labels are not integers.
        ValueError
            If fibers, beams or constraints are ill-defined or not sorted.

        Returns
        -------
        mesh : pandas.DataFrame
            Validated `Mesh` object.

        Notes
        -----
        If `mesh` is None, it returns an empty `Mesh` object.

        """
        if mesh is None:
            mesh = Mesh()

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
            if not np.all(mesh.beam == np.r_[*(
                    mesh.groupby("fiber").apply(lambda x: np.roll(x.index, -1),
                                                include_groups=False)
            )]):
                raise ValueError("Beams are not sorted.")
            if not np.all(mesh.constraint[mesh.constraint].values == mesh.index):
                raise ValueError("Constraints are not reciprocal.")
            end1 = mesh.loc[(mesh.groupby("fiber")
                             .apply(lambda x: x.index[0],
                                    include_groups=False))]
            end2 = mesh.loc[(mesh.groupby("fiber")
                             .apply(lambda x: x.index[-1],
                                    include_groups=False))]
            if (not np.all(end1.constraint == end1.index)
                    or not np.all(end2.constraint == end2.index)):
                raise ValueError(
                    "End node constraints must be equal to node labels.")

        # Return the `Mesh` object
        return mesh


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from fibermat import Mat

    # Generate a set of fibers
    mat = Mat(10)
    # Build the fiber network
    net = Net(mat)
    # Create the fiber mesh
    mesh = Mesh(net)

    # Figure
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', aspect='equal',
                                           xlabel="X", ylabel="Y", zlabel="Z"))
    ax.view_init(azim=45, elev=30, roll=0)
    for i, j, k in tqdm(zip(mesh.index, mesh.beam, mesh.constraint),
                        total=len(mesh)):
        a, b, c = mesh.iloc[[i, j, k]][[*"xyz"]].values
        if mesh.iloc[i].s < mesh.iloc[j].s:
            # Draw intra-fiber connections
            plt.plot(*np.c_[a, b],
                     c=plt.cm.tab10(mesh.fiber.iloc[i] % 10))
        if mesh.iloc[i].z < mesh.iloc[k].z:
            # Draw inter-fiber connections
            plt.plot(*np.c_[a, c], '--ok',
                     lw=1, mfc='none', ms=3, alpha=0.2)
        if mesh.iloc[i].fiber == mesh.iloc[k].fiber:
            # Draw fiber end nodes
            plt.plot(*np.c_[a, c], '+k', ms=3, alpha=0.2)
    ax.set_xlim(-0.5 * mesh.attrs["size"], 0.5 * mesh.attrs["size"])
    ax.set_ylim(-0.5 * mesh.attrs["size"], 0.5 * mesh.attrs["size"])
    plt.show()
