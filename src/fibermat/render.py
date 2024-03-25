#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv
from scipy.interpolate import CubicHermiteSpline
from sklearn.neighbors import KDTree
from tqdm import tqdm

from fibermat import Mat, Net, Stack, Mesh


def vtk_fiber(length=25., width=1., thickness=1., x=0., y=0., z=0.,
              u=1., v=0., w=0., shear=1., tensile=np.inf, index=None,
              r_resolution=1, theta_resolution=8, z_resolution=20, **kwargs):
    """
    Export a fiber as VTK mesh using `pyvista.CylinderStructured <https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.CylinderStructured.html>`_.

    Parameters
    ----------
    length : float, optional
        Fiber length (mm). Default is 25 mm.
    width : float, optional
        Fiber width (mm). Default is 1 mm.
    thickness : float, optional
        Fiber thickness (mm). Default is 1 mm.
    x : float, optional
        Fiber position: X-coordinate (mm). Default is 0 mm.
    y : float, optional
        Fiber position: Y-coordinate (mm). Default is 0 mm.
    z : float, optional
        Fiber position: Z-coordinate (mm). Default is 0 mm.
    u : float, optional
        Fiber orientation: X-component. Default is 1.
    v : float, optional
        Fiber orientation: Y-component. Default is 0.
    w : float, optional
        Fiber orientation: Z-component. Default is 0.
    shear : float, optional
        Shear modulus (MPa). Default is 1 MPa.
    tensile : float, optional
        Tensile modulus (MPa). Default is ∞ MPa.
    index : int, optional
        Fiber label.

    Returns
    -------
    pyvista.StructuredGrid
        VTK mesh.

    Other Parameters
    ----------------
    r_resolution : int, optional
        Number of elements along the radius of the fiber. Default is 1.
    theta_resolution : int, optional
        Number of points on the circular face of the fiber. Default is 8.
    z_resolution : int, optional
        Number of points along the length of the fiber. Default is 20.
    kwargs :
        Additional keyword arguments ignored by the function.

    .. NOTE::
        If `index` is not None, the following fields are added to the VTK mesh:
            - "fiber" : fiber index
            - "lbh" : fiber dimensions (mm)
            - "xyz" : local fiber coordinates (mm)
            - "uvw" : fiber orientation vector
            - "G" : shear modulus (MPa)
            - "E" : tensile modulus (MPa)

    """
    # Create the VTK mesh (cylindrical structured grid)
    vtk = pv.CylinderStructured(radius=np.linspace(0.5, 0, r_resolution + 1),
                                theta_resolution=theta_resolution,
                                z_resolution=z_resolution)

    l, b, h = length, width, thickness

    # Add fields to mesh data
    if index is not None:
        vtk["fiber"] = np.full(len(vtk.points), index)
        vtk["lbh"] = np.tile([l, b, h], (len(vtk.points), 1))
        vtk["xyz"] = vtk.points * np.array([[l, b, h]])
        vtk["uvw"] = np.tile([u, v, w], (len(vtk.points), 1))
        vtk["G"] = np.full(len(vtk.points), shear)
        vtk["E"] = np.full(len(vtk.points), tensile)

    # Transform the mesh (scale, rotate, and translate)
    vtk.scale([l, b, h], inplace=True)
    pv.translate(vtk,
                 center=(x, y, z),
                 direction=(u, v, w))

    # Return VTK mesh
    return vtk


def vtk_mat(mat=None, **kwargs):
    """
    Export a :class:`Mat` object as VTK mesh.

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a :class:`Mat` object.

    Returns
    -------
    pyvista.UnstructuredGrid
        VTK mesh.

    Other Parameters
    ----------------
    kwargs :
        Additional keyword arguments passed to :meth:`vtk_fiber` function.

    .. NOTE::
        The following fields are added to the VTK mesh:
            - "fiber" : fiber index
            - "lbh" : fiber dimensions (mm)
            - "xyz" : local fiber coordinates (mm)
            - "uvw" : fiber orientation vector
            - "G" : shear modulus (MPa)
            - "E" : tensile modulus (MPa)

    """
    # Optional
    mat = Mat.check(mat)

    fibers = []  # : list to store individual fiber meshes

    for i in tqdm(mat.index, desc="Create VTK mat"):
        # Get fiber
        fiber = mat.loc[i].astype(float)
        # Create the VTK mesh (cylindrical structured grid)
        fiber_mesh = vtk_fiber(*fiber[[*"lbhxyzuvwGE"]].values,
                               index=i,
                               **kwargs)
        # Append fiber mesh to list
        fibers.append(fiber_mesh)

    # Combine all individual fiber meshes into a single VTK mesh
    return pv.MultiBlock(fibers).combine()


def vtk_mesh(mat=None, mesh=None, displacement=None, rotation=None,
             force=None, moment=None, **kwargs):
    """
    Export a :class:`Mesh` object as VTK mesh.

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a :class:`Mat` object.
    mesh : pandas.DataFrame, optional
        Fiber mesh represented by a :class:`Mesh` object.

    Returns
    -------
    pyvista.UnstructuredGrid
        VTK mesh.

    Other Parameters
    ----------------
    displacement : numpy.ndarray, optional
        Displacement field.
    rotation : numpy.ndarray, optional
        Rotation field.
    force : numpy.ndarray, optional
        Load field.
    moment : numpy.ndarray, optional
        Torque field.
    kwargs :
        Additional keyword arguments passed to :meth:`vtk_fiber` function.

    .. HINT::
        The following fields are added to the VTK mesh:
            - "fiber" : fiber index
            - "lbh" : fiber dimensions (mm)
            - "xyz" : local fiber coordinates (mm)
            - "uvw" : fiber orientation vector
            - "G" : shear modulus (MPa)
            - "E" : tensile modulus (MPa)
        If `displacement` is not None:
            - "displacement" : displacement field (mm)
            - "rotation" : rotation field (rad)
            - "curvature" : curvature field (1 / mm)
        If `force` is not None:
            - "force" : force field (N)

    """
    # Optional
    mat = Mat.check(mat)
    mesh = Mesh.check(mesh)

    fibers = []  # : list to store individual fiber meshes
    by_fiber = mesh.groupby("fiber")

    for i in tqdm(mat.index, desc="Create VTK mat"):
        # Get fiber
        fiber = mat.loc[i].astype(float)
        # Create the VTK mesh (cylindrical structured grid)
        fiber_mesh = vtk_fiber(*fiber[[*"lbhxyzuvwGE"]].values,
                               index=i,
                               **kwargs)
        # Append fiber mesh to list
        fibers.append(fiber_mesh)

        # Prepare interpolation data
        fiber = by_fiber.get_group(i)
        s = fiber.s.values[:, None]
        x = fiber_mesh["xyz"][:, [0]] * 0.9999
        k = KDTree(s).query(x, return_distance=False).ravel()
        s, x = s.ravel(), x.ravel()
        # Correct indices (s_k <= x_i < s_{k+1}, k \in [-1, n])
        k = k * (x > s[k]) + (k - 1) * (x <= s[k])
        # Indices containing relative distances (np.floor(j) = k)
        j = ((x - s[k]) / (s[k + 1] - s[k]))
        # Correct issues for periodic mesh
        j = np.array([*j])
        j[j == np.inf] = 0
        # Add to vtk_fiber
        j += fiber.index[k]
        fiber_mesh["node"] = j

    # Combine all individual fiber meshes into a single VTK mesh
    vtk = pv.MultiBlock(fibers).combine()

    if len(mat):
        # Interpolate fields
        s = np.arange(len(mesh))
        x = vtk["node"]
        if displacement is not None:
            if rotation is None:
                rotation = 0 * displacement
            displacement = CubicHermiteSpline(s, displacement, rotation)
            vtk["displacement"] = np.zeros(vtk.points.shape)
            vtk["displacement"][:, 2] = displacement(x)
            vtk["rotation"] = displacement.derivative()(x)
            vtk["curvature"] = displacement.derivative(2)(x)
            vtk.points += vtk["displacement"]
        if force is not None:
            if moment is None:
                moment = 0 * force
            force = CubicHermiteSpline(s, force, moment)
            vtk["force"] = force(x)

    # Periodic boundary conditions (optional)
    if len(mat) and mesh.attrs["periodic"]:
        X = Y = mat.attrs["size"]
        Z1, Z2 = np.min(vtk.points), np.max(vtk.points)
        # Duplicate mesh for periodic conditions
        vtk = pv.MultiBlock([
            vtk,
            vtk.copy().translate([-X, 0, 0]),
            vtk.copy().translate([X, 0, 0]),
            vtk.copy().translate([0, -Y, 0]),
            vtk.copy().translate([0, Y, 0]),
            vtk.copy().translate([-X, -Y, 0]),
            vtk.copy().translate([-X, Y, 0]),
            vtk.copy().translate([X, -Y, 0]),
            vtk.copy().translate([X, Y, 0]),
        ]).combine().clip_box([-X, X, -Y, Y, Z1, Z2], invert=False)

    # Return VTK mesh
    return vtk


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    from fibermat import *

    # Create a VTK fiber
    vtk_fiber().plot()

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat, periodic=True)
    # Stack fibers
    stack = Stack(mat, net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Create a VTK mat
    vtk_mat(mat).plot()

    # Create a VTK mesh
    vtk_mesh(mat, mesh).plot()

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solve(
        mat, mesh, packing=4, lmin=0.01, coupling=0.99
    )

    # Export as VTK
    vtk = vtk_mesh(mat, mesh,
                   *u(1).reshape(-1, 2).T,
                   *(f(1) @ C).reshape(-1, 2).T)
    vtk.plot(scalars="force", cmap=plt.cm.twilight_shifted)
