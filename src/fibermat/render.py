#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 Render
---------

Functions
---------
vtk_fiber() :
    Export a fiber as VTK mesh.
vtk_mat() :
    Export a `Mat` object as VTK mesh.
vtk_mesh() :
    Export a `Mesh` object as VTK mesh.

"""

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
    Export a fiber as VTK mesh.

    length : float
        Fiber length (mm). Default is 25 mm.
    width : float
        Fiber width (mm). Default is 1 mm.
    thickness : float
        Fiber thickness (mm). Default is 1 mm.
    x : float
        Fiber position: X-coordinate (mm). Default is 0 mm.
    y : float
        Fiber position: Y-coordinate (mm). Default is 0 mm.
    z : float
        Fiber position: Z-coordinate (mm). Default is 0 mm.
    u : float
        Fiber orientation: X-component. Default is 1.
    v : float
        Fiber orientation: Y-component. Default is 0.
    w : float
        Fiber orientation: Z-component. Default is 0.
    shear : float
        Shear modulus (MPa). Default is 1 MPa.
    tensile : float
        Tensile modulus (MPa). Default is ∞ MPa.
    index : int, optional
        Fiber label.
    r_resolution : int
        Number of elements along the radius of the fiber. Default is 1.
    theta_resolution : int
        Number of points on the circular face of the fiber. Default is 8.
    z_resolution : int
        Number of points along the length of the fiber. Default is 20.
    **kwargs
        Additional keyword arguments ignored by the function.

    Notes
    -----
    If `index` is not None, the following fields are added to the VTK mesh:
        - "fiber" : fiber index
        - "lbh" : fiber dimensions (mm)
        - "xyz" : local fiber coordinates (mm)
        - "uvw" : fiber orientation vector
        - "G" : shear modulus (MPa)
        - "E" : tensile modulus (MPa)

    Returns
    -------
    pv.StructuredGrid
        VTK mesh.

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
    Export a `Mat` object as VTK mesh.

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a `Mat` object.
    **kwargs
        Additional keyword arguments passed to `vtk_fiber` function.

    Notes
    -----
    The following fields are added to the VTK mesh:
        - "fiber" : fiber index
        - "lbh" : fiber dimensions (mm)
        - "xyz" : local fiber coordinates (mm)
        - "uvw" : fiber orientation vector
        - "G" : shear modulus (MPa)
        - "E" : tensile modulus (MPa)

    Returns
    -------
    pv.UnstructuredGrid
        VTK mesh.

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
    Export a `Mesh` object as VTK mesh.

    Parameters
    ----------
    mat : pandas.DataFrame, optional
        Set of fibers represented by a `Mat` object.
    mesh : pandas.DataFrame, optional
        Fiber mesh represented by a `Mesh` object.
    displacement : numpy.ndarray, optional
        Displacement vector.
    rotation : numpy.ndarray, optional
        Rotation vector.
    force : numpy.ndarray, optional
        Load vector.
    moment : numpy.ndarray, optional
        Torque vector.
    **kwargs
        Additional keyword arguments passed to `vtk_fiber` function.

    Notes
    -----
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

    Returns
    -------
    pv.UnstructuredGrid
        VTK mesh.

    """
    # Optional
    mat = Mat.check(mat)
    mesh = Mesh.check(mesh)

    fibers = []  # : list to store individual fiber meshes
    by_fiber = mesh.groupby("fiber").apply(lambda x: x)

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
        fiber = by_fiber.loc[i]
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

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(mat, net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Export as VTK
    vtk_mesh(mat, mesh).plot()
