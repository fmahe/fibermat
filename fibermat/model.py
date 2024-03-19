#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Model
--------

Functions
---------
stiffness(mat, mesh) :
    Assembly quadratic system to be minimized.
constraint(mat, mesh) :
    Assembly linear constraints.
plot_system(K, u, F, du, dF, C, f, H, df, dH) :
    Visualize the system of equations.

"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from fibermat import Mat, Mesh


def stiffness(mat, mesh, lmin=None, lmax=None, coupling=1.0, **kwargs):
    """
    Assembly quadratic system to be minimized:

        𝕂·𝒖 = 𝑭

    Parameters
    ----------
    mat : pandas.DataFrame
        Set of fibers represented by a `Mat` object.
    mesh : pandas.DataFrame
        Fiber mesh represented by a `Mesh` object.
    lmin : float, optional
        Lower bound used to rescale beam lengths (mm).
    lmax : float, optional
        Upper bound used to rescale beam lengths (mm).
    coupling : float
        Coupling numerical constant between zero and one. Default is 1.0.
    **kwargs :
        Additional keyword arguments ignored by the function.

    Returns
    -------
    tuple
        K : sparse matrix
            Stiffness matrix (symmetric positive-semi definite).
        u : numpy.ndarray
            Displacement vector.
        F : numpy.ndarray
            Load vector.
        du : numpy.ndarray
            Incremental displacement vector.
        dF : numpy.ndarray
            Incremental load vector.

    """
    # Optional
    Mat.check(mat)
    Mesh.check(mesh)

    # Get mesh data
    mask = (mesh.index.values < mesh.beam.values)
    fiber = mesh.fiber[mask].values
    i = mesh.index[mask].values
    j = mesh.beam[mask].values

    # Get material data
    fiber = mat.loc[fiber]
    l = mesh.s.loc[j].values - mesh.s.loc[i].values
    if lmin is None:
        lmin = np.min(l)
    if lmax is None:
        lmax = np.max(l)
    l = interp1d([min(np.min(l), lmin), max(np.max(l), lmax)], [lmin, lmax])(l)

    # Timoshenko number : Ψ² = E / G * (h / l) ^ 2
    k0 = np.pi / 4 * fiber[[*"Gbh"]].prod(axis=1).values / l
    k0 /= (1 + (fiber.G / fiber.E) * (l / fiber.h) ** 2)
    k1 = k0 * l / 2
    k1 *= coupling  # Numerical regularization
    k2 = k0 * l ** 2 / 3
    k2 += k0 * (fiber.E / fiber.G) * fiber.h ** 2
    k3 = k0 * l ** 2 / 2
    k4 = k2 - k3
    i *= 2
    j *= 2

    # Create stiffness data
    row = np.array([
        i + 0, i + 0, i + 0, i + 0,
        i + 1, i + 1, i + 1, i + 1,
        j + 0, j + 0, j + 0, j + 0,
        j + 1, j + 1, j + 1, j + 1,
    ]).ravel()
    col = np.array([
        i + 0, i + 1, j + 0, j + 1,
        i + 0, i + 1, j + 0, j + 1,
        i + 0, i + 1, j + 0, j + 1,
        i + 0, i + 1, j + 0, j + 1,
    ]).ravel()
    data = np.array([
         k0,  k1, -k0,  k1,
         k1,  k2, -k1, -k4,
        -k0, -k1,  k0, -k1,
         k1, -k4, -k1,  k2
    ]).ravel()

    # Initialize 𝕂 matrix
    K = sp.sparse.coo_matrix((data, (row, col)),
                             shape=(2 * len(mesh), 2 * len(mesh)))

    # Initialize 𝒖 and 𝑭 vectors
    u = np.zeros(K.shape[0])
    F = np.zeros(K.shape[0])
    du = np.zeros(K.shape[0])
    dF = np.zeros(K.shape[0])

    return K, u, F, du, dF


def constraint(mat, mesh, **kwargs):
    """
    Assembly linear constraints:

        ℂ·𝒖 ≤ 𝑯, 𝒖 ≤ 0

    with Lagrangian multipliers:

        𝒇 ≥ 0 and 𝒇·(𝑯 - ℂ·𝒖) = 0

    Parameters
    ----------
    mat : pandas.DataFrame
        Set of fibers represented by a `Mat` object.
    mesh : pandas.DataFrame
        Fiber mesh represented by a `Mesh` object.
    **kwargs
        Additional keyword arguments ignored by the function.

    Returns
    -------
    tuple
        C : sparse matrix
            Constraint matrix.
        f : numpy.ndarray
            Force vector.
        H : numpy.ndarray
            Upper bound vector.
        df : numpy.ndarray
            Incremental force vector.
        dH : numpy.ndarray
            Incremental upper bound vector.

    """
    # Optional
    Mat.check(mat)
    Mesh.check(mesh)

    # Get mesh data
    mask = (mesh.index.values <= mesh.constraint.values)
    i = mesh.index[mask].values
    j = mesh.constraint[mask].values
    k = np.arange(len(i))
    O = i * 0  # : zero
    I = O + 1  # : one

    # Get material data
    mesh["h"] = mat.h.loc[mesh.fiber].values
    zi = mesh.z.loc[i].values
    zj = mesh.z.loc[j].values
    hi = mesh.h.loc[i].values
    hj = mesh.h.loc[j].values
    Z = (mesh.z.values + 0.5 * mesh.h.values).max()  # : upper boundary position
    i *= 2
    j *= 2
    k *= 3

    # Create constraint data
    row = np.array([k, k + 1, k + 1, k + 2]).ravel()
    col = np.array([i, i, j, j]).ravel()
    data = np.array([-I, I, -I, I]).ravel()

    # Initialize ℂ matrix
    C = sp.sparse.coo_matrix((data, (row, col)),
                             shape=(3 * len(mesh[mask]), 2 * len(mesh)))

    # Initialize 𝒇 and 𝑯 vectors
    f = np.zeros(C.shape[0])
    H = np.zeros(C.shape[0])
    df = np.zeros(C.shape[0])
    dH = np.zeros(C.shape[0])
    # (X₁ + u₁) ≥ ½h₁ ⟺ -u₁ ≤ X₁ - ½h₁
    H[::3] += zi - 0.5 * hi
    # (X₂ + u₂) - (X₁ + u₁) ≥ ½(h₁ + h₂) ⟺ u₁ - u₂ ≤ X₂ - X₁ - ½(h₁ + h₂)
    H[1::3] += zj - zi - 0.5 * (hi + hj)
    # (X₂ + u₂) ≤ Z - ½h₂ ⟺ u₂ ≤ Z - X₂ - ½h₂
    H[2::3] += Z - zj - 0.5 * hj
    dH[2::3] = 1
    # For end nodes
    H[1::3][mesh[mask].index == mesh[mask].constraint.values] = np.inf

    return C, f, H, df, dH


def plot_system(K, u, F, du, dF, C, f, H, df, dH, ax=None, tol=1e-6):
    """
    Visualize the system of equations and calculate the step error.

    Parameters
    ----------
    K : sparse matrix
        Stiffness matrix (symmetric positive-semi definite).
    u : numpy.ndarray
        Displacement vector.
    F : numpy.ndarray
        Load vector.
    du : numpy.ndarray
        Incremental displacement vector.
    dF : numpy.ndarray
        Incremental load vector.
    C : sparse matrix
        Constraint matrix.
    f : numpy.ndarray
        Force vector.
    H : numpy.ndarray
        Upper bound vector.
    df : numpy.ndarray
        Incremental force vector.
    dH : numpy.ndarray
        Incremental upper bound vector.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes.
    tol : float, optional
        Tolerance used for contact detection (mm). Default is 1e-6 mm.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Matplotlib axes.

    """
    # Assembly quadratic programming system
    P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')
    x = np.r_[u, f]
    q = np.r_[F, H]
    dx = np.r_[du, df]
    dq = np.r_[dF, dH]

    mask0 = np.array([True] * K.shape[0] + [False] * C.shape[0])
    D0 = sp.sparse.diags(mask0.astype(float))
    mask1 = np.array([False] * K.shape[0] + [*(
                np.isclose(C @ u, H) & np.tile([True, False, False],
                                               C.shape[0] // 3))])
    D1 = sp.sparse.diags(mask1.astype(float))
    mask2 = np.array([False] * K.shape[0] + [*(
                np.isclose(C @ u, H) & np.tile([False, True, False],
                                               C.shape[0] // 3))])
    D2 = sp.sparse.diags(mask2.astype(float))
    mask3 = np.array([False] * K.shape[0] + [*(
                np.isclose(C @ u, H) & np.tile([False, False, True],
                                               C.shape[0] // 3))])
    D3 = sp.sparse.diags(mask3.astype(float))
    mask4 = np.array([False] * K.shape[0] + [*(~np.isclose(C @ u, H))])
    D4 = sp.sparse.diags(mask4.astype(float))

    # Figure
    if ax is None:
        fig, ax = plt.subplots()
    ax.spy(D0 @ P @ D0, ms=3, color='black', alpha=0.5, label="stiffness")
    ax.spy(D2 @ P + P @ D2, ms=3, color='tab:blue', alpha=0.25, label="inner")
    ax.spy(D1 @ P + P @ D1, ms=3, color='tab:green', alpha=0.5, label="lower")
    ax.spy(D3 @ P + P @ D3, ms=3, color='tab:red', alpha=0.5, label="upper")
    ax.spy(D4 @ P + P @ D4, ms=1, color='gray', zorder=-1, alpha=0.1,
           label="inactive")
    ax.legend()

    mask = (np.real(q - P @ x) <= tol)
    mask &= np.array(np.sum(np.abs(np.real(P)), axis=0) > 0).ravel()

    # Solve linear problem
    sol = sp.sparse.linalg.spsolve(P[np.ix_(mask, mask)], dq[mask])
    dx[mask] += np.real(sol)
    # Calculate error
    err = np.linalg.norm(P[np.ix_(mask, mask)] @ sol - dq[mask])
    print(err)

    return ax


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    from fibermat import Mesh, Stack

    # Linear
    mat = Mat(1, length=1, width=1, thickness=1, shear=1, tensile=np.inf)
    net = Net(mat)
    mesh = Mesh(net)
    print("Linear (Ψ² ≫ 1) =")
    print(4 / np.pi * stiffness(mat, mesh, coupling=1)[0].todense())
    print()

    # Timoshenko
    mat = Mat(1, length=1, width=1, thickness=1, shear=2, tensile=2)
    net = Net(mat)
    mesh = Mesh(net)
    print("Timoshenko (Ψ² = 1) = 1 / 2 *")
    print(4 / np.pi * stiffness(mat, mesh, coupling=1)[0].todense())
    print()

    # Euler
    mat = Mat(1, length=1, width=1, thickness=1, shear=1e12, tensile=12)
    net = Net(mat)
    mesh = Mesh(net)
    print("Euler (Ψ² ≪ 1) = 1 / 12 *")
    print(4 / np.pi * stiffness(mat, mesh, coupling=1)[0].todense())
    print()

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(mat, net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Assembly quadratic programming system
    K, u, F, du, dF = stiffness(mat, mesh)
    C, f, H, df, dH = constraint(mat, mesh)
    plot_system(K, u, F, du, dF, C, f, H, df, dH)
    plt.show()
