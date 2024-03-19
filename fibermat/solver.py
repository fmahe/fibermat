#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧩 Solver
---------

Functions
---------
solver(mat, mesh) :
    Mechanical fiber packing solver.

"""

import numpy as np
import scipy as sp
import warnings
from tqdm import tqdm

from fibermat import Mat, Mesh, stiffness, constraint, Interpolation


def solver(mat, mesh, packing=5., solver=sp.sparse.linalg.spsolve,
           itermax=1000, tol=1e-6, errtol=1e-6, interp_size=None,
           verbose=True, **kwargs):
    """
    Mechanical fiber packing by solving the quadratic programming problem:

        min_{𝒖}(½𝒖·𝕂·𝒖 - 𝑭·𝒖) s.t. ℂ·𝒖 ≤ 𝑯 and 𝒖 ≤ 0

    Constraints are solved using Lagrangian multipliers :

        min_{𝒖, 𝒇}(½𝒖·𝕂·𝒖 - 𝑭·𝒖 - 𝒇·(𝑯 - ℂ·𝒖))

        s.t. ℂ·𝒖 ≤ 𝑯, 𝒖 ≤ 0, 𝒇 ≥ 0 and 𝒇·(𝑯 - ℂ·𝒖) = 0

    which leads to solving the inequality system:

        ⎡ K  Cᵀ⎤⎡u⎤ ≤ ⎡F⎤
        ⎣ C  0 ⎦⎣f⎦   ⎣H⎦

    by an Updated Lagrangian Approach.

    Parameters
    ----------
    mat : pandas.DataFrame
        Set of fibers represented by a `Mat` object.
    mesh : pandas.DataFrame
        Fiber mesh represented by a `Mesh` object.
    packing : float
        Targeted value of packing. Must be greater than 1.0.
    solver : callable
        Sparse solver. Default is `scipy.sparse.linalg.spsolve`.
    itermax : int
        Maximum number of solver iterations. Default is 1000.
    tol : float
        Tolerance used for contact detection (mm). Default is 1e-6 mm.
    errtol : float
        Tolerance for the numerical error. Default is 1e-6.
    interp_size : int, optional
        Size of array used for interpolation. Default is None.
    verbose : bool
        If True, displays a progress bar during simulation. Default is True.
    **kwargs
        Additional keyword arguments passed to matrix constructors.

    Returns
    -------
    tuple
        K : sparse matrix
            Stiffness matrix (symmetric positive-semi definite).
        C : sparse matrix
            Constraint matrix.
        u : Interpolation
            Displacement vector.
        f : Interpolation
            Force vector.
        F : Interpolation
            Load vector.
        H : Interpolation
            Upper bound vector.
        Z : Interpolation
            Upper boundary position.
        rlambda : Interpolation
            Compaction.
        mask : Interpolation
            Active rows and columns in the system of equations.
        err : Interpolation
            Numerical error of the linear solver.

    """
    # Assembly quadratic programming system
    K, u, F, du, dF = stiffness(mat, mesh, **kwargs)
    C, f, H, df, dH = constraint(mat, mesh, **kwargs)
    P = sp.sparse.bmat([[K, C.T], [C, None]], format='csc')
    x = np.r_[u, f]
    q = np.r_[F, H]
    dx = np.r_[du, df]
    dq = np.r_[dF, dH]

    u, f = np.split(x, [K.shape[0]])  # Memory-shared
    F, H = np.split(q, [K.shape[0]])  # Memory-shared
    du, df = np.split(dx, [K.shape[0]])  # Memory-shared
    dF, dH = np.split(dq, [K.shape[0]])  # Memory-shared

    u_ = [u.copy()]
    f_ = [f.copy()]
    F_ = [F.copy()]
    H_ = [H.copy()]
    Z_ = [(mesh.z.values + 0.5 * mesh.h.values).max()]
    rlambda_ = [1.0]
    mask_ = [(np.real(q - P @ x) <= tol)]
    err_ = [0]

    # Incremental solver
    with tqdm(total=itermax, desc="Packing: {:.2}".format(rlambda_[-1]),
              disable=not verbose) as pbar:
        i = 0
        while (i < pbar.total) and (rlambda_[-1] < packing):
            # Solve step
            dx *= 0
            # mask = [True] * K.shape[0] + [*(C @ u >= H - tol)]
            mask = (np.real(q - P @ x) <= tol)
            mask &= np.array(np.sum(np.abs(np.real(P)), axis=0) > 0).ravel()
            # Solve linear problem
            sol = solver(P[np.ix_(mask, mask)], dq[mask])
            dx[mask] += np.real(sol)
            # Calculate error
            err = np.linalg.norm(P[np.ix_(mask, mask)] @ dx[mask] - dq[mask])
            # Calculate evolution
            d = np.real(H - C @ u)
            v = np.real(dH - C @ du)

            try:
                # Calculate the next step
                dU = -min(d[(d > 0) & (v > 0)] / v[(d > 0) & (v > 0)])
                # Stopping criteria
                stop = False
                if err > errtol:
                    if verbose:
                        warnings.warn("Stopping criteria: err = {}".format(err),
                                      UserWarning)
                    stop = True
                if stop:
                    raise ValueError
            except ValueError:
                break

            # Jump to the next step
            x += dx * dU
            q += dq * dU
            # Store results
            u_ += [u.copy()]
            f_ += [f.copy()]
            F_ += [F.copy()]
            H_ += [H.copy()]
            Z_ += [Z_[-1] + dU]
            rlambda_ += [Z_[0] / Z_[-1]]
            mask_ += [mask.copy()]
            err_ += [err.copy()]

            # Update
            i += 1
            pbar.set_description("Packing: {:.2}".format(rlambda_[-1]))
            pbar.update()

    # Interpolate results
    with warnings.catch_warnings():
        # Ignore warning messages due to infinite values in 𝑯
        warnings.filterwarnings('ignore')
        u = Interpolation(u_, size=interp_size)
        f = Interpolation(f_, size=interp_size)
        F = Interpolation(F_, size=interp_size)
        H = Interpolation(H_, size=interp_size)
        Z = Interpolation(Z_, size=interp_size)
        rlambda = Interpolation(rlambda_)
        mask = Interpolation(mask_, kind='previous')
        err = Interpolation(err_, kind='previous')

    # Return interpolated results
    return K, C, u, f, F, H, Z, rlambda, mask, err


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    from fibermat import Net, Stack

    # Generate a set of fibers
    mat = Mat(100, tensile=625)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(mat, net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solver(mat, mesh, lmin=0.01, coupling=0.99)
