#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import warnings
from tqdm import tqdm

from fibermat import Mat, Net, Stack, Mesh, stiffness, constraint, Interpolate


def solver(mat, mesh, packing=1., solve=sp.sparse.linalg.spsolve,
           itermax=1000, tol=1e-6, errtol=1e-6, interp_size=None,
           verbose=True, **kwargs):
    r"""An iterative mechanical solver for fiber packing problems. It solves the *quadratic programming problem*:

    .. MATH::
        \min_{\mathbf{u}, \mathbf{f}} \left( \frac{1}{2} \, \mathbf{u} \, \mathbb{K} \, \mathbf{u} - \mathbf{F} \, \mathbf{u} - \mathbf{f} \, (\mathbf{H} - \mathbb{C} \, \mathbf{u}) \right)
    .. MATH::
        \quad s.t. \quad \mathbb{C} \, \mathbf{u} \leq \mathbf{H} \, ,
        \quad \mathbf{u} \leq 0 \, ,
        \quad \mathbf{f} \geq 0
        \quad and \quad \mathbf{f} \, (\mathbf{H} - \mathbb{C} \, \mathbf{u}) = 0

    where:
        - :math:`\mathbf{u}` is the vector of generalized displacements (*unknowns of the problem*).
        - :math:`\mathbf{f}` is the vector of generalized forces (*unknowns Lagrange multipliers*).
        - :math:`\mathbb{K}` is the stiffness matrix of the fiber set.
        - :math:`\mathbf{F}` is the vector of external efforts.
        - :math:`\mathbb{C}` is the matrix of non-penetration constraints.
        - :math:`\mathbf{H}` is the vector of minimal distances between fibers (minimal distances).

    The *mechanical equilibrium* allows reformulating the problem as a system of inequalities:

    .. MATH::
        \Rightarrow \quad \left[\begin{matrix}
            \mathbb{K} & \mathbb{C}^T \\
            \mathbb{C} & 0
        \end{matrix}\right] \binom{\mathbf{u}}{\mathbf{f}}
            \leq \binom{\mathbf{F}}{\mathbf{H}}

    which is solved using an iterative *Updated Lagrangian Approach*.

    .. HINT::
        Models used to build the matrices are implemented in :ref:`🔧 Model`:
            - 𝕂 and 𝑭 : :func:`~.model.stiffness`.
            - ℂ and 𝑯 : :func:`~.model.constraint`.

    Parameters
    ----------
    mat : pandas.DataFrame
        Set of fibers represented by a :class:`Mat` object.
    mesh : pandas.DataFrame
        Fiber mesh represented by a :class:`Mesh` object.
    packing : float, optional
        Targeted value of packing. Must be greater than 1. Default is 1.0.

    Returns
    -------
    tuple
        K : sparse matrix
            Stiffness matrix (symmetric positive-semi definite).
        C : sparse matrix
            Constraint matrix.
        u : Interpolate
            Displacement vector.
        f : Interpolate
            Force vector.
        F : Interpolate
            Load vector.
        H : Interpolate
            Upper-bound vector.
        Z : Interpolate
            Upper-boundary position.
        rlambda : Interpolate
            Compaction.
        mask : Interpolate
            Active rows and columns in the system of inequations.
        err : Interpolate
            Numerical error of the linear solver.

    .. SEEALSO::
        Simulation results are given as functions of a pseudo-time parameter (between 0 and 1) using :class:`~.interpolation.Interpolate` objects.

    Other Parameters
    ----------------
    solve : callable, optional
        Sparse solver. It is a callable object that takes as inputs a sparse symmetric matrix 𝔸 and a vector 𝒃 and returns the solution 𝒙 of the linear system: 𝔸 𝒙 = 𝒃. Default is `scipy.sparse.linalg.spsolve`.
    itermax : int, optional
        Maximum number of solver iterations. Default is 1000.
    tol : float, optional
        Tolerance used for contact detection (mm). Default is 1e-6 mm.
    errtol : float, optional
        Tolerance for the numerical error. Default is 1e-6.
    interp_size : int, optional
        Size of array used for interpolation. Default is None.
    verbose : bool, optional
        If True, displays a progress bar during simulation. Default is True.
    kwargs :
        Additional keyword arguments passed to matrix constructors.

    """
    # Assemble the quadratic programming system
    # TODO: pass a model as argument instead
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
            sol = solve(P[np.ix_(mask, mask)], dq[mask])
            dx[mask] += np.real(sol)
            # Calculate error
            err = np.linalg.norm(P[np.ix_(mask, mask)] @ dx[mask] - dq[mask])
            # Calculate evolution
            # TODO: use P and q instead
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
        u = Interpolate(u_, size=interp_size)
        f = Interpolate(f_, size=interp_size)
        F = Interpolate(F_, size=interp_size)
        H = Interpolate(H_, size=interp_size)
        Z = Interpolate(Z_, size=interp_size)
        rlambda = Interpolate(rlambda_)
        mask = Interpolate(mask_, kind='previous')
        err = Interpolate(err_, kind='previous')

    # Return interpolated results
    return K, C, u, f, F, H, Z, rlambda, mask, err


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    from fibermat import *

    # Generate a set of fibers
    mat = Mat(100)
    # Build the fiber network
    net = Net(mat)
    # Stack fibers
    stack = Stack(mat, net)
    # Create the fiber mesh
    mesh = Mesh(stack)

    # Solve the mechanical packing problem
    K, C, u, f, F, H, Z, rlambda, mask, err = solver(
        mat, mesh, packing=4, lmin=0.01, coupling=0.99
    )

    # Deform the mesh
    mesh.z += u(1)[::2]

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
