#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class Interpolate(interp1d):
    """
    A class for interpolating array values inherited from `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

    Parameters
    ----------
    y : array-like
        Data to be interpolated.

    Other Parameters
    ----------------
    size : int, optional
        Number of points used for interpolation.
    kwargs :
        Additional keyword arguments passed to `interp1d` constructor.

    Attributes
    ----------
    x : array-like
        Interpolation parameter.
    t : array-like
        Interpolation parameter between 0 and 1.
    dtype : type
        Data type used for interpolation.

    Methods
    -------
    :meth:`__call__` :
        Return interpolated data.

    ----

    """

    def __init__(self, y, size=None, **kwargs):
        """
        Initialize an Interpolation object.

        Parameters
        ----------
        y : array-like
            Data to be interpolated.

        Other Parameters
        ----------------
        size : int, optional
            Number of points used for interpolation.
        kwargs :
            Additional keyword arguments passed to `interp1d` constructor.

        """
        t = np.linspace(0, 1, len(y))
        f = interp1d(t, y, axis=0,
                     bounds_error=False, fill_value=y[-1], **kwargs)
        if size is not None:
            t = np.linspace(0, 1, size)
        # Initialize interpolation
        super().__init__(t, f(t), axis=0,
                         bounds_error=False, fill_value=y[-1], **kwargs)
        # Set attributes
        self.t = t
        self.dtype = np.array(y).dtype

    def __call__(self, t=None):
        """
        Return interpolated data.

        Parameters
        ----------
        t : array-like, optional
            Interpolation parameter between 0 and 1.

        Returns
        -------
        array-like
            Interpolated data.

        """
        if t is None:
            t = self.t
        return super().__call__(t).astype(self.dtype)


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    import numpy as np
    from matplotlib import pyplot as plt

    # Reference solution
    x = np.linspace(0, 10, 1001)
    y = np.sin(x)

    # Interpolation nodes
    X = x[::100]
    Y = y[::100]

    # Interpolated functions
    f_ = Interpolate(Y, size=11, kind='previous')
    g_ = Interpolate(Y, size=11, kind='linear')
    h_ = Interpolate(Y, size=11, kind='cubic')
    x_ = Interpolate(X, size=11)
    t = np.linspace(0, 1, 100)

    # Figure
    plt.figure()
    p, = plt.plot(x, y, label="Reference solution")
    plt.plot(X, Y, 'o', color=p.get_color(), zorder=np.inf, label="Interpolation nodes")
    p, = plt.plot(x_(t), f_(t), '--')
    plt.plot(x_(), f_(), 'o', ms=10, mfc='none', c=p.get_color())
    plt.plot([], [], 'o--', mfc='none', c=p.get_color(), label="Previous")
    p, = plt.plot(x_(t), g_(t), '--')
    plt.plot(x_(), g_(), 'o', ms=14, mfc='none', c=p.get_color())
    plt.plot([], [], 'o--', mfc='none', c=p.get_color(), label="Linear")
    p, = plt.plot(x_(t), h_(t), '--')
    plt.plot(x_(), h_(), 'o', ms=18, mfc='none', c=p.get_color())
    plt.plot([], [], 'o--', mfc='none', c=p.get_color(), label="Cubic")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Interpolations")
    plt.show()
