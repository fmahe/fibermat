#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ipywidgets
import numpy as np
from IPython.display import display, HTML


class Settings(dict):
    """
    A class for setting parameters in Jupyter notebooks.

    Methods
    -------
    :meth:`__iter__`
        Iterate over widgets.
    :meth:`__getitem__`
        Get the value of a given widget.
    :meth:`__setitem__`
        Set the value of a given widget.
    :meth:`__call__`
        Display widgets for setting parameters.

    ----

    """
    def __init__(self, n=0, length=25., width=1., thickness=1., size=50.,
                 theta=1., shear=1., tensile=np.inf, seed=0, periodic=True,
                 threshold=None, packing=1., itermax=1000, tol=1e-6, lmin=None,
                 lmax=None, coupling=1.0, interp_size=None, verbose=True):
        """
        Initialize a `Settings` widget.

        Parameters
        ----------
        TODO: Complete documentation.

        """
        # Input widgets
        self.n = ipywidgets.IntText(value=n, description="N")
        self.length = ipywidgets.FloatText(value=length, step=5.0, description="LENGTH")
        self.width = ipywidgets.FloatText(value=width, step=1.0, description="WIDTH")
        self.thickness = ipywidgets.FloatText(value=thickness, step=0.1, description="THICKNESS")
        self.size = ipywidgets.FloatText(value=size, step=10.0, description="SIZE")
        self.theta = ipywidgets.FloatSlider(value=theta, min=0.0, max=1.0, step=0.01, readout_format=".1%", description="THETA")
        self.shear = ipywidgets.FloatText(value=(shear if shear != np.inf else 1.0), step=1.0, layout=ipywidgets.Layout(width='100px'))
        self.shear.switch = ipywidgets.Checkbox(value=not (shear is np.inf), description="SHEAR", layout=ipywidgets.Layout(width='196px'))
        self.tensile = ipywidgets.FloatText(value=(tensile if tensile  != np.inf else shear * (length / thickness) ** 2), step=1.0, layout=ipywidgets.Layout(width='100px'))
        self.tensile.switch = ipywidgets.Checkbox(value=not (tensile is np.inf), description="TENSILE", layout=ipywidgets.Layout(width='196px'))
        self.seed = ipywidgets.IntText(value=(seed if seed is not None else 0), layout=ipywidgets.Layout(width='100px'))
        self.seed.switch = ipywidgets.Checkbox(value=not (seed is None), description="SEED", layout=ipywidgets.Layout(width='196px'))

        self.periodic = ipywidgets.Checkbox(value=periodic, description="PERIODIC")
        self.threshold = ipywidgets.FloatText(value=(threshold if threshold is not None else 20 * thickness), step=0.5, layout=ipywidgets.Layout(width='100px'))
        self.threshold.switch = ipywidgets.Checkbox(value=not (threshold is None), description="THRESHOLD", layout=ipywidgets.Layout(width='196px'))

        self.packing = ipywidgets.FloatText(value=packing, step=0.5, description="PACKING")
        self.itermax = ipywidgets.FloatLogSlider(value=itermax, min=0, max=6, step=1, readout_format=".0e", description="ITERMAX")
        self.tol = ipywidgets.FloatLogSlider(value=tol, min=-10.0, max=-1.0, step=1.0, readout_format=".0e", description="TOL")

        self.lmin = ipywidgets.FloatText(value=(lmin if lmin is not None else 0.01), step=0.01, layout=ipywidgets.Layout(width='100px'))
        self.lmin.switch = ipywidgets.Checkbox(value=not (lmin is None), description="LMIN", layout=ipywidgets.Layout(width='196px'))
        self.lmax = ipywidgets.FloatText(value=(lmax if lmax is not None else length), step=0.01, layout=ipywidgets.Layout(width='100px'))
        self.lmax.switch = ipywidgets.Checkbox(value=not (lmax is None), description="LMAX", layout=ipywidgets.Layout(width='196px'))
        self.coupling = ipywidgets.FloatText(value=coupling, step=0.01, description="COUPLING")

        self.interp_size = ipywidgets.IntText(value=(interp_size if interp_size is not None else 100), layout=ipywidgets.Layout(width='100px'))
        self.interp_size.switch = ipywidgets.Checkbox(value=not (interp_size is None), description="INTERPSIZE", layout=ipywidgets.Layout(width='196px'))
        self.verbose = ipywidgets.Checkbox(value=verbose, description="VERBOSE")

        def shear_on_changed(*args):
            self.shear.disabled = not self.shear.switch.value

        def tensile_on_changed(*args):
            self.tensile.disabled = not self.tensile.switch.value

        def seed_on_changed(*args):
            self.seed.disabled = not self.seed.switch.value

        def threshold_on_changed(*args):
            self.threshold.disabled = not self.threshold.switch.value

        def lmin_on_changed(*args):
            self.lmin.disabled = not self.lmin.switch.value

        def lmax_on_changed(*args):
            self.lmax.disabled = not self.lmax.switch.value

        def interp_size_on_changed(*args):
            self.interp_size.disabled = not self.interp_size.switch.value

        # Click events
        self.shear.switch.observe(shear_on_changed) ; shear_on_changed()
        self.tensile.switch.observe(tensile_on_changed) ; tensile_on_changed()
        self.seed.switch.observe(seed_on_changed) ; seed_on_changed()
        self.threshold.switch.observe(threshold_on_changed) ; threshold_on_changed()
        self.lmin.switch.observe(lmin_on_changed) ; lmin_on_changed()
        self.lmax.switch.observe(lmax_on_changed) ; lmax_on_changed()
        self.interp_size.switch.observe(interp_size_on_changed) ; interp_size_on_changed()

        # Initialize `Settings` object
        super().__init__(dict(
            n=self.n,
            length=self.length,
            width=self.width,
            thickness=self.thickness,
            size=self.size,
            theta=self.theta,
            shear=ipywidgets.HBox([self.shear.switch, self.shear]),
            tensile=ipywidgets.HBox([self.tensile.switch, self.tensile]),
            seed=ipywidgets.HBox([self.seed.switch, self.seed]),
            #
            periodic=self.periodic,
            threshold=ipywidgets.HBox([self.threshold.switch, self.threshold]),
            #
            packing=self.packing,
            itermax=self.itermax,
            tol=self.tol,
            #
            lmin=ipywidgets.HBox([self.lmin.switch, self.lmin]),
            lmax=ipywidgets.HBox([self.lmax.switch, self.lmax]),
            coupling=self.coupling,
            #
            interp_size=ipywidgets.HBox([self.interp_size.switch, self.interp_size]),
            verbose=self.verbose,
        ))

    # ~~~ Private methods ~~~ #

    def __iter__(self):
        """
        Iterate over widgets with syntax `*self`.

        Returns
        -------
        iterator
            Set of widgets as iterator.

        Notes
        -----
        Required for single star operator.

        """
        return iter(self.values())

    def __getitem__(self, name):
        """
        Get the value of a given widget with syntax `self[name]` and `**self`.

        Parameters
        ----------
        name : str
            Name of the widget.

        Notes
        -----
        Required for bracket accessor and double star operator.

        """
        if name == "theta":
            return self.__getattribute__(name).value * np.pi
        elif name in ["shear", "tensile"]:
            return (self.__getattribute__(name).value
                    if self.__getattribute__(name).switch.value
                    else np.inf)
        elif name in ["seed", "threshold", "lmin", "lmax", "interp_size"]:
            return (self.__getattribute__(name).value
                    if self.__getattribute__(name).switch.value
                    else None)
        elif name == "itermax":
            return int(self.__getattribute__(name).value)
        else:
            # Default getter
            return self.__getattribute__(name).value

    def __setitem__(self, name, value):
        """
        Set the value of a given widget with syntax `self[name] = value`.

        Parameters
        ----------
        name : str
            Name of the widget.
        value : Any
            Value of the widget.

        Notes
        -----
        Required for bracket accessor and setting via :meth:`__call__` method.

        """
        if name in ["shear", "tensile"]:
            self.__getattribute__(name).switch.value = (False if value is np.inf
                                                        else True)
            if value is not np.inf:
                self.__getattribute__(name).value = value
        elif name in ["seed", "threshold", "lmin", "lmax", "interp_size"]:
            self.__getattribute__(name).switch.value = (False if value is None
                                                        else True)
            if value is not None:
                self.__getattribute__(name).value = value
        else:
            # Default setter
            self.__getattribute__(name).value = value

    def __call__(self, **kwargs):
        """
        Display widgets for setting parameters with syntax `self(**kwargs)`.

        Other Parameters
        ----------------
        kwargs :
            Additional keyword arguments to set widget values.

        """
        for name, value in kwargs.items():
            self.__setitem__(name, value)
        if len(kwargs) == 0:
            display(*self)


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    if "inputs" not in globals():
        inputs = Settings()
