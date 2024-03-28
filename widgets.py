#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗹 Widgets
----------

Classes
-------
PipButton :
    Display a button widget to install Python packages in Jupyter notebooks.
MatplotlibButton :
    Display a button widget to toggle between matplotlib inline and qt display modes in Jupyter notebooks.

"""

import IPython
import ipywidgets
import os
import subprocess
from IPython.display import display, HTML


class PipButton(ipywidgets.Button):
    """
    A class to install Python packages in Jupyter notebooks.

    Parameters
    ----------
    packages : str,...
        Names of Python packages to install.

    Examples
    --------
    >>> pipButton = PipButton("matplotlib", "numpy", "pandas")
    >>> pipButton()

    Methods
    -------
    PipButton(*packages)
        Initialize a `PipButton` button widget.
    __call__()
        Display widget.

    """

    def __init__(self, *packages, option=None, **kwargs):
        """
        Initialize a `PipButton` button widget.

        Parameters
        ----------
        packages : str,...
            Names of Python packages to install.

        Other Parameters
        ----------------
        option : str, optional
            Additional options passed to `pip install` command.
        kwargs :
            Additional Python packages to install. The keys are passed to `pip install` and the values are used for `import`.

        """
        # Output stream
        output = ipywidgets.Output()

        pip_packages = [*packages, *kwargs.keys()]
        py_packages = [*packages, *kwargs.values()]

        if option is None:
            option = ""

        @output.capture(clear_output=True)
        def is_clicked(*args):
            self.button_style = 'primary'
            self.description = "Running..."
            # Installation command line
            result = os.system("pip install -q {} {}"
                               .format(option, " ".join(pip_packages)))
            if result == 0:
                # -> If successful
                self.button_style = 'success'
                self.description = "Installed"
                # Get package version
                cmd = 'python -Wignore -c "{}"'.format(
                    "".join([
                        "from {0} import __version__; print(__version__);"
                        .format(package) for package in py_packages
                    ])
                )
                versions = (subprocess.check_output(cmd, shell=True)
                            .decode()
                            .split('\n'))
                # Display package installation
                display(HTML(
                    "<table><thead><tr><th>Package</th><th>Version</th>"
                    + "</tr></thead><tbody>"
                    + "".join([
                        "<tr>"
                        "<th>{0}</th>"
                        "<th><a href='https://pypi.org/project/{0}'>{1}</a>"
                        "</tr>"
                        .format(package, version)
                        for package, version in zip(pip_packages, versions)])
                    + "</tbody></table>"
                ))
            else:
                # -> If unsuccessful
                self.button_style = 'danger'
                self.description = "Not Installed"

        # Initialize `PipButton` widget
        super().__init__()
        self.description = "Install packages"
        # Click events
        self.on_click(is_clicked)
        self.output = output

    # ~~~ Private methods ~~~ #

    def __call__(self):
        """Display widget."""
        display(self)
        display(self.output)


class MatplotlibButton(ipywidgets.Button):
    """
    A class to display a widget for switching between:
        - `%matplotlib inline`
        - `%matplotlib qt`
    in Jupyter notebooks.

    Examples
    --------
    >>> matplotlibButton = MatplotlibButton()
    >>> if "matplotlibButton" not in globals():
    >>>     matplotlibButton()

    Methods
    -------
    MatplotlibButton()
        Initialize a `MatplotlibButton` button widget.
    __call__()
        Display widget.

    """

    def __init__(self, **kwargs):
        """
        Initialize a `MatplotlibButton` widget.

        Parameters
        ----------
        kwargs :
            Additional keyword arguments to pass to `Button` constructor.

        """
        def is_clicked(*args):
            if self.description == "%matplotlib inline":
                self.description = "%matplotlib qt"
                self.button_style = 'success'
                IPython.get_ipython().run_cell("%matplotlib qt")
            else:
                self.description = "%matplotlib inline"
                self.button_style = 'primary'
                IPython.get_ipython().run_cell("%matplotlib inline")
            self.value = self.description

        # Initialize `MatplotlibButton` widget
        super().__init__(**kwargs)
        is_clicked()
        # Click events
        self.on_click(is_clicked)

    # ~~~ Private methods ~~~ #

    def __call__(self):
        """Display widget."""
        display(self)


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    pipButton = PipButton(
        "fibermat",
        "matplotlib",
        "numpy",
        "pandas",
        "pyvista",
        "scipy",
        **{"scikit-learn": "sklearn"},
        tqdm="tqdm",
        option="--upgrade"
    )
    pipButton()

    if "matplotlibButton" not in globals():
        matplotlibButton = MatplotlibButton()
