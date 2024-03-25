# Welcome to FiberMat’s documentation!

<a href="https://github.com/fmahe/fibermat">
    <img alt="banner" src="https://github.com/fmahe/fibermat/raw/main/images/banner.png">
</a>

[![pypi version](https://img.shields.io/pypi/v/fibermat?logo=pypi)](https://pypi.org/project/fibermat/)
[![GitHub Badge](https://img.shields.io/badge/Github-fibermat-blue?logo=github)](https://github.com/fmahe/fibermat)
[![Read the Docs](https://img.shields.io/readthedocs/fibermat)](https://fibermat.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![François Mahé](https://img.shields.io/badge/Author-François%20Mahé-green)](https://img.shields.io/badge/francois.mahe@ens--rennes.fr-Univ%20Rennes,%20ENS%20Rennes,%20CNRS,%20IPR%20--%20UMR%206251,%20F--35000%20Rennes,%20France-royalblue)
[![Mail](https://img.shields.io/badge/✉-francois.mahe@ens--rennes.fr-royalblue)](mailto:francois.mahe@ens-rennes.fr)

<details>
<summary>
<b> License </b> <a id="license"></a>

</summary>

```
                                        ██╖
████████╖  ████┐  ████╖       ██╖      ██╓╜
██╔═════╝  ██╔██ ██╔██║       ██║    ██████╖
█████─╖    ██║ ███╓╜██║██████╖██████╖██║ ██║
██╔═══╝    ██║ ╘══╝ ██║██║ ██║██╓─██║██╟───╜
██║    ██┐ ██║      ██║███ ██║██║ ██║│█████╖
╚═╝    └─┘ ╚═╝      ╚═╝╚══╧══╝╚═╝ ╚═╝╘═════╝
 █████┐       █████┐       ██┐
██╔══██┐     ██╓──██┐      └─┘       █╖████╖
 ██╖ └─█████ └███ └─┘      ██╖██████╖██╔══█║
██╔╝  ██╔══██   ███╖ ████╖ ██║██║ ██║██║  └╜
│██████╓╜   ██████╓╜ ╚═══╝ ██║██████║██║
╘══════╝    ╘═════╝        ╚═╝██╔═══╝╚═╝
      Rennes                  ██║
                              ╚═╝
@author: François Mahé
@mail: francois.mahe@ens-rennes.fr
(Univ Rennes, ENS Rennes, CNRS, IPR - UMR 6251, F-35000 Rennes, France)

@project: FiberMat
@version: v1.0

License:
--------
MIT License

Copyright (c) 2024 François Mahé

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Description:
------------
A mechanical solver to simulate fiber packing and perform statistical analysis.

References:
-----------
Mahé, F. (2023). Statistical mechanical framework for discontinuous composites:
  application to the modeling of flow in SMC compression molding (Doctoral
  dissertation, École centrale de Nantes).

```
</details>

**FiberMat** is a mechanical solver to simulate fiber packing and perform statistical analysis. It generates realistic 3D fiber mesostructures and computes internal forces and deformations.

This code is the result of thesis work that can be found in:
> [Mahé, F. (2023). Statistical mechanical framework for discontinuous composites:
  application to the modeling of flow in SMC compression molding (Doctoral
  dissertation, École centrale de Nantes).](https://theses.hal.science/tel-04189271/)

## Installation

Requirements
+ `python>=3.8`
+ `jupyter`

### Install the package with Pip

Run the following commands:
```shell
# Install `FiberMat`
pip install fibermat

# Try it out
python -c "import fibermat"

```

### Install the package in an Anaconda environment

1. Create a conda environment:
    ```shell
    # Create conda environment
    conda create -n fibermat python=3.11

    # Activate the environment
    conda activate fibermat

    # Optional
    pip install jupyter

    ```

2. Install FiberMat:
    ```shell
    # Install `FiberMat`
    pip install --upgrade fibermat

    # Try it out
    python -c "import fibermat"

    ```

3. To leave `fibermat` environment, run ``conda deactivate``.

### Directly from the sources

Clone the repository and run `pip` command:
```shell
# Clone the repository
git clone git@github.com:fmahe/fibermat.git
cd ./fibermat

# Install `FiberMat`
pip install --upgrade .

```

### Build the sources

FiberMat's documentation is created using [Sphinx](https://www.sphinx-doc.org/en/master/) [<sup>[1]</sup>](#note-1).

1. Clone the repository:
    ```shell
    # Clone the repository
    git clone git@github.com:fmahe/fibermat.git
    cd ./fibermat

    ```

2. Install dependencies required to compile documentation:

    - Install the packages listed in `requirements.txt` with Pip:
        ```shell
        # Install dependencies
        pip install -r requirements.txt

        ```

    - You can also create an environment that already satisfies requirements instead:
        ```shell
        # Create an environment from the `environment.yml` file
        conda env create -n fibermat -f ./environment.yml

        # Activate the environment
        conda activate fibermat

        ```

3. Execute `./make` script:
    ```shell
    # Build the sources
    ./make --all

    ```

<a id="note-1"> [1] </a> : Here a tutorial (fr) for Sphinx: [Introduction à Sphinx, un outil de documentation puissant](https://blog.flozz.fr/2020/09/07/introduction-a-sphinx-un-outil-de-documentation-puissant/).

## Documentation

See the tutorial in `jupyter-notebook.ipynb`.

## Example

```python
from fibermat import *

# Generate a set of fibers
mat = Mat(100, length=25, width=2, thickness=0.5, tensile=2500)
# Build the fiber network
net = Net(mat, periodic=True)
# Stack fibers
stack = Stack(mat, net, threshold=10)
# Create the fiber mesh
mesh = Mesh(stack)

# Solve the mechanical packing problem
K, C, u, f, F, H, Z, rlambda, mask, err = solve(
    mat, mesh,
    packing=4, itermax=1000, lmin=0.01, coupling=0.99, interp_size=100
)

# Export as VTK
vtk = vtk_mesh(mat, mesh,
               *u(1).reshape(-1, 2).T,
               *(f(1) @ C).reshape(-1, 2).T)
vtk.plot(scalars="force", cmap=plt.cm.twilight_shifted)
vtk.save("outputs/vtk.vtk")

```

![example](https://github.com/fmahe/fibermat/raw/main/images/example.png)
