#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

"""

from fibermat.mat import *
from fibermat.net import *
from fibermat.mesh import *
from fibermat.interpolation import *
from fibermat.model import *
from fibermat.solver import *
from fibermat.render import *
from fibermat.widgets import *

__author__ = "François Mahé"
__authors__ = ["François Mahé"]
__contact__ = "francois.mahe@ens-rennes.fr"
__copyright__ = "Copyright (c) 2024 François Mahé"
__credits__ = ["François Mahé"]
__date__ = "19/03/2024"
__deprecated__ = False
__email__ = "francois.mahe@ens-rennes.fr"
__header__ = """
████████╖██┐██╖                   ████┐  ████╖       ██╖
██╔═════╝└─┘██║    ██████╖█╖████╖ ██╔██ ██╔██║       ██║
█████─╖  ██╖██████╖██║ ██║██╔══█║ ██║ ███╓╜██║██████╖█████╖
██╔═══╝  ██║██║ ██║██╟───╜██║  └╜ ██║ ╘══╝ ██║██║ ██║██╔══╝
██║      ██║█████╓╜│█████╖██║     ██║      ██║███ ██║█████╖
╚═╝      ╚═╝╚════╝ ╘═════╝╚═╝     ╚═╝      ╚═╝╚══╧══╝╚════╝
 █████┐       █████┐       ██┐
██╔══██┐     ██╓──██┐      └─┘       █╖████╖
 ██╖ └─█████ └███ └─┘      ██╖██████╖██╔══█║
██╔╝  ██╔══██   ███╖ ████╖ ██║██║ ██║██║  └╜
│██████╓╜   ██████╓╜ ╚═══╝ ██║██████║██║
╘══════╝    ╘═════╝        ╚═╝██╔═══╝╚═╝
      Rennes                  ██║
                              ╚═╝
"""
__license__ = "MIT"
__maintainer__ = "François Mahé"
__status__ = "Production"
__version__ = "1.0"
