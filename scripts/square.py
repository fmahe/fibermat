#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

from fibermat import Mat, Mesh, Net, Stack, Timoshenko, solve, vtk_mesh

# Generate a set of fibers
mat = Mat(3000, length=4, width=0.2, thickness=0.05, size=40, tensile=1600)
# Build the fiber network
net = Net(mat, periodic=True)
# Stack fibers
stack = Stack(net, threshold=1)
# Create the fiber mesh
mesh = Mesh(stack)
# Instantiate model
model = Timoshenko(mesh)

# Solve the mechanical packing problem
solution = solve(model, packing=4, itermax=10000)

# Export as VTK
msh = vtk_mesh(
    mesh,
    solution.displacement(1),
    solution.rotation(1),
    solution.force(1),
    solution.torque(1),
)
msh.plot(scalars="force", cmap=plt.cm.twilight_shifted)
# msh.save("outputs/msh.vtk")
