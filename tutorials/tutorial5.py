import gmsh
import sys

import meshio
import pyvista as pv
import numpy as np

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

# OpenCASCADE also allows general extrusions along a smooth path. Let's first
# define a spline curve:
nturns = 1.
npts = 20
r = 1.
h = 1. * nturns
p = []
for i in range(0, npts):
    theta = i * 2 * np.pi * nturns / npts
    gmsh.model.occ.addPoint(r * np.cos(theta), r * np.sin(theta),
                            i * h / npts, 1, 1000 + i)
    p.append(1000 + i)
gmsh.model.occ.addSpline(p, 1000)

# A wire is like a curve loop, but open:
gmsh.model.occ.addWire([1000], 1000)

# We define the shape we would like to extrude along the spline (a disk):
gmsh.model.occ.addDisk(1, 0, 0, 0.2, 0.2, 1000)
gmsh.model.occ.rotate([(2, 1000)], 0, 0, 0, 1, 0, 0, np.pi / 2)

# We extrude the disk along the spline to create a pipe (other sweeping types
# can be specified; try e.g. 'Frenet' instead of 'DiscreteTrihedron'):
gmsh.model.occ.addPipe([(2, 1000)], 1000, 'DiscreteTrihedron')

# We delete the source surface, and increase the number of sub-edges for a
# nicer display of the geometry:
gmsh.model.occ.remove([(2, 1000)])
gmsh.option.setNumber("Geometry.NumSubEdges", 1000)

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.08)
gmsh.model.mesh.generate(3)


gmsh.write("./tutorials/mesh_files/t5.msh")
gmsh.finalize()

mesh = meshio.read("./tutorials/mesh_files/t5.msh")
print(mesh)
mesh.cell_sets.clear()
mesh_pv = pv.from_meshio(mesh)

volume_mesh = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TETRA)

volume_mesh.plot(
    show_edges=True,
    color="lightblue",
    edge_color="black",
    line_width=0.5,
    background="white",
    opacity=0.3,
    style="surface",
    cpos = "xy"
) 

