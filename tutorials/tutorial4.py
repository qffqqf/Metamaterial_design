import gmsh
import sys

import meshio
import pyvista as pv
import numpy as np

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

# Volumes can be constructed from (closed) curve loops thanks to the
# `addThruSections()' function
gmsh.model.occ.addCircle(0, 0, 0, 0.5, 1, zAxis=[0,0,1])
gmsh.model.occ.addCurveLoop([1], 1)
gmsh.model.occ.addCircle(0.1, 0.05, 1, 0.2, 2)
gmsh.model.occ.addCurveLoop([2], 2)
gmsh.model.occ.addCircle(-0.1, -0.1, 2, 0.1, 3)
gmsh.model.occ.addCurveLoop([3], 3)
gmsh.model.occ.addThruSections([1, 2, 3], 1)
# 3. Synchronize the OCC CAD to the Gmsh model
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.08)
gmsh.model.mesh.generate(3)


gmsh.write("./tutorial/mesh_files/t4.msh")
gmsh.finalize()

mesh = meshio.read("./tutorial/mesh_files/t4.msh")
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

