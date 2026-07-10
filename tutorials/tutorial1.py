import gmsh
import sys

import meshio
import pyvista as pv

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

lc = 0.03
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
p2 = gmsh.model.geo.addPoint(.1, 0, 0, lc, 2)
p3 = gmsh.model.geo.addPoint(.1, .3, 0, lc, 3)
p4 = gmsh.model.geo.addPoint(0, .3, 0, lc)
gmsh.model.geo.addLine(p1, p2, 1)
gmsh.model.geo.addLine(p2, p3, 2)
gmsh.model.geo.addLine(p3, p4, 3)
gmsh.model.geo.addLine(p4, p1, 4)
gmsh.model.geo.addCurveLoop([4, 1, 2, 3], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, [1, 2, 4], name="My curves")
gmsh.model.addPhysicalGroup(2, [1], name="My surface")
gmsh.model.mesh.generate(2)

gmsh.write("./tutorial/mesh_files/t1.msh")
gmsh.finalize()

mesh = meshio.read("./tutorial/mesh_files/t1.msh")
print(mesh)
mesh.cell_sets.clear()
mesh_pv = pv.from_meshio(mesh)
triangles_only = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TRIANGLE)

triangles_only.plot(
    show_edges=True,
    color="lightblue",
    edge_color="black",
    line_width=0.5,
    background="white",
    opacity=0.3,
    style="surface",
    cpos = "xy"
) 

