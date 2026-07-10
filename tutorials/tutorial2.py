import gmsh
import sys

import meshio
import pyvista as pv
import numpy as np

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

lc = 3e-2
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(.1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(.1, .3, 0, lc, 3)
gmsh.model.geo.addPoint(0, .3, 0, lc, 4)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(3, 2, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
gmsh.model.addPhysicalGroup(2, [1], name="My surface")

gmsh.model.geo.addPoint(0, .4, 0, lc, 5)
gmsh.model.geo.addLine(4, 5, 5)

gmsh.model.geo.translate([(0, 5)], -0.02, 0, 0)
gmsh.model.geo.rotate([(0, 5)], 0, 0.3, 0, 0, 0, 1, -np.pi / 4)

ov = gmsh.model.geo.copy([(0, 3)])
gmsh.model.geo.translate(ov, 0, 0.05, 0)

# The new point tag is available in ov[0][1], and can be used to create new
# lines:
gmsh.model.geo.addLine(3, ov[0][1], 7)
gmsh.model.geo.addLine(ov[0][1], 5, 8)
gmsh.model.geo.addCurveLoop([5, -8, -7, 3], 10)
gmsh.model.geo.addPlaneSurface([10], 11)

# In the same way, we can translate copies of the two surfaces 1 and 11 to the
# right with the following command:
ov = gmsh.model.geo.copy([(2, 1), (2, 11)])
gmsh.model.geo.translate(ov, 0.12, 0, 0)

print("New surfaces " + str(ov[0][1]) + " and " + str(ov[1][1]))

gmsh.model.geo.addPoint(0., 0.3, 0.12, lc, 100)
gmsh.model.geo.addPoint(0.1, 0.3, 0.12, lc, 101)
gmsh.model.geo.addPoint(0.1, 0.35, 0.12, lc, 102)

gmsh.model.geo.synchronize()
xyz = gmsh.model.getValue(0, 5, [])
gmsh.model.geo.addPoint(xyz[0], xyz[1], 0.12, lc, 103)

gmsh.model.geo.addLine(4, 100, 110)
gmsh.model.geo.addLine(3, 101, 111)
gmsh.model.geo.addLine(6, 102, 112)
gmsh.model.geo.addLine(5, 103, 113)
gmsh.model.geo.addLine(103, 100, 114)
gmsh.model.geo.addLine(100, 101, 115)
gmsh.model.geo.addLine(101, 102, 116)
gmsh.model.geo.addLine(102, 103, 117)

gmsh.model.geo.addCurveLoop([115, -111, 3, 110], 118)
gmsh.model.geo.addPlaneSurface([118], 119)
gmsh.model.geo.addCurveLoop([111, 116, -112, -7], 120)
gmsh.model.geo.addPlaneSurface([120], 121)
gmsh.model.geo.addCurveLoop([112, 117, -113, -8], 122)
gmsh.model.geo.addPlaneSurface([122], 123)
gmsh.model.geo.addCurveLoop([114, -110, 5, 113], 124)
gmsh.model.geo.addPlaneSurface([124], 125)
gmsh.model.geo.addCurveLoop([115, 116, 117, 114], 126)
gmsh.model.geo.addPlaneSurface([126], 127)

gmsh.model.geo.addSurfaceLoop([127, 119, 121, 123, 125, 11], 128)
gmsh.model.geo.addVolume([128], 129)

ov2 = gmsh.model.geo.extrude([ov[1]], 0, 0, 0.12)

gmsh.model.geo.mesh.setSize([(0, 103), (0, 105), (0, 109), (0, 102), (0, 28),
                             (0, 24), (0, 6), (0, 5)], lc * 3)

gmsh.model.geo.synchronize()


gmsh.model.addPhysicalGroup(3, [129, 130], 1, "The volume")

gmsh.model.mesh.generate(3)


gmsh.write("./tutorials/mesh_files/t2.msh")
gmsh.finalize()

mesh = meshio.read("./tutorials/mesh_files/t2.msh")
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

