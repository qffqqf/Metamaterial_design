from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import netgen.gui  
import ngsolve
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from ngsolve import Draw

# -----------------------
# 0) User parameters
# -----------------------
# Duct size (x = propagation direction)
Lx = 0.40
Hy = 0.13
Hz = 0.13

# Plate position and thicknesses
xp = 0.20          # steel plate starts at x=xp
ts = 0.005         # steel thickness
tr = 0.005         # rubber layer thickness (downstream of steel)
r_hole = 0.05     # hole radius

maxh = 0.03
minh = 0.01
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 2

# -----------------------
# 1) Geometry (duct + steel plate w/ hole + rubber layer + rubber plug sealing hole)
# -----------------------
yc, zc = Hy/2, Hz/2
duct = Box((0,0,0), (Lx,Hy,Hz))
# Steel plate box (full cross-section)
steel_box = Box((xp, 0, 0), (xp+ts, Hy, Hz))
# Hole through steel thickness (axis along +x)
hole_steel = Cylinder(Pnt(xp, yc, zc), gp_Vec(1,0,0), r_hole, ts)
steel = steel_box - hole_steel
steel.mat("steel")

# Rubber layer downstream, annulus (hole continues) ...
rubber = Box((xp+ts, 0, 0), (xp+ts+tr, Hy, Hz))
tip_rubber = Cylinder(Pnt(xp, yc, zc), gp_Vec(1,0,0), r_hole/3, ts)
rubber = rubber + tip_rubber
rubber.mat("rubber")

# Fluid = duct minus solid union
solid_union = steel + rubber
fluid = duct - solid_union
fluid.mat("fluid")

for f in fluid.faces:
    f.name = "fluid_boundary"

combined_geo = Glue([fluid, solid_union])

for f in solid_union.faces:
    if f.name == "fluid_boundary":
        f.name = "fsi_interface"

geo = OCCGeometry(solid_union)
mesh = Mesh(geo.GenerateMesh(mp=mp))


interface_marker = mesh.BoundaryCF({"fsi_interface": 1}, default=0)
# Draw(interface_marker, mesh, "FSI_Interface")
settings = {"mesh": False, "coloring": (0, 0, 0)}
# Draw(mesh, settings=settings)
# input("Mesh generated successfully! Press Enter to continue...")


import pyvista as pv
import numpy as np
import ngsolve as ng

def ngs_to_pyvista(mesh):
    # ---- Extract vertices ----
    # v.point returns a tuple of coordinates
    points = np.array([list(v.point) for v in mesh.vertices], dtype=np.float64)

    # ---- Build cells (VTK format) ----
    cells = []
    cell_types = []
    
    # Iterate over VOLUME elements specifically
    for el in mesh.Elements(ng.VOL):
        vert_indices = [v.nr for v in el.vertices]   # 0-based vertex indices
        n_verts = len(vert_indices)
        
        # VTK cell format: [n_points, id1, id2, ..., idn]
        cells.append(n_verts)
        cells.extend(vert_indices)
        
        # Safely assign VTK cell type based on NGSolve element type
        if el.type == ng.TET:
            cell_types.append(pv.CellType.TETRA)
        elif el.type == ng.HEX:
            cell_types.append(pv.CellType.HEXAHEDRON)
        elif el.type == ng.PRISM:
            cell_types.append(pv.CellType.WEDGE)
        elif el.type == ng.PYRAMID:
            cell_types.append(pv.CellType.PYRAMID)
        else:
            # Fallback: remove the appended vertices if the shape is unsupported
            cells = cells[: -(n_verts + 1)]

    # Convert to numpy arrays with correct integer types required by PyVista
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    # Create PyVista unstructured grid
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    return grid

# Convert mesh
pv_mesh = ngs_to_pyvista(mesh)

# Create plotter
plotter = pv.Plotter()

# Add the mesh (translucent with edges)
plotter.add_mesh(pv_mesh, color='grey', show_edges=True)

# Compute bounding box
xmin, xmax, ymin, ymax, zmin, zmax = pv_mesh.bounds

# 2. Define your custom xmin and xmax values here
custom_xmin = Lx*0.2  # Change this to whatever you want
custom_xmax = Lx*0.75   # Change this to whatever you want

# 3. Create the box with your updated bounds tuple
modified_bounds = (custom_xmin, custom_xmax, ymin, ymax, zmin, zmax)
box = pv.Box(modified_bounds)

# Add semi-transparent bounding box
plotter.add_mesh(box, color='lightblue', opacity=0.2, style='surface')
plotter.add_mesh(box, color='lightblue', style='wireframe', line_width=2)


center_x = Lx*0.3
center_y = (ymin + ymax) / 2
center_z = (zmin + zmax) / 2

# Example 1: An XY-plane cutting horizontally through the center of the duct
# i_hat=(1,0,0) and j_hat=(0,1,0) define the orientation span of the plane
plane = pv.Plane(
    center=(center_x, center_y, center_z),
    direction=(1, 0, 0),              # Normal vector pointing straight up (+Z)
    i_size=ymax - ymin,                # Width along Y
    j_size=zmax - zmin                # Height along Z
)
# Add the plane to the plotter with custom opacity and color
plotter.add_mesh(plane, color='pink', opacity=0.4, show_edges=False)

# Show the combined view
plotter.parallel_projection = True
plotter.show()