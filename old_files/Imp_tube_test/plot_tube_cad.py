from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import netgen.gui  
import ngsolve as ng
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

import pyvista as pv
import numpy as np
from pyvista.utilities.geometric_objects import translate

# 1. Geometry Setup
r_rbr = 51e-3 # rubber
h_rbr = 1e-3
rho_rbr = 1590.0
E_rbr = 1.5e6
nu_rbr = 0.49
mu_rbr = E_rbr / (2 * (1 + nu_rbr))
lam_rbr = E_rbr * nu_rbr / ((1 + nu_rbr) * (1 - 2 * nu_rbr))

r_stl = 61.25e-3 # steel
h_stl = 1.5e-3
rho_stl = 7800.0
E_stl = 210e9
nu_stl = 0.3
mu_stl = E_stl / (2 * (1 + nu_stl))
lam_stl = E_stl * nu_stl / ((1 + nu_stl) * (1 - 2 * nu_stl))

r_res = 5e-3 # resonator
h_res = 3e-3
rho_res = 7855.6
E_res = 210e9
nu_res = 0.3
mu_res = E_res / (2 * (1 + nu_res))
lam_res = E_res * nu_res / ((1 + nu_res) * (1 - 2 * nu_res))

r_tube = 61.25e-3 # tube
h_tube = 200e-3
rho_air = 1.21
c_air = 343.0* (1 + 0.001j) # speed of sound in air

h_plate = 140e-3 # plate position

freq = 200.0
omega = 2 * math.pi * freq
k = 2 * math.pi * freq / c_air

# Meshing parameters
maxh = 4e-3
minh = 1e-3
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 2

# 2. Plate Volumes
stl_1 = Cylinder(Pnt(r_tube, r_tube, h_plate), \
                 gp_Vec(0,0,1), r_tube, h_stl, mantle="stl1_outer") \
      - Cylinder(Pnt(r_tube, r_tube, h_plate), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_1.mat("steel1")

rbbr = Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl), \
               gp_Vec(0,0,1), r_tube, h_rbr, mantle="rbbr_outer")
rbbr.mat("rubber")

stl_2 = Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
                 gp_Vec(0,0,1), r_tube, h_stl, mantle="stl2_outer") \
      - Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_2.mat("steel2")

reso = Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
               gp_Vec(0,0,1), r_res, h_res)
reso.mat("resonator")
solid_domains = Glue([stl_1, rbbr, stl_2, reso])

# Separate the mesh
for s in solid_domains.solids:
    if s.name == "steel1" or s.name == "steel2":
        s.maxh = 5e-3
    elif s.name == "rubber":
        s.maxh = 5e-3
    elif s.name == "resonator":
        s.maxh = 5e-3

geo = OCCGeometry(solid_domains, dim=3)
ngmesh = geo.GenerateMesh()
mesh = Mesh(ngmesh)

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
# plotter.add_mesh(pv_mesh, color='grey', show_edges=True)


# 3. Create the box with your updated bounds tuple
cylinder = pv.Cylinder(
    center=(r_tube, r_tube, h_tube/2),  # Centroid location in [x, y, z]
    direction=(0.0, 0.0, 1.0), # Orientation vector
    radius=r_tube,
    height=h_tube,
    resolution=100,       # Optional: smoothness of the circular face
    capping=True          # Optional: whether to close the ends (default True)
)

# Add semi-transparent bounding box
plotter.add_mesh(cylinder, color='lightblue', opacity=0.2, style='surface')

center_x = r_tube
center_y = r_tube
center_z = h_tube*0.4

# Example 1: An XY-plane cutting horizontally through the center of the duct
# i_hat=(1,0,0) and j_hat=(0,1,0) define the orientation span of the plane

circle = pv.Circle(
    radius=r_tube,
)
translate(circle, center=(center_x, center_y, center_z), direction=(1, 0, 0))

# Add the plane to the plotter with custom opacity and color
plotter.add_mesh(circle, color='pink', opacity=0.4)

# plotter.parallel_projection = True
plotter.show()