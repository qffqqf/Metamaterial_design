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

# 1. Geometry Setup
L = 2.0

Lx = L/2
Ly = L/16
Lz = L
Lz_1 = 0.25 * L
Lz_plate = 0.5 * L

L_source = Lz

# Meshing parameters
maxh = 0.2
minh = 0.1
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# 2. Plate Volumes
plate_domains = Box((0, 0, Lz_1), (Lx, Ly, Lz_1 + Lz_plate))
plate_domains.mat("solid")

geo = OCCGeometry(plate_domains)
mesh = Mesh(geo.GenerateMesh(mp=mp))

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
plotter.add_mesh(pv_mesh, color='lightgrey', opacity=0.3, show_edges=True)


# 3. Create the box with your updated bounds tuple
modified_bounds = (0, Lx, 0, Ly, 0, Lz)
box = pv.Box(modified_bounds)

# Add semi-transparent bounding box
plotter.add_mesh(box, color='lightblue', opacity=0.2, style='surface')
plotter.add_mesh(box, color='lightblue', style='wireframe', line_width=2)


center_x = Lx / 2
center_y = Ly / 2
center_z = L_source

# Example 1: An XY-plane cutting horizontally through the center of the duct
# i_hat=(1,0,0) and j_hat=(0,1,0) define the orientation span of the plane
plane = pv.Plane(
    center=(center_x, center_y, center_z),
    direction=(0, 0, 1),              # Normal vector pointing straight up (+Z)
    i_size=Lx,                # Width along Y
    j_size=Ly                # Height along Z
)
# Add the plane to the plotter with custom opacity and color
plotter.add_mesh(plane, color='pink', opacity=0.8, show_edges=False)

# Show the combined view
# plotter.parallel_projection = True
plotter.show()