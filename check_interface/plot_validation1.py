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

import pyvista as pv
import numpy as np

# 1. Geometry Setup
L = 4.0
Lx = L/2
Ly = L/8
Lz = L
Lz_1 = 0.8 * L
Lz_plate = 0.01 * L

# Create plotter
plotter = pv.Plotter()

# 3. Create the box with your updated bounds tuple
modified_bounds = (0, Lx, 0, Ly, 0, Lz)
box = pv.Box(modified_bounds)

# Add semi-transparent bounding box
plotter.add_mesh(box, color='lightblue', opacity=0.2, style='surface')
plotter.add_mesh(box, color='lightblue', style='wireframe', line_width=2)


center_x = Lx / 2
center_y = Ly / 2
center_z = Lz*0.4

# Example 1: An XY-plane cutting horizontally through the center of the duct
# i_hat=(1,0,0) and j_hat=(0,1,0) define the orientation span of the plane
plane = pv.Plane(
    center=(center_x, center_y, center_z),
    direction=(0, 0, 1),              # Normal vector pointing straight up (+Z)
    i_size=Lx,                # Width along Y
    j_size=Ly                # Height along Z
)
# Add the plane to the plotter with custom opacity and color
plotter.add_mesh(plane, color='pink', opacity=0.4, show_edges=False)

# Show the combined view
# plotter.parallel_projection = True
plotter.show()