from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import netgen.gui  
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
ts = 0.002         # steel thickness
tr = 0.003         # rubber layer thickness (downstream of steel)
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
Draw(interface_marker, mesh, "FSI_Interface")
input("Mesh generated successfully! Press Enter to continue...")

