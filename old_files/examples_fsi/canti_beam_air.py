from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import netgen.gui  
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from old_files.HybridWB_FEM.wbm_top import WBM_Top
from old_files.HybridWB_FEM.wbm_bottom import WBM_Bottom

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
L = 4.0

Lx = L
Ly = L/64
Lz = L
Lz_1 = 0.98 * L/2
Lz_2 = 0.98 * L/2      

Lx_plate = 0.9* Lx/2  
Lz_plate = L - Lz_1 - Lz_2

# Meshing parameters
maxh = 0.1
minh = 0.05
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 2

# 1. Air Volumes
duct = Box((0, 0, 0), (Lx, Ly, Lz))

# 2. Plate Volumes
plate_left = Box((0, 0, Lz_1), (Lx_plate, Ly, Lz_1 + Lz_plate))
plate_right = Box((Lx - Lx_plate, 0, Lz_1), (Lx, Ly, Lz_1 + Lz_plate))

# Glue them together to ensure conforming mesh nodes at the interfaces
plate_domains = plate_left + plate_right
plate_domains.mat("solid")
air_domains = duct - plate_domains
air_domains.mat("fluid")

# Get fixed boundary faces for plate (clamped BCs)
plate_domains.faces.Min(X).name = "plate1_left"
plate_domains.faces.Max(X).name = "plate2_right"

# Detect interfaces and rename them for coupling
for f in air_domains.faces:
    f.name = "fluid_boundary"

combined_geo = Glue([air_domains, plate_domains])

for f in plate_domains.faces:
    if f.name == "fluid_boundary":
        f.name = "fsi_interface"

geo = OCCGeometry(combined_geo)
mesh = Mesh(geo.GenerateMesh(mp=mp))

#interface_marker = mesh.BoundaryCF({"fsi_interface": 1}, default=0)
#Draw(interface_marker, mesh, "FSI_Interface")
#input("Mesh generated successfully! Press Enter to continue...")
print("Mesh generated successfully!")

# =====================================================================
# 2. Define physics and finite element space
# =====================================================================
print("\n 2. Define physics and finite element space...\n")
# Air Parameters
freq = 343.0     
c_0 = 343.0* (1 - 1j * 0.001)  
k = 2 * math.pi * freq / c_0  
rho_air = 1.21
omega = 2 * math.pi * freq

# Solid (Steel Plate) Parameters
E = 210e9              # Young's Modulus (Pa)
nu = 0.3               # Poisson's ratio
rho_s = 7800.0/1e5         # Density (kg/m^3)
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

# Finite Element Space
fes_air = H1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("fluid"))
fes_plate = VectorH1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("solid"), dirichlet="plate1_left|plate2_right")  # Clamped BCs on plate edges

fes = fes_air * fes_plate
(p, u), (q, w) = fes.TnT()  # p,q = acoustic ; u,w = elastic

print(f"Total FSI Degrees of Freedom: {fes.ndof}")


# =====================================================================
# 3. Define variational forms
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# Differential Operators 
def strain(u): return Sym(Grad(u))
def stress(u): return 2*mu*strain(u) + lam*Trace(strain(u))*Id(3)

# 2. Bilinear Form (LHS)
Z_fem = BilinearForm(fes)
Z_fem += (grad(p) * grad(q) - k**2 * p * q) * dx("fluid")
Z_fem += (InnerProduct(stress(u), strain(w)) - rho_s * omega**2 * InnerProduct(u, w)) * dx("solid")

# FSI Interface Coupling 
dom_marker = mesh.MaterialCF({"solid": 1, "fluid": 0}, default=0)
n_plate_to_air = specialcf.normal(mesh.dim)

Z_fem += rho_air * omega**2 * (n_plate_to_air * u)* q * ds("fsi_interface")    
Z_fem += p * (n_plate_to_air *w) * ds("fsi_interface")                      

# 3. Linear Form (RHS) - Surface source on the bottom boundary
s_fem = LinearForm(fes)
source_func = exp(-((z-1*L/3)**2)/(L/20)**2)
s_fem += source_func* q * dx("fluid")


# 4. Assemble 
gfu = GridFunction(fes)
with TaskManager():
    Z_fem.Assemble()
    s_fem.Assemble()
    gfu.vec.data = Z_fem.mat.Inverse(fes.FreeDofs()) * s_fem.vec

gfu_p, gfu_u = gfu.components

Draw(Norm(gfu_u), mesh, "Disp_Norm", deformation=gfu_u)
Draw(gfu_p, mesh, "Pressure")
input("FSI simulation completed! Press Enter to exit...")


