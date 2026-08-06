from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import netgen.gui 
import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from old_files.HybridWB_FEM.wbm_top import WBM_Top
from old_files.HybridWB_FEM.wbm_bottom import WBM_Bottom
from old_files.HybridWB_FEM.hybrid_system import HybridSystem

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
L = 4.0

Lx = L
Ly = L
Lz = L
Lz_1 = 0.2 * L
Lz_plate = 0.02 * L

r_hole = 0.5* Lx/2  

# Meshing parameters
maxh = 0.6
minh = 0.4
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# 1. Air Volumes
duct = Box((0, 0, 0), (Lx, Ly, Lz))
duct.faces.Max(X).Identify(duct.faces.Min(X), "periodic_x")
duct.faces.Max(Y).Identify(duct.faces.Min(Y), "periodic_y")

# 2. Plate Volumes
plate_steel = Box((0, 0, Lz_1), (Lx, Ly, Lz_1 + Lz_plate))
hole_steel = Cylinder(Pnt(Lx/2, Ly/2, Lz_1), gp_Vec(0,0,1), r_hole, Lz_plate)

plate_domains = plate_steel - hole_steel
plate_domains.mat("solid")
plate_domains.faces.Max(X).Identify(plate_domains.faces.Min(X), "periodic_x")
plate_domains.faces.Max(Y).Identify(plate_domains.faces.Min(Y), "periodic_y")

# 3. Air Domains (subtract plate volumes from duct)
air_domains = duct - plate_domains
air_domains.mat("fluid")

# Detect interfaces and rename them for coupling
for f in air_domains.faces:
    f.name = "fluid_boundary"

combined_geo = Glue([air_domains, plate_domains])

for f in plate_domains.faces:
    if f.name == "fluid_boundary":
        f.name = "fsi_interface"

combined_geo.faces.Max(Z).name = "top"
combined_geo.faces.Min(Z).name = "bottom"

geo = OCCGeometry(combined_geo)
mesh = Mesh(geo.GenerateMesh(mp=mp))

# interface_marker = mesh.BoundaryCF({"fsi_interface": 1}, default=0)
# Draw(interface_marker, mesh, "FSI_Interface")
# input("Mesh generated successfully! Press Enter to continue...")
print("Mesh generated successfully!")

# =====================================================================
# 2. Define physics and finite element space
# =====================================================================
print("\n 2. Define physics and finite element space...\n")
# Air Parameters
freq = 343.0/2
c_0 = 343.0* (1 - 1j * 0.001)  
k = 2 * math.pi * freq / c_0  
rho_air = 1.21
omega = 2 * math.pi * freq

# Solid (Steel Plate) Parameters
E = 210e9              # Young's Modulus (Pa)
nu = 0.3               # Poisson's ratio
rho_s = 7800.0         # Density (kg/m^3)
mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))

# Incident angles (e.g., 45 degrees)
theta = (math.pi / 4)  # Polar angle (0 for normal incidence)
phi = 0.0
kx = k * math.sin(theta) * math.cos(phi)
ky = k * math.sin(theta) * math.sin(phi)
kz = k * math.cos(theta)
k_vec = CF((kx, ky, 0)) 

# Finite Element Space
fes_air = Periodic(H1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("fluid")))
fes_plate = Periodic(VectorH1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("solid"), dirichlet="plate1_left|plate2_right"))  # Clamped BCs on plate edges

fes = (fes_air * fes_plate)
(p, u), (q, w) = fes.TnT()  # p,q = acoustic ; u,w = elastic

print(f"Total FSI Degrees of Freedom: {fes.ndof}")

# =====================================================================
# 3. Define variational forms
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# Differential Operators 
def grad_p(p): return grad(p) + 1j * k_vec * p
def grad_q(q): return grad(q) - 1j * k_vec * q
def eps_u(u): return Sym(Grad(u) + 1j * OuterProduct(u, k_vec))
def eps_w(w): return Sym(Grad(w) - 1j * OuterProduct(w, k_vec))
def stress(u): return 2*mu*eps_u(u) + lam*Trace(eps_u(u))*Id(3)

# Uncoupled Bilinear Form (FEM)
Z_fem = BilinearForm(fes)
Z_fem += (grad_p(p) * grad_q(q) - k**2 * p * q) * dx("fluid")
Z_fem += (InnerProduct(stress(u), eps_w(w)) - rho_s * omega**2 * InnerProduct(u, w)) * dx("solid")

# FSI Interface Coupling 
n_plate_to_air = specialcf.normal(mesh.dim)
Z_fem += rho_air * omega**2 * (n_plate_to_air * u)* q * ds("fsi_interface")    
Z_fem += p * (n_plate_to_air *w) * ds("fsi_interface")                      

# Linear Form (RHS) - Surface source on the bottom boundary
s_fem = LinearForm(fes)
source_func = exp(-((z-3*L/4)**2)/(L/20)**2)
s_fem += source_func* q * dx("fluid")

# Assemble 
with TaskManager():
    Z_fem.Assemble()
    s_fem.Assemble()

print("Assembly complete!")

# =====================================================================
# 4. Build WBM model and coupling matrices
# =====================================================================
print("\n 4. Build WBM model and coupling matrices...\n")
m_max = 4
n_max = 4
wbm_top = WBM_Top(Lx, Ly, L, freq, c_0, rho_air, m_max, n_max, theta, phi)
Z_hyb_top, Z_wbm_top = wbm_top.assemble_matrices(mesh, fes, q, "top")
wbm_bottom = WBM_Bottom(Lx, Ly, 0, freq, c_0, rho_air, m_max, n_max, theta, phi)
Z_hyb_bottom, Z_wbm_bottom = wbm_bottom.assemble_matrices(mesh, fes, q, "bottom")

total_waves = wbm_top.total_waves  + wbm_bottom.total_waves
print(f"Total WBM wave functions: {total_waves}")
Z_hyb = np.hstack([Z_hyb_top, Z_hyb_bottom])
Z_wbm = la.block_diag(Z_wbm_top, Z_wbm_bottom)
print("condition number of Z_wbm:", np.linalg.cond(Z_wbm))

# =====================================================================
# 5. Build and solve the coupled system
# =====================================================================
print("\n 5. Build and solve the coupled system...\n")
HBSys = HybridSystem(fes, total_waves)
Global_Matrix, Global_RHS = HBSys.combine_matrices(Z_fem, s_fem, Z_hyb, Z_wbm)
fem_gf, wbm_factors = HBSys.solve_coupled_system(Global_Matrix, Global_RHS)
print("Coupled system solved successfully!")

# 1. Reconstruct Field
phase = exp(1j * (kx * x + ky * y))
gfu_p, gfu_u = fem_gf.components
gfu_p = gfu_p * phase
gfu_u = gfu_u * phase

Draw(Norm(gfu_u), mesh, name="Disp_Norm", deformation=gfu_u)
Draw(gfu_p, mesh, name="Pressure")
input("FSI simulation completed! Press Enter to exit...")

