from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import scipy.sparse.linalg as spla

import netgen.gui 
import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
from HybridWB_FEM_new.wbm_top_c import WBM_Top_C
from HybridWB_FEM_new.wbm_bottom_c import WBM_Bottom_C
from HybridWB_FEM_new.hybrid_system import HybridSystem

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
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
h_tube = 12e-3
rho_air = 1.21
c_air = 343.0 # speed of sound in air

h_plate = 3e-3 # plate position

freq = 1000.0
omega = 2 * math.pi * freq
k = 2 * math.pi * freq / c_air

# Meshing parameters
curve_order = 3

# 1. Air Volumes
duct2 = Cylinder(Pnt(0, 0, 0), \
                gp_Vec(0,0,1), r_tube, h_tube)

# 2. Plate Volumes
stl_1 = Cylinder(Pnt(0, 0, h_plate), \
                 gp_Vec(0,0,1), r_tube, h_stl, mantle="stl1_outer") \
      - Cylinder(Pnt(0, 0, h_plate), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_1.mat("steel1")

rbbr = Cylinder(Pnt(0, 0, h_plate+h_stl), \
               gp_Vec(0,0,1), r_tube, h_rbr, mantle="rbbr_outer")
rbbr.mat("rubber")

stl_2 = Cylinder(Pnt(0, 0, h_plate+h_stl+h_rbr), \
                 gp_Vec(0,0,1), r_tube, h_stl, mantle="stl2_outer") \
      - Cylinder(Pnt(0, 0, h_plate+h_stl+h_rbr), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_2.mat("steel2")

reso = Cylinder(Pnt(0, 0, h_plate+h_stl+h_rbr), \
               gp_Vec(0,0,1), r_res, h_res)
reso.mat("resonator")
solid_domains = Glue([stl_1, rbbr, stl_2, reso])

# 3. Air Domains (subtract plate volumes from duct)
air_cut = duct2 - solid_domains
air_cut.mat("fluid")

# Detect interfaces and rename them for coupling
for f in air_cut.faces:
    f.name = "fluid_boundary"

combined_geo = Glue([air_cut, solid_domains])

for f in solid_domains.faces:
    if f.name == "fluid_boundary":
        f.name = "fsi_interface"

combined_geo.faces.Max(Z).name = "top"
combined_geo.faces.Min(Z).name = "bottom"

# Separate the mesh
"""
for s in combined_geo.solids:
    print(f"Solid: {s.name}")
    if s.name == "fluid":
        s.maxh = 64e-3
    elif s.name == "rubber":
        s.maxh = 16e-3
    elif s.name == "resonator":
        s.maxh = 8e-3
"""

geo = OCCGeometry(combined_geo, dim=3)
# ngmesh = geo.GenerateMesh(mp=mp)
ngmesh = geo.GenerateMesh()
mesh = Mesh(ngmesh)

# interface_marker = mesh.BoundaryCF({"fsi_interface": 1}, default=0)
# Draw(interface_marker, mesh, "fsi_interface")
# Draw(mesh)
# input("Mesh generated successfully! Press Enter to continue...")
# print("Mesh generated successfully!")


# =====================================================================
# 2. Define physics and finite element space
# =====================================================================
print("\n 2. Define physics and finite element space...\n")
# Finite Element Space
fes_air = H1(mesh, order=curve_order, complex=True, \
             definedon=mesh.Materials("fluid"))
fes_plate = VectorH1(mesh, order=curve_order, complex=True, \
                     definedon=mesh.Materials("steel1|rubber|steel2|resonator"), \
                     dirichlet="stl1_outer|rbbr_outer|stl2_outer")
fes = (fes_air * fes_plate)
(p, u), (q, w) = fes.TnT()  # p,q = acoustic ; u,w = elastic
print(f"Total FSI Degrees of Freedom: {fes.ndof}")

# =====================================================================
# 3. Define variational forms
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# Differential Operators 
sig_x0 = Parameter(5e4) 
sig_y0 = Parameter(5e4)  
sigma_0 = CoefficientFunction((sig_x0, 0, 0,
                               0, sig_y0, 0,
                               0, 0,      0), dims=(3,3))
def strain(u): return Sym(Grad(u))
def stress_stl(u): return 2*mu_stl*strain(u) + lam_stl*Trace(strain(u))*Id(3)
def stress_rbr(u): return 2*mu_rbr*strain(u) + lam_rbr*Trace(strain(u))*Id(3)
def stress_res(u): return 2*mu_res*strain(u) + lam_res*Trace(strain(u))*Id(3)
# Uncoupled Bilinear Form (FEM)
Z_fem = BilinearForm(fes)
Z_fem += (grad(p) * grad(q) - k**2 * p * q) * dx("fluid")
Z_fem += (InnerProduct(stress_stl(u), strain(w)) - rho_stl * omega**2 * InnerProduct(u, w)) * dx("steel1|steel2")
Z_fem += (InnerProduct(stress_rbr(u), strain(w)) - rho_rbr * omega**2 * InnerProduct(u, w)) * dx("rubber")
Z_fem += InnerProduct(Grad(u) * sigma_0, Grad(w)) * dx("rubber")
Z_fem += (InnerProduct(stress_res(u), strain(w)) - rho_res * omega**2 * InnerProduct(u, w)) * dx("resonator")
# FSI Interface Coupling 
n_plate_to_air = specialcf.normal(mesh.dim)
Z_fem += rho_air * omega**2 * (n_plate_to_air * u)* q * ds("fsi_interface")    
Z_fem += p * (n_plate_to_air *w) * ds("fsi_interface")         
# Assemble 
with TaskManager():
    Z_fem.Assemble()

# =====================================================================
# 4. Build WBM model and coupling matrices
# =====================================================================
print("\n 4. Build WBM model and coupling matrices...\n")
m_max = 2
n_max = 0
wbm_top = WBM_Top_C(r_tube, h_tube, freq, c_air, rho_air, m_max, n_max)
Z_hyb_top, Z_wbm_top, s_hyb_top, s_wbm_top = wbm_top.assemble_matrices(mesh, fes, q, "top")
wbm_bottom = WBM_Bottom_C(r_tube, h_tube, freq, c_air, rho_air, m_max, n_max)
Z_hyb_bottom, Z_wbm_bottom, s_hyb_bottom, s_wbm_bottom = wbm_bottom.assemble_matrices(mesh, fes, q, "bottom")

total_waves = wbm_top.total_waves  + wbm_bottom.total_waves
print(f"Total WBM wave functions: {total_waves}")
Z_hyb = np.hstack([Z_hyb_top, Z_hyb_bottom])
Z_wbm = la.block_diag(Z_wbm_top, Z_wbm_bottom)
s_hyb = np.concatenate([s_hyb_top, s_hyb_bottom])
s_wbm = np.concatenate([s_wbm_top, s_wbm_bottom])
print("condition number of Z_wbm:", np.linalg.cond(Z_wbm))

# =====================================================================
# 5. Build and solve the coupled system
# =====================================================================
print("\n 5. Build and solve the coupled system...\n")
HBSys = HybridSystem(fes, total_waves)
Global_Matrix, Global_RHS = HBSys.combine_matrices(Z_fem, Z_hyb, Z_wbm, s_hyb, s_wbm)
gfu, wbm_factors = HBSys.solve_coupled_system(Global_Matrix, Global_RHS)
print(abs(wbm_factors))
print("Coupled system solved successfully!")

gfu_p, gfu_u = gfu.components
Draw(Norm(gfu_u), mesh, name="Disp_Norm", deformation=gfu_u)
Draw(gfu_p, mesh, name="Pressure")
input("FSI simulation completed! Press Enter to exit...")
