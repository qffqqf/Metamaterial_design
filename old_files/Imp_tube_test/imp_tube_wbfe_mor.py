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
sig_x0 = Parameter(100.0) 
sig_y0 = Parameter(100.0)  
sigma_0 = CoefficientFunction((sig_x0, 0, 0,
                            0, sig_y0, 0,
                            0, 0,      0), dims=(3,3))
def strain(u): return Sym(Grad(u))
def stress_stl(u): return 2*mu_stl*strain(u) + lam_stl*Trace(strain(u))*Id(3)
def stress_rbr(u): return 2*mu_rbr*strain(u) + lam_rbr*Trace(strain(u))*Id(3)
def stress_res(u): return 2*mu_res*strain(u) + lam_res*Trace(strain(u))*Id(3)
# Uncoupled Bilinear Form (FEM)
K_fem = BilinearForm(fes)
M_fem = BilinearForm(fes)

K_fem += (grad(p) * grad(q)) * dx("fluid")
K_fem += (InnerProduct(stress_stl(u), strain(w))) * dx("steel1|steel2")
K_fem += (InnerProduct(stress_rbr(u), strain(w))) * dx("rubber")
K_fem += InnerProduct(Grad(u) * sigma_0, Grad(w)) * dx("rubber")
K_fem += (InnerProduct(stress_res(u), strain(w))) * dx("resonator")

M_fem += ((1/c_air)**2 * p * q) * dx("fluid")
M_fem += (rho_stl * InnerProduct(u, w)) * dx("steel1|steel2")
M_fem += (rho_rbr * InnerProduct(u, w)) * dx("rubber")
M_fem += (rho_res * InnerProduct(u, w)) * dx("resonator")

# FSI Interface Coupling 
n_plate_to_air = specialcf.normal(mesh.dim)
K_fem += p * (n_plate_to_air *w) * ds("fsi_interface")     
M_fem += - rho_air * (n_plate_to_air * u)* q * ds("fsi_interface")    

# Assemble 
with TaskManager():
    K_fem.Assemble()
    M_fem.Assemble()

# ---------------------------------------------------------------------
# GET FREE INDICES FOR "top|bottom" AND THE REST
# ---------------------------------------------------------------------
# BitArrays of length ndof: True if DOF satisfies the condition
free_dofs_bitarray = fes.FreeDofs()
top_bot_dofs_bitarray = fes.GetDofs(mesh.Boundaries("top|bottom"))

# DOFs on top|bottom that are free
top_bottom_indices = [
    i for i, (is_free, is_tb) in enumerate(zip(free_dofs_bitarray, top_bot_dofs_bitarray)) 
    if is_free and is_tb
]

# The interior free DOFs
int_indices = [
    i for i, (is_free, is_tb) in enumerate(zip(free_dofs_bitarray, top_bot_dofs_bitarray)) 
    if is_free and not is_tb
]

print(f"Total free DOFs: {sum(free_dofs_bitarray)}")
print(f"Free DOFs on top/bottom: {len(top_bottom_indices)}")
print(f"Interior free DOFs: {len(int_indices)}")

# =====================================================================
# 4. Craig-Bampton Reduction
# =====================================================================
print("\n 4. Applying Craig-Bampton Method...\n")
import scipy.sparse as sp

# --- Step 4.1: Convert NGSolve matrices to SciPy CSR matrices ---
rows, cols, vals = K_fem.mat.COO()
K_sp = sp.csr_matrix((vals, (rows, cols)), shape=(fes.ndof, fes.ndof))

rows, cols, vals = M_fem.mat.COO()
M_sp = sp.csr_matrix((vals, (rows, cols)), shape=(fes.ndof, fes.ndof))

# --- Step 4.2: Extract submatrices ---
print("Extracting partitioned matrices...")
K_ii = K_sp[np.ix_(int_indices, int_indices)].tocsc()
K_ib = K_sp[np.ix_(int_indices, top_bottom_indices)].tocsc()
M_ii = M_sp[np.ix_(int_indices, int_indices)].tocsc()

# --- Step 4.3: Compute interior eigenmodes (Phi) ---
n_modes = 7  # Adjust the number of interior modes you want to retain
print(f"Computing the first {n_modes} interior eigenmodes (Phi)...")
# We use shift-invert (sigma) to efficiently target the lowest frequency modes
eigenvalues, Phi = spla.eigs(K_ii, k=n_modes, M=M_ii, sigma=1e-5)

# --- Step 4.4: Compute static constraint modes (Psi) ---
print("Computing static constraint modes (Psi)...")
# Solve K_ii * Psi = -K_ib
Psi = spla.spsolve(K_ii, K_ib)
if sp.issparse(Psi):
    Psi = Psi.toarray()

# --- Step 4.5: Build the full projection matrix ---
print("Building full projection matrix [Phi, Psi; 0, I]...")
# The reduced dimension is the number of retained modes + the interface DOFs
red_dim = n_modes + len(top_bottom_indices)

# Using LIL matrix for efficient index-based assignments before converting to CSR
T_full = sp.lil_matrix((fes.ndof, red_dim), dtype=complex)

# 1. Map Phi (Interior modes to interior DOFs)
# Columns 0 to n_modes-1
T_full[np.ix_(int_indices, np.arange(n_modes))] = Phi

# 2. Map Psi (Constraint modes to interior DOFs)
# Columns n_modes to end
T_full[np.ix_(int_indices, np.arange(n_modes, red_dim))] = Psi

# 3. Map I (Identity matrix to interface DOFs)
# Columns n_modes to end
I_interface = np.eye(len(top_bottom_indices))
T_full[np.ix_(top_bottom_indices, np.arange(n_modes, red_dim))] = I_interface

# Convert to CSR matrix for efficient matrix multiplications moving forward
T_full = T_full.tocsr()

# Compute the reduced system matrices: K_cb = T^H * K * T, M_cb = T^H * M * T
K_cb = T_full.conj().transpose() @ K_sp @ T_full
M_cb = T_full.conj().transpose() @ M_sp @ T_full

print(f"Original full DOFs: {fes.ndof}")
print(f"Reduced Craig-Bampton DOFs: {red_dim}")
print("Craig-Bampton reduction completed successfully!")