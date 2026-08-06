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

Lx_plate = 0.998* Lx/2  
Lz_plate = L - Lz_1 - Lz_2

# Meshing parameters
maxh = 0.16
minh = 0.08
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 2

# 1. Air Volumes
duct = Box((0, 0, 0), (Lx, Ly, Lz))

# 2. Plate Volumes

# Glue them together to ensure conforming mesh nodes at the interfaces
air_domains = duct 
air_domains.mat("fluid")

air_domains.faces.Max(X).Identify(air_domains.faces.Min(X), "periodic_x")
air_domains.faces.Max(Y).Identify(air_domains.faces.Min(Y), "periodic_y")


# Detect interfaces and rename them for coupling

air_domains.faces.Max(Z).name = "top"
air_domains.faces.Min(Z).name = "bottom"

geo = OCCGeometry(air_domains)
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


# Incident angles (e.g., 45 degrees)
theta = (math.pi / 4)  # Polar angle (0 for normal incidence)
phi = 0.0
kx = k * math.sin(theta) * math.cos(phi)
ky = k * math.sin(theta) * math.sin(phi)
kz = k * math.cos(theta)
k_vec = CF((kx, ky, 0)) 

# Finite Element Space
fes_air = Periodic(H1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("fluid")))

fes = (fes_air)
p,q = fes.TnT()  # p,q = acoustic ; u,w = elastic

print(f"Total FSI Degrees of Freedom: {fes.ndof}")

# =====================================================================
# 3. Define variational forms
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# Differential Operators 
def grad_p(p): return grad(p) + 1j * k_vec * p
def grad_q(q): return grad(q) - 1j * k_vec * q

# 2. Bilinear Form (LHS)
Z_fem = BilinearForm(fes)
Z_fem += (grad_p(p) * grad_q(q) - k**2 * p * q) * dx("fluid")

# 3. Linear Form (RHS) - Surface source on the bottom boundary
s_fem = LinearForm(fes)
source_func = exp(-((z-3*L/4)**2)/(L/20)**2)
s_fem += source_func* q * dx("fluid")

# 4. Assemble 
with TaskManager():
    Z_fem.Assemble()
    s_fem.Assemble()

# 5. Extract free DOFs
freedofs = fes.FreeDofs()
free_indices = [i for i, is_free in enumerate(freedofs) if is_free]
slave_indices = [i for i, is_free in enumerate(freedofs) if not is_free]
print("Assembly complete!")

# =====================================================================
# 4. Build WBM model and coupling matrices
# =====================================================================
print("\n 4. Build WBM model and coupling matrices...\n")
m_max = 2
n_max = 0
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

# Convert sparse Z_fem to SciPy CSC format
row, col, val = Z_fem.mat.COO()
Z_fem_scipy = sp.coo_matrix((val, (row, col)), shape=(fes.ndof, fes.ndof)).tocsc()
Z_fem_free = Z_fem_scipy[free_indices, :][:, free_indices]
Z_hyb_free = Z_hyb[free_indices, :]

f_f_np = s_fem.vec.FV().NumPy()
s_fem_free = f_f_np[free_indices]

# Block Matrix 
top_row = sp.hstack([Z_fem_free, Z_hyb_free])
bottom_row = np.hstack([Z_hyb_free.conj().T, Z_wbm])
Global_Matrix = sp.vstack([top_row, bottom_row]).tocsc()

# Global RHS Vector 
s_wbm = np.zeros(total_waves, dtype=complex)
Global_RHS = np.concatenate([s_fem_free, s_wbm])

# Solve the dense/sparse hybrid system using SuperLU
solution = spla.spsolve(Global_Matrix, Global_RHS)

fem_vals = solution[:len(free_indices)]
wbm_factors = solution[len(free_indices):]

# Map back to an NGSolve GridFunction
gfu_p = GridFunction(fes)
gfu_p.vec.FV().NumPy()[free_indices] = fem_vals

print("Coupled system solved successfully!")


# 1. Reconstruct Field
phase = exp(1j * (kx * x + ky * y))
gfu_p = gfu_p * phase

Draw(gfu_p, mesh, "Pressure")
input("FSI simulation completed! Press Enter to exit...")

