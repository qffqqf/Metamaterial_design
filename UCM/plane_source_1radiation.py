from netgen.occ import *
from ngsolve import *
import netgen.gui  
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from HybridWB_FEM.wbm_top import WBM_Top
from HybridWB_FEM.wbm_bottom import WBM_Bottom

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
L = 2.0
Lx = L/4
Ly = L/32

# Air
geometry = Box((0, 0, 0), (Lx, Ly, L))
geometry.mat("air")     

# Tag the boundaries for Boundary Conditions
geometry.faces.Max(Z).name = "top"
geometry.faces.Min(Z).name = "bottom"

# 3. Identify Periodic Faces
geometry.faces.Max(X).Identify(geometry.faces.Min(X), "periodic_x")
geometry.faces.Max(Y).Identify(geometry.faces.Min(Y), "periodic_y")

# 4. Generate Mesh
geo = OCCGeometry(geometry)
mesh = Mesh(geo.GenerateMesh(maxh=0.2))

print("Mesh generated successfully!")

# =====================================================================
# 2. Define physics and finite element space
# =====================================================================
print("\n 2. Define physics and finite element space...\n")
# 1. Physics Parameters
freq = 343.0     
c_0 = 343.0* (1 - 1j * 0.001)  # Complex speed of sound to introduce a small amount of damping
k = 2 * math.pi * freq / c_0  
rho_air = 1.21

# Incident angles (e.g., 45 degrees)
theta = (math.pi / 4)  # Polar angle (0 for normal incidence)
phi = 0.0

# Wave vector components
kx = k * math.sin(theta) * math.cos(phi)
ky = k * math.sin(theta) * math.sin(phi)
kz = k * math.cos(theta)

# Transverse wave vector for the Floquet shift
k_vec = CF((kx, ky, 0)) 

# Finite Element Space
fes = Periodic(H1(mesh, order=3, complex=True))

p,q = fes.TnT()

print(f"Degrees of freedom: {fes.ndof}")
print(f"wave vector k: ({kx:.2f}, {ky:.2f}, {kz:.2f})")

# =====================================================================
# 3. Define variational forms and assemble FE model
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# 1. Modified Gradients (Floquet Trick)
grad_p = grad(p) + 1j * k_vec * p
grad_q = grad(q) - 1j * k_vec * q 

# 2. Bilinear Form (LHS)
Z_fem = BilinearForm(fes)
Z_fem += (grad_p * grad_q - k**2 * p * q) * dx

# 3. Linear Form (RHS) - Surface source on the bottom boundary
s_fem = LinearForm(fes)
# Gaussian source
source_func = exp(-(z-L/3)**2/(L/20)**2)
s_fem += source_func* q * dx

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
n_max = 2
# wbm_top = WBM_Top(Lx, Ly, L, freq, c_0, rho_air, m_max, n_max, theta, phi)
# Z_hyb, Z_wbm = wbm_top.assemble_matrices(mesh, fes, q, "top")

wbm_bottom = WBM_Bottom(Lx, Ly, 0, freq, c_0, rho_air, m_max, n_max, theta, phi)
Z_hyb, Z_wbm = wbm_bottom.assemble_matrices(mesh, fes, q, "bottom")
print("shape of Z_wbm:", Z_wbm.shape)
print("shape of Z_hyb:", Z_hyb.shape)
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
s_wbm = np.zeros(wbm_bottom.total_waves, dtype=complex)
Global_RHS = np.concatenate([s_fem_free, s_wbm])

# Solve the dense/sparse hybrid system using SuperLU
solution = spla.spsolve(Global_Matrix, Global_RHS)

p_fem_vals = solution[:len(free_indices)]
p_wbm_factors = solution[len(free_indices):]

# Map back to an NGSolve GridFunction
p_fem_gf = GridFunction(fes)
p_fem_gf.vec.FV().NumPy()[free_indices] = p_fem_vals

print("Coupled system solved successfully!")

# =====================================================================
# 6. visualize results
# =====================================================================
print("\n 6. Visualize results...\n")

# 1. Reconstruct Field
phase = exp(1j * (kx * x + ky * y))
p_true = p_fem_gf * phase
# 2. Visualize
print("Rendering total pressure field ...")
Draw(p_true, mesh, "Acoustic Pressure", animate_complex=True)
input("Press Enter to exit...")