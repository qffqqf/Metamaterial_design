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
r_tube = 61.25e-3 # tube
h_tube = 20e-3
rho_air = 1.21
c_air = 343.0 # speed of sound in air

freq = 1000.0
omega = 2 * math.pi * freq
k = 2 * math.pi * freq / c_air

# Meshing parameters
maxh = 16e-3
minh = 8e-3
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# 1. Air Volumes
duct2 = Cylinder(Pnt(0, 0, 0), \
                gp_Vec(0,0,1), r_tube, h_tube)
duct2.mat("fluid")

duct2.faces.Max(Z).name = "top"
duct2.faces.Min(Z).name = "bottom"

geo = OCCGeometry(duct2, dim=3)
ngmesh = geo.GenerateMesh(mp=mp)
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
fes = H1(mesh, order=curve_order, complex=True, \
             definedon=mesh.Materials("fluid"))
p, q = fes.TnT()  # p,q = acoustic ; u,w = elastic
print(f"Total FSI Degrees of Freedom: {fes.ndof}")

# =====================================================================
# 3. Define variational forms
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# Uncoupled Bilinear Form (FEM)
Z_fem = BilinearForm(fes)
Z_fem += (grad(p) * grad(q) - k**2 * p * q) * dx("fluid")
with TaskManager():
    Z_fem.Assemble()
# =====================================================================
# 4. Build WBM model and coupling matrices
# =====================================================================
print("\n 4. Build WBM model and coupling matrices...\n")
m_max = 0
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
gfu_p, wbm_factors = HBSys.solve_coupled_system(Global_Matrix, Global_RHS)
print(abs(wbm_factors))
print("Coupled system solved successfully!")

"""
Draw(gfu_p, mesh, name="Pressure")
input("FSI simulation completed! Press Enter to exit...")
"""

# 1. Reconstruct Field
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 200)
xs = 0* t 
ys = 0* t 
zs = t* h_tube
vals = gfu_p(mesh(xs, ys, zs)).reshape(-1)
rel_error = np.abs(abs(vals) - 1) 

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Top Plot: Comparison
ax1.plot(zs, abs(vals), '-', label="FEM-WBM", alpha=0.7)
ax1.plot(zs, 1*np.ones_like(zs), '-.', label="Analytical solution", color='green')
ax1.set_ylim(0, 1.5)
ax1.set_ylabel("$|p(z)|$ [Pa]", fontsize=12)
ax1.legend()
ax1.grid(True)

# Bottom Plot: Relative Error
ax2.semilogy(zs, rel_error, 'k-', label="Relative Error", alpha=0.7)
ax2.set_xlabel("z [m]", fontsize=12)
ax2.set_ylabel("Relative Error [-]", fontsize=12)
ax2.grid(True, which="both", ls="-", alpha=0.5)
ax2.legend()

plt.show()