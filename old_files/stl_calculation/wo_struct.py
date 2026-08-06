from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import netgen.gui 
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
from HybridWB_FEM_new.wbm_top import WBM_Top
from HybridWB_FEM_new.wbm_bottom import WBM_Bottom
from HybridWB_FEM_new.hybrid_system import HybridSystem

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
L = 1.0

Lx = L
Ly = L/8
Lz = L
Lz_1 = 0.1 * L
Lz_plate = 0.02 * L

# Meshing parameters
maxh = 0.2
minh = 0.1
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# 1. Air Volumes
air_domains = Box((0, 0, 0), (Lx, Ly, Lz))
air_domains.faces.Max(X).Identify(air_domains.faces.Min(X), "periodic_x")
air_domains.faces.Max(Y).Identify(air_domains.faces.Min(Y), "periodic_y")

# 3. Air Domains (subtract plate volumes from duct)
air_domains.mat("fluid")
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
freq = 200.0
c_0 = 343.0 
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
fes = Periodic(H1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("fluid")))

p, q = fes.TnT()  # p,q = acoustic ; u,w = elastic

print(f"Total FSI Degrees of Freedom: {fes.ndof}")

# =====================================================================
# 3. Define variational forms
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# Differential Operators 
def grad_p(p): return grad(p) + 1j * k_vec * p
def grad_q(q): return grad(q) - 1j * k_vec * q

# Uncoupled Bilinear Form (FEM)
Z_fem = BilinearForm(fes)
Z_fem += (grad_p(p) * grad_q(q) - k**2 * p * q) * dx("fluid")               

# Linear Form (RHS) - Surface source on the bottom boundary
# s_fem = LinearForm(fes)
# source_func = exp(-((z-Lz*0.4)**2)/(L/20)**2) / (L/20 * sqrt(pi))
# s_fem += 0* q * dx("fluid")

# Assemble 
with TaskManager():
    Z_fem.Assemble()
    # s_fem.Assemble()

print("Assembly complete!")

# =====================================================================
# 4. Build WBM model and coupling matrices
# =====================================================================
print("\n 4. Build WBM model and coupling matrices...\n")
m_max = 0
n_max = 0
wbm_top = WBM_Top(Lx, Ly, L, freq, c_0, rho_air, m_max, n_max, theta, phi)
Z_hyb_top, Z_wbm_top, s_hyb_top, s_wbm_top = wbm_top.assemble_matrices(mesh, fes, q, "top")
wbm_bottom = WBM_Bottom(Lx, Ly, 0, freq, c_0, rho_air, m_max, n_max, theta, phi)
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

# 1. Reconstruct Field
phase = exp(1j * (kx * x + ky * y))
gfu_p = gfu_p * phase

t = np.linspace(0, 1, 200)
xs = 0* t + Lx/2
ys = 0* t + Ly/2
zs = t* Lz
vals = gfu_p(mesh(xs, ys, zs)).reshape(-1)


rel_error = np.abs(abs(vals) - 1) 

"""
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

"""
Draw(gfu_p, mesh, name="Pressure")
input("FSI simulation completed! Press Enter to exit...")
