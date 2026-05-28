from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import scipy.sparse.linalg as spla

import netgen.gui 
from get_sys_mat import get_bare_plate_matrices

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
L = 4.0
Lx = L/2
Ly = L/8
Lz = L
Lz_1 = 0.8 * L
Lz_plate = 0.01 * L

# Meshing parameters
maxh = 0.2
minh = 0.1
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# Physics parameters
freq = 200.0
omega = 2 * math.pi * freq

theta = math.pi / 4
phi = 0.0
c_air = 343.0
rho_air = 1.21
E_steel = 210e9
nu_steel = 0.3
rho_steel = 7800.0
m_max = 1
n_max = 1

# 1. Air Volumes
duct = Box((0, 0, 0), (Lx, Ly, Lz))
duct.faces.Max(X).Identify(duct.faces.Min(X), "periodic_x")
duct.faces.Max(Y).Identify(duct.faces.Min(Y), "periodic_y")

# 2. Plate Volumes
plate_domains = Box((0, 0, Lz_1), (Lx, Ly, Lz_1 + Lz_plate))
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
# 2. Get system matrices for the bare plate case (no WBM coupling)
# =====================================================================
print("\n 2. Build and solve the coupled system...\n")
Global_Matrix, Global_RHS, free_indices, fes, kx, ky, kz = get_bare_plate_matrices(
    freq, theta, phi, Lx, Ly, Lz, c_air, rho_air, E_steel, nu_steel, rho_steel, mesh, curve_order, m_max, n_max)

solution = spla.spsolve(Global_Matrix, Global_RHS)
fem_vals = solution[:len(free_indices)]
wbm_factors = solution[len(free_indices):]
fem_gf = GridFunction(fes)
fem_gf.vec.FV().NumPy()[free_indices] = fem_vals
print("Coupled system solved successfully!")

# 1. Reconstruct Field
phase = exp(1j * (kx * x + ky * y))
gfu_p, gfu_u = fem_gf.components
gfu_p = gfu_p * phase
gfu_u = gfu_u * phase

"""
Draw(gfu_p, mesh, name="Pressure")
input("FSI simulation completed! Press Enter to exit...")
"""

t = np.linspace(0, 1, 200)
xs = 0* t + Lx/2
ys = 0* t + Ly/2
zs = t* Lz
t1 = np.linspace(0, 1, 10)
vals = gfu_p(mesh(xs, ys, zs)).reshape(-1)
u_fem = gfu_u(mesh(t1*Lx, t1*Ly, t1*Lz_plate + Lz_1)).reshape(-1, 3)
print("displacement at plate center:", np.mean(abs(u_fem), axis=0))

b_w = ((E_steel * Lz_plate**3) / 12 / (1 - nu_steel**2)* kx**4 - rho_steel * (freq*2*np.pi)**2 * Lz_plate)/ (rho_air * (freq*2*np.pi)**2)
beta = 1j* kz * b_w
k = 2 * math.pi * freq / c_air
Inc = np.exp(- kz**2 * (L/40)**2)/(2*kz)
z0 = Lz*0.8

D_kl = E_steel * Lz_plate**3 / (12 * (1 - nu_steel**2))
S_mh = E_steel / (2 * (1 + nu_steel)) * Lz_plate *5/6
I_mh = rho_steel * Lz_plate**3 / 12
d_mh = S_mh * (D_kl * kx**4 - I_mh * omega**2 * kx**2) / (D_kl* kx**2 + S_mh - I_mh * omega**2)
l_mh = (d_mh - rho_steel * omega**2 * Lz_plate)/ (rho_air * omega**2)
beta_mh = 1j* kz * l_mh

u_kl = kz/ (rho_air * omega**2) * 2/(beta + 2) * Inc
print("KL model plate center displacement:", abs(u_kl))
u_mh = kz/ (rho_air * omega**2) * 2/(beta_mh + 2) * Inc
print("MH model plate center displacement:", abs(u_mh))
print("check periodic BCs (should be close to 0):", exp(1j* kx * Lx) - gfu_p(mesh(Lx, Ly, Lz))/gfu_p(mesh(0, Ly, Lz)))

def kl_plate_solution(z, z0, beta, Inc):
    
    p_I = Inc * np.exp(1j* kz * (z - z0)) * np.where((z > Lz* 0.4) & (z < z0), 1, 0)
    p_R = beta/(beta + 2) * Inc * np.exp(-1j* kz * (z - z0)) * np.where((z < z0), 1, 0)
    p_T = 2/(beta + 2) * Inc * np.exp(1j* kz * (z - z0)) * np.where((z > z0), 1, 0)
    p_A = Inc * np.exp(-1j* kz * (z + z0 - Lz*0.4*2)) * np.where((z < Lz* 0.4), 1, 0)
    return p_I + p_R + p_T + p_A

P_kl = kl_plate_solution(zs, z0, beta, Inc)
P_mh = kl_plate_solution(zs, z0, beta_mh, Inc)

rel_error_kl = np.abs(abs(vals) - abs(P_kl)) / (abs(P_kl) + 1e-15)
rel_error_mh = np.abs(abs(vals) - abs(P_mh)) / (abs(P_mh) + 1e-15)

vals[(zs > Lz_1) & (zs < Lz_1 + Lz_plate)] = np.nan
P_kl[(zs > Lz_1) & (zs < Lz_1 + Lz_plate)] = np.nan
P_mh[(zs > Lz_1) & (zs < Lz_1 + Lz_plate)] = np.nan
rel_error_kl[(zs > Lz_1) & (zs < Lz_1 + Lz_plate)] = np.nan
rel_error_mh[(zs > Lz_1) & (zs < Lz_1 + Lz_plate)] = np.nan

# Create subplots
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Top Plot: Comparison
ax1.semilogy(zs, abs(vals), '-', label="FEM-WBM")
ax1.semilogy(zs, abs(P_kl), '--', label="Kirchhoff model")
ax1.semilogy(zs, abs(P_mh), '-.', label="Mindlin model")
# ax1.set_xlim(Lz/2, Lz)
ax1.set_ylabel("$|p(z)|$ [Pa]", fontsize=12)
ax1.legend()
ax1.grid(True, which='both', linestyle='-', alpha=0.3)

# Bottom Plot: Relative Error
ax2.semilogy(zs, rel_error_kl, '-', label="Kirchhoff model")
ax2.semilogy(zs, rel_error_mh, '-', label="Mindlin model")
# ax2.set_xlim(Lz/2, Lz)
ax2.set_xlabel("z [m]", fontsize=12)
ax2.set_ylabel("Relative difference [-]", fontsize=12)
ax2.grid(True, which='both', linestyle='-', alpha=0.3)
ax2.legend()

plt.show()
