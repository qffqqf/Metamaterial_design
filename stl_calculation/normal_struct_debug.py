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
Lx = L/20
Ly = L/20
Lz = L
Lz_1 = 0.5 * L
Lz_plate = 0.02 * L

# Meshing parameters
maxh = 0.08
minh = 0.06
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# Physics parameters
freq = 343.0/2
theta = math.pi / 4*0
phi = 0.0
c_air = 343.0* (1 - 1j * 0.001)
rho_air = 1.21
E_steel = 210e9
nu_steel = 0.3
rho_steel = 7800.0
m_max = 2
n_max = 2

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

t = np.linspace(0, 1, 200)
xs = 0* t + Lx/2
ys = 0* t + Ly/2
zs = t* Lz
vals = gfu_p(mesh(xs, ys, zs)).reshape(-1)

b_w = ((E_steel * Lz_plate**3) / 12 / (1 - nu_steel**2)* kx**4 - rho_steel * (freq*2*np.pi)**2 * Lz_plate)/ (rho_air * (freq*2*np.pi)**2)
beta = 1j* kz * b_w
k = 2 * math.pi * freq / c_air
Inc = np.exp(- k**2 * (L/40)**2)/(2*k)
z0 = Lz/2

def kl_plate_solution(z, z0, beta, Inc):
    
    p_I = Inc * np.exp(1j* kz * (z - z0)) * np.where((z > Lz/4) & (z < z0), 1, 0)
    p_R = beta/(beta + 2) * Inc * np.exp(-1j* kz * (z - z0)) * np.where((z < z0), 1, 0)
    p_T = 2/(beta + 2) * Inc * np.exp(1j* kz * (z - z0)) * np.where((z > z0), 1, 0)
    p_A = Inc * np.exp(-1j* kz * (z - z0)) * np.where((z < Lz/4), 1, 0)
    return p_I + p_R + p_T + p_A

P_analytical = kl_plate_solution(zs, z0, beta, Inc)

rel_error = np.abs(abs(vals) - abs(P_analytical)) / (abs(P_analytical) + 1e-15)

# Create subplots
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Top Plot: Comparison
ax1.semilogy(zs, abs(vals), 'b-', label="Numerical", alpha=0.7)
ax1.semilogy(zs, abs(P_analytical), 'r--', label="Analytical")
ax1.semilogy(zs, abs(np.exp(- (2*np.pi*freq/c_air)**2 * (L/40)**2)/(4*np.pi*freq/c_air))*np.ones_like(zs), '--', label="Asymptotic")

ax1.set_ylabel("Pressure $p(z)$")
ax1.set_title("Pressure distribution along z-axis")
ax1.legend()
ax1.grid(True)

# Bottom Plot: Relative Error
ax2.semilogy(zs, rel_error, 'k-', label="Relative Error")
ax2.set_xlabel("z")
ax2.set_ylabel("Relative Error (log scale)")
ax2.grid(True, which="both", ls="-", alpha=0.5)
ax2.legend()

plt.show()
