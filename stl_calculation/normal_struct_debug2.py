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
freq_max = 1000.0
freq_min = 10.0
N_freq = 20

theta = math.pi /16
phi = 0.0
c_air = 343.0* (1 - 1j * 0.00001)
rho_air = 1.21
E_steel = 210e9* (1 - 1j * 0.001)
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
freq_arr = np.logspace(np.log10(freq_min), np.log10(freq_max), N_freq)
tau_2d = np.zeros((N_freq, 1))


for i_freq, freq in enumerate(freq_arr):
    print("----------------------------------------------------------------")
    print(f"Solving for frequency {freq:.1f} Hz...")
    Global_Matrix, Global_RHS, free_indices, fes, kx, ky, kz = get_bare_plate_matrices(
        freq, theta, phi, Lx, Ly, Lz, c_air, rho_air, E_steel, nu_steel, rho_steel, mesh, curve_order, m_max, n_max)
    solution = spla.spsolve(Global_Matrix, Global_RHS)
    fem_vals = solution[:len(free_indices)]
    wbm_factors = solution[len(free_indices):]
    fem_gf = GridFunction(fes)
    fem_gf.vec.FV().NumPy()[free_indices] = fem_vals
    # 1. Reconstruct Field
    phase = exp(1j * (kx * x + ky * y))
    gfu_p, gfu_u = fem_gf.components
    gfu_p = gfu_p * phase
    gfu_u = gfu_u * phase
    p_ref = np.exp(- (2*np.pi*freq/c_air)**2 * (L/40)**2)/(4*np.pi*freq/c_air)
    tau_2d[i_freq] = abs(gfu_p(mesh(0, 0, Lz)))**2 / abs(p_ref)**2

TL_2d = - 10 * np.log10(tau_2d)

kx_arr = 2 * math.pi * freq_arr / c_air * math.sin(theta) * math.cos(phi)
kz_arr = 2 * math.pi * freq_arr / c_air * math.cos(theta)
b_w = ((E_steel * Lz_plate**3) / 12 / (1 - nu_steel**2)* kx_arr**4 - rho_steel * (freq_arr*2*np.pi)**2 * Lz_plate)/ (rho_air * (freq_arr*2*np.pi)**2)
beta = 1j* kz_arr * b_w
TL_an = - 10 * np.log10(abs(2/(beta + 2))**2)

import matplotlib.pyplot as plt
plt.figure()
# Plot results for selected angles
plt.semilogx(freq_arr, TL_2d, 'b-', linewidth=2, label='Diffuse field (0°–80°)')
plt.semilogx(freq_arr, TL_an, 'r--', linewidth=2, label='Analytical solution')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Transmission Loss (dB)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlim(10, 1000)
plt.tight_layout()
plt.show()   