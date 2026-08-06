from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import scipy.sparse.linalg as spla

import netgen.gui 
from get_sys_mat import get_bare_plate_matrices, get_lrm_plate_matrices

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
L = 0.005
Lx = 0.05
Ly = 0.05
Lz = L
Lz_1 = 0.2 * L
Lz_plate = 0.001

# Meshing parameters
maxh = 0.03
minh = 0.01
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# Physics parameters
freq_max = 3500.0
freq_min = 10.0
N_freq = 800

theta = math.pi / 4* 0
phi = 0.0
c_air = 343.0* (1 - 1j * 0.0)
rho_air = 1.21
E_steel = 70e9* (1 - 1j * 0.0)
nu_steel = 0.3
rho_steel = 2700.0
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
freq_arr = np.linspace(freq_min, freq_max, N_freq)
tau_2d = np.zeros((N_freq))


for i_freq, freq in enumerate(freq_arr):
    print("----------------------------------------------------------------")
    print(f"Solving for frequency {freq:.1f} Hz...")
    Global_Matrix, Global_RHS, free_indices, fes, kx, ky, kz = get_lrm_plate_matrices(
        freq, theta, phi, Lx, Ly, Lz, Lz_1, Lz_plate, c_air, rho_air, E_steel, nu_steel, rho_steel, mesh, curve_order, m_max, n_max)
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
    p_ref = np.exp(- kz**2 * (L/40)**2)/(2*kz)
    tau_2d[i_freq] = abs(gfu_p(mesh(0, 0, 0)))**2

TL_2d = - 10 * np.log10(tau_2d)

kx_arr = 2 * math.pi * freq_arr / c_air * math.sin(theta) * math.cos(phi)
kz_arr = 2 * math.pi * freq_arr / c_air * math.cos(theta)
omega_arr = 2 * math.pi * freq_arr
D_kl = E_steel * Lz_plate**3 / (12 * (1 - nu_steel**2))
b_kl = (D_kl * kx_arr**4 - rho_steel * omega_arr**2 * Lz_plate)/ (rho_air * omega_arr**2)
S_mh = E_steel / (2 * (1 + nu_steel)) * Lz_plate *5/6
I_mh = 1.8*rho_steel * Lz_plate**3 / 12
d_mh = S_mh * (D_kl * kx_arr**4 - I_mh * omega_arr**2 * kx_arr**2) / (D_kl* kx_arr**2 + S_mh - I_mh * omega_arr**2)
l_mh = (d_mh - 1.8*rho_steel * omega_arr**2 * Lz_plate)/ (rho_air * omega_arr**2)

TL_kl = - 10 * np.log10(abs(2/(1j* kz_arr * b_kl + 2))**2)
TL_mh = - 10 * np.log10(abs(2/(1j* kz_arr * l_mh + 2))**2)

"""
Draw(gfu_p, mesh, "Pressure_Field")
input("Simulation completed! Press Enter to continue...")
"""
import matplotlib.pyplot as plt
# Plot results for selected angles
plt.axvline(x=446, ymin=0, ymax=80/plt.ylim()[1] if plt.ylim()[1] else 1, 
            color='k', linestyle='-', linewidth=.5, alpha=0.7)
plt.axvline(x=644, ymin=0, ymax=80/plt.ylim()[1] if plt.ylim()[1] else 1,
            color='k', linestyle='-', linewidth=.5, alpha=0.7)

plt.plot(freq_arr, TL_2d, linewidth=2, linestyle="-", label='LRM')
plt.plot(freq_arr, TL_kl, linewidth=2, linestyle="--", label='$\\rho_{host}$')
plt.plot(freq_arr, TL_mh, linewidth=2, linestyle="-.", label='$1.8\\rho_{host}$')
plt.legend(loc='lower right')
plt.ylabel('Transmission Loss [dB]', fontsize=12)   # <- changed from plt.set_ylabel
plt.xlabel('Frequency [Hz]', fontsize=12)           # optional, recommended
plt.grid(True, which='both', linestyle='-', alpha=0.3)
plt.xlim(freq_min, freq_max)                        # <- changed from plt.set_xlim
plt.ylim(0, 80)                                

plt.tight_layout()
plt.show()
