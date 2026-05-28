from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.integrate import simpson

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
Lz_1 = 0.8 * L
Lz_plate = 0.01 * L

# Meshing parameters
maxh = 0.1
minh = 0.08
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# Physics parameters
freq_max = 1000.0
freq_min = 10.0
N_freq = 100

theta_min = 0.0
theta_max = math.pi / 30*13
N_theta = 30

phi = 0.0
c_air = 343.0* (1 - 1j * 0.001)
rho_air = 1.21
E_steel = 210e9* (1 - 1j * 0.02)
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
# freq_arr = np.logspace(np.log10(freq_min), np.log10(freq_max), N_freq)
freq_arr = np.linspace(freq_min, freq_max, N_freq)
theta_arr = np.linspace(theta_min, theta_max, N_theta)
tau_2d = np.zeros((N_freq, N_theta))


for i_freq, freq in enumerate(freq_arr):
    print("----------------------------------------------------------------")
    print(f"Solving for frequency {freq:.1f} Hz...")
    for j_theta, theta in enumerate(theta_arr):
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
        p_ref = np.exp(- kz**2 * (L/40)**2)/(2*kz)
        tau_2d[i_freq, j_theta] = abs(gfu_p(mesh(0, 0, Lz)))**2 / abs(p_ref)**2

integrand = 2 * tau_2d * np.cos(theta_arr) * np.sin(theta_arr)
integral  = simpson(integrand, x=theta_arr, axis=1)  
integrand_c = 2 * np.cos(theta_arr) * np.sin(theta_arr)
integral_c = simpson(integrand_c, x=theta_arr)

TL_diffuse = - 10 * np.log10(np.clip(integral/integral_c, 1e-12, None))

# grid
freq_grid, theta_grid = np.meshgrid(freq_arr, theta_arr, indexing='ij')
kz_grid = 2 * math.pi * freq_grid / c_air * np.cos(theta_grid)
kx_grid = 2 * math.pi * freq_grid / c_air * np.sin(theta_grid) * np.cos(phi)
omega_grid = 2 * math.pi * freq_grid

D_kl = E_steel * Lz_plate**3 / (12 * (1 - nu_steel**2))
b_kl = (D_kl * kx_grid**4 - rho_steel * omega_grid**2 * Lz_plate)/ (rho_air * omega_grid**2)
S_mh = E_steel / (2 * (1 + nu_steel)) * Lz_plate *5/6
I_mh = rho_steel * Lz_plate**3 / 12
d_mh = S_mh * (D_kl * kx_grid**4 - I_mh * omega_grid**2 * kx_grid**2) / (D_kl* kx_grid**2 + S_mh - I_mh * omega_grid**2)
l_mh = (d_mh - rho_steel * omega_grid**2 * Lz_plate)/ (rho_air * omega_grid**2)

# Integrate over θ with diffuse‑field weighting
tau_kl = np.abs(2 / (1j * kz_grid * b_kl + 2))**2
integrand_kl = 2 * tau_kl * np.cos(theta_grid) * np.sin(theta_grid)
integral_kl  = simpson(integrand_kl, x=theta_arr, axis=1)
TL_diffuse_kl = -10 * np.log10(np.clip(integral_kl/integral_c, 1e-12, None))

tau_mh = np.abs(2 / (1j * kz_grid * l_mh + 2))**2
integrand_mh = 2 * tau_mh * np.cos(theta_grid) * np.sin(theta_grid)
integral_mh  = simpson(integrand_mh, x=theta_arr, axis=1)
TL_diffuse_mh = -10 * np.log10(np.clip(integral_mh/integral_c, 1e-12, None))

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot results for selected angles
ax1.semilogx(freq_arr, TL_diffuse, linewidth=2, linestyle="-", label='FEM-WBM')
ax1.semilogx(freq_arr, TL_diffuse_kl, linewidth=2, linestyle="--", label='Kirchhoff model')
ax1.semilogx(freq_arr, TL_diffuse_mh, linewidth=2, linestyle="-.", label='Mindlin model')
ax1.legend()
ax1.set_ylabel('Transmission Loss [dB]', fontsize=12)
ax1.grid(True, which='both', linestyle='-', alpha=0.3)
ax1.set_xlim(10, 1000)

rel_error_kl = np.abs(abs(TL_diffuse) - abs(TL_diffuse_kl)) / (abs(TL_diffuse_kl) + 1e-15)
rel_error_mh = np.abs(abs(TL_diffuse) - abs(TL_diffuse_mh)) / (abs(TL_diffuse_mh) + 1e-15)

ax2.loglog(freq_arr, rel_error_kl, linewidth=2, linestyle="-", label='Kirchhoff model')
ax2.loglog(freq_arr, rel_error_mh, linewidth=2, linestyle="-", label='Mindlin model')
ax2.legend()
ax2.set_xlabel('Frequency [Hz]', fontsize=12)
ax2.set_ylabel('Relative difference [-]', fontsize=12)
ax2.grid(True, which='both', linestyle='-', alpha=0.3)
ax2.set_xlim(10, 1000)
plt.tight_layout()
plt.show()   


