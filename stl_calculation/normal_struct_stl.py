from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

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
Lz_plate = 0.01 * L

# Meshing parameters
maxh = 0.2
minh = 0.1
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# Physics parameters
freq_max = 1000.0
freq_min = 100.0
N_freq = 10
theta_max = 78.0/180.0*math.pi
theta_min = 0.0
N_theta = 5

phi = 0.0
c_air = 343.0* (1 - 1j * 0.001)
rho_air = 1.21
E_steel = 210e9* (1 - 1j * 0.01)
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
freq_arr = np.linspace(freq_min, freq_max, N_freq)
theta_arr = np.linspace(theta_min, theta_max, N_theta)
tau_2d = np.zeros((N_freq, N_theta))


for i_freq, freq in enumerate(freq_arr):
    print("----------------------------------------------------------------")
    print(f"Solving for frequency {freq:.1f} Hz...")
    for j_theta, theta in enumerate(theta_arr):
        print(f"  Incidence angle: {theta*180/math.pi:.1f} degrees")
        Global_Matrix, Global_RHS, free_indices, fes, kx, ky = get_bare_plate_matrices(
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
        tau_2d[i_freq, j_theta] = abs(gfu_p(mesh(Lx/2, Ly/2, Lz*0.99)))**2 / abs(p_ref)**2

integrand = 2 * tau_2d * np.cos(theta_arr) * np.sin(theta_arr)
integral  = np.trapezoid(integrand, x=theta_arr, axis=1)   
TL_diffuse = - 10 * np.log10(np.clip(integral, 1e-12, None))
tau_2d = np.clip(tau_2d, 1e-12, None)
TL_2d = - 10 * np.log10(tau_2d)

plt.figure()

# Plot results for selected angles
for i_angle, angle in enumerate(theta_arr):
    plt.semilogx(freq_arr, TL_2d[:, i_angle], label=f'{angle*180/np.pi:.1f}°')
# Diffuse field curve
plt.semilogx(freq_arr, TL_diffuse, 'k--', linewidth=2, label='Diffuse field (0°–80°)')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Transmission Loss (dB)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlim(100, 1000)
plt.tight_layout()
plt.show()   
