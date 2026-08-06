from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import scipy.sparse.linalg as spla

import netgen.gui 

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
r_rbr = 51e-3 # rubber
h_rbr = 1e-3
rho_rbr = 1590.0
E_rbr = 1.5e6
nu_rbr = 0.49
mu_rbr = E_rbr / (2 * (1 + nu_rbr))
lam_rbr = E_rbr * nu_rbr / ((1 + nu_rbr) * (1 - 2 * nu_rbr))

r_stl = 61.25e-3 # steel
h_stl = 1.5e-3
rho_stl = 7800.0
E_stl = 210e9
nu_stl = 0.3
mu_stl = E_stl / (2 * (1 + nu_stl))
lam_stl = E_stl * nu_stl / ((1 + nu_stl) * (1 - 2 * nu_stl))

r_res = 5e-3 # resonator
h_res = 3e-3
rho_res = 7855.6
E_res = 210e9
nu_res = 0.3
mu_res = E_res / (2 * (1 + nu_res))
lam_res = E_res * nu_res / ((1 + nu_res) * (1 - 2 * nu_res))

r_tube = 61.25e-3 # tube
h_tube = 20e-3
rho_air = 1.21
c_air = 343.0 # speed of sound in air
l_tube = 2* r_tube

h_plate = 5e-3 # plate position

# Physics parameters
freq_max = 3500.0
freq_min = 10.0
N_freq = 800

theta = math.pi / 4* 0
phi = 0.0

# Meshing parameters
maxh = 8e-3
minh = 4e-3
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 2

# 1. Air Volumes
duct = Box((0, 0, 0), (l_tube, l_tube, h_tube))
duct.faces.Max(X).Identify(duct.faces.Min(X), "periodic_x")
duct.faces.Max(Y).Identify(duct.faces.Min(Y), "periodic_y")

# 2. Plate Volumes
stl_1 = Box((0, 0, h_plate), (l_tube, l_tube, h_plate + h_stl))\
      - Cylinder(Pnt(r_tube, r_tube, h_plate), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_1.mat("steel1")

rbbr = Box((0, 0, h_plate+h_stl), (l_tube, l_tube, h_plate+h_stl+h_rbr))
rbbr.mat("rubber")

stl_2 = Box((0, 0, h_plate+h_stl+h_rbr), (l_tube, l_tube, h_plate+h_stl+h_rbr + h_stl))\
      - Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_2.mat("steel2")

reso = Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
               gp_Vec(0,0,1), r_res, h_res)
reso.mat("resonator")
solid_domains = Glue([stl_1, rbbr, stl_2, reso])
solid_domains.mat("solid")
solid_domains.faces.Max(X).Identify(solid_domains.faces.Min(X), "periodic_x")
solid_domains.faces.Max(Y).Identify(solid_domains.faces.Min(Y), "periodic_y")

# 3. Air Domains (subtract plate volumes from duct)
air_cut = duct - solid_domains
air_cut.mat("fluid")

# Detect interfaces and rename them for coupling
for f in air_cut.faces:
    f.name = "fluid_boundary"

combined_geo = Glue([air_cut, solid_domains])

for f in solid_domains.faces:
    if f.name == "fluid_boundary":
        f.name = "fsi_interface"

combined_geo.faces.Max(Z).name = "top"
combined_geo.faces.Min(Z).name = "bottom"

# Separate the mesh
for s in combined_geo.solids:
    print(f"Solid: {s.name}")
    if s.name == "fluid1" or s.name == "fluid3":
        print(f"Setting maxh for {s.name} to 2e-3")
        s.maxh = 30e-3
    elif s.name == "fluid2":
        s.maxh = 5e-3
    elif s.name == "rubber":
        s.maxh = 5e-3
    elif s.name == "resonator":
        s.maxh = 5e-3

geo = OCCGeometry(combined_geo, dim=3)
ngmesh = geo.GenerateMesh()
mesh = Mesh(ngmesh)

interface_marker = mesh.BoundaryCF({"fsi_interface": 1}, default=0)
Draw(interface_marker, mesh, "fsi_interface")
input("Mesh generated successfully! Press Enter to continue...")
print("Mesh generated successfully!")

# =====================================================================
# 2. Get system matrices for the bare plate case (no WBM coupling)
# =====================================================================
print("\n 2. Build and solve the coupled system...\n")
freq_arr = np.linspace(freq_min, freq_max, N_freq)
tau_2d = np.zeros((N_freq))

"""
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

TL_kl = - 10 * np.log10(abs(2/(1j* kz_arr * b_kl + 2))**2)
TL_mh = - 10 * np.log10(abs(2/(1j* kz_arr * l_mh + 2))**2)

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
"""

