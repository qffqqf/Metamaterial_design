from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import numpy as np
import math
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

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
h_tube = 200e-3
rho_air = 1.21
c_air = 343.0* (1 + 0.001j) # speed of sound in air

h_plate = 150e-3 # plate position

freq = 2000.0
omega = 2 * math.pi * freq
k = 2 * math.pi * freq / c_air

# Meshing parameters
maxh = 12e-3
minh = 8e-3
mp = MeshingParameters(maxh=maxh, minh=minh)
curve_order = 3

# 1. Air Volumes
duct = Cylinder(Pnt(r_tube, r_tube, 0), \
                gp_Vec(0,0,1), r_tube, h_tube)

# 3. Air Domains (subtract plate volumes from duct)
air_domains = duct
air_domains.mat("fluid")
air_domains.faces.Max(Z).name = "top"
air_domains.faces.Min(Z).name = "bottom"

# Detect interfaces and rename them for coupling
geo = OCCGeometry(air_domains)
mesh = Mesh(geo.GenerateMesh(mp=mp))

# Draw(mesh)
# input("Mesh generated successfully! Press Enter to continue...")
print("Mesh generated successfully!")

# =====================================================================
# 2. Define finite element space
# =====================================================================
print("\n 2. Define finite element space...\n")
# Finite Element Space
fes = H1(mesh, order=curve_order, complex=True)
p, q = fes.TnT()  # p,q = acoustic ; u,w = elastic
print(f"Total FSI Degrees of Freedom: {fes.ndof}")

# =====================================================================
# 3. Define variational forms
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# Uncoupled Bilinear Form (FEM)
Z_fem = BilinearForm(fes)
Z_fem += (grad(p) * grad(q) - k**2 * p * q) * dx             
Z_fem += -1j * k * p * q * ds(definedon=mesh.Boundaries("bottom|top"))

# Linear Form (RHS) - Surface source on the bottom boundary
s_fem = LinearForm(fes)
source_func = exp(-((z-h_tube*0.4)**2)/(h_tube/20)**2) / (h_tube/20 * sqrt(pi))
s_fem += source_func* q * dx

# Assemble 
gfu = GridFunction(fes)
with TaskManager():
    Z_fem.Assemble()
    s_fem.Assemble()
    gfu.vec.data = Z_fem.mat.Inverse() * s_fem.vec
print("Solve complete!")


# Draw(gfu, mesh, name="Pressure")
# input("FSI simulation completed! Press Enter to exit...")

# 1. Reconstruct Field
t = np.linspace(0, 1, 200)
xs = 0* t + r_tube
ys = 0* t + r_tube
zs = t* h_tube
vals = gfu(mesh(xs, ys, zs)).reshape(-1)

from scipy.special import erf, erfc
def gaussian_source_solution(x, mu, sigma, B, k):
    """
    Exact solution of d^2p/dx^2 + k^2 p = B exp(-(x-mu)^2/(2 sigma^2))
    on the whole real line.
    """
    const = B * sigma * np.sqrt(np.pi / 2) / (2j * k)
    exp_factor = np.exp(-0.5 * k**2 * sigma**2)
    
    # Correction: +1j for z1, -1j for z2
    z1 = (x - mu + 1j * k * sigma**2) / (np.sqrt(2) * sigma)
    z2 = (x - mu - 1j * k * sigma**2) / (np.sqrt(2) * sigma)
    
    term1 = np.exp(1j * k * (x - mu)) * (1.0 + erf(z1))
    term2 = np.exp(-1j * k * (x - mu)) * erfc(z2)
    
    return const * exp_factor * (term1 + term2)

P_analytical = gaussian_source_solution(zs, mu=h_tube*0.4, sigma=h_tube/40*sqrt(2), B=1 / (h_tube/20 * sqrt(pi)), k=k)

rel_error = np.abs(abs(vals) - abs(P_analytical)) / (abs(P_analytical) + 1e-15)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Top Plot: Comparison
ax1.plot(zs, abs(vals), '-', label="FEM-WBM", alpha=0.7)
ax1.plot(zs, abs(P_analytical), '--', label="Analytical solution")
ax1.plot(zs, abs(np.exp(- k**2 * (h_tube/40)**2)/(2*k))*np.ones_like(zs), '-.', label="Asymptotic solution", color='green')

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