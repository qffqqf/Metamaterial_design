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
h_tube = 200e-3
rho_air = 1.21
c_air = 343.0* (1 + 0.001j) # speed of sound in air

h_plate = 140e-3 # plate position

freq_min = 20.0
freq_max = 2000.0
N_freq = 12
freq_arr = np.linspace(freq_min, freq_max, N_freq)

# Meshing parameters
curve_order = 2

# 1. Air Volumes
duct1 = Cylinder(Pnt(r_tube, r_tube, 0), \
                gp_Vec(0,0,1), r_tube, h_plate-2*h_stl)
duct1.mat("fluid1")
duct2 = Cylinder(Pnt(r_tube, r_tube, h_plate-2*h_stl), \
                gp_Vec(0,0,1), r_tube, h_rbr+12*h_stl+h_res)
duct3 = Cylinder(Pnt(r_tube, r_tube, h_plate+h_rbr+10*h_stl+h_res), \
                gp_Vec(0,0,1), r_tube, h_tube-(h_plate+h_rbr+10*h_stl+h_res))
duct3.mat("fluid3")

# 2. Plate Volumes
stl_1 = Cylinder(Pnt(r_tube, r_tube, h_plate), \
                 gp_Vec(0,0,1), r_tube, h_stl, mantle="stl1_outer") \
      - Cylinder(Pnt(r_tube, r_tube, h_plate), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_1.mat("steel1")

rbbr = Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl), \
               gp_Vec(0,0,1), r_tube, h_rbr, mantle="rbbr_outer")
rbbr.mat("rubber")

stl_2 = Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
                 gp_Vec(0,0,1), r_tube, h_stl, mantle="stl2_outer") \
      - Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
                 gp_Vec(0,0,1), r_rbr, h_stl)
stl_2.mat("steel2")

reso = Cylinder(Pnt(r_tube, r_tube, h_plate+h_stl+h_rbr), \
               gp_Vec(0,0,1), r_res, h_res)
reso.mat("resonator")
solid_domains = Glue([stl_1, rbbr, stl_2, reso])

# 3. Air Domains (subtract plate volumes from duct)
air_cut = duct2 - solid_domains
air_cut.mat("fluid2")

# Detect interfaces and rename them for coupling
for f in air_cut.faces:
    f.name = "fluid_boundary"

combined_geo = Glue([air_cut, duct1, duct3, solid_domains])

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

# interface_marker = mesh.BoundaryCF({"stl2_outer": 1}, default=0)
# Draw(interface_marker, mesh, "stl2_outer")
# Draw(mesh)
# input("Mesh generated successfully! Press Enter to continue...")
print("Mesh generated successfully!")

tau = np.zeros((N_freq))

for i_freq, freq in enumerate(freq_arr):
    print("----------------------------------------------------------------")
    print(f"Solving for frequency {freq:.1f} Hz...")
    omega = 2 * np.pi * freq
    k = omega / c_air
    # =====================================================================
    # 2. Define physics and finite element space
    # =====================================================================
    print("\n 2. Define physics and finite element space...\n")
    # Finite Element Space
    fes_air = H1(mesh, order=curve_order, complex=True, \
                definedon=mesh.Materials("fluid1|fluid2|fluid3"))
    fes_plate = VectorH1(mesh, order=curve_order, complex=True, \
                        definedon=mesh.Materials("steel1|rubber|steel2|resonator"), \
                        dirichlet="stl1_outer|rbbr_outer|stl2_outer")
    fes = (fes_air * fes_plate)
    (p, u), (q, w) = fes.TnT()  # p,q = acoustic ; u,w = elastic
    print(f"Total FSI Degrees of Freedom: {fes.ndof}")

    # =====================================================================
    # 3. Define variational forms
    # =====================================================================
    print("\n 3. Define variational forms and assemble FE model...\n")
    # Differential Operators 
    def strain(u): return Sym(Grad(u))
    def stress_stl(u): return 2*mu_stl*strain(u) + lam_stl*Trace(strain(u))*Id(3)
    def stress_rbr(u): return 2*mu_rbr*strain(u) + lam_rbr*Trace(strain(u))*Id(3)
    def stress_res(u): return 2*mu_res*strain(u) + lam_res*Trace(strain(u))*Id(3)
    # Uncoupled Bilinear Form (FEM)
    Z_fem = BilinearForm(fes)
    Z_fem += (grad(p) * grad(q) - k**2 * p * q) * dx("fluid1|fluid2|fluid3")
    Z_fem += -1j * k * p * q * ds(definedon=mesh.Boundaries("bottom|top"))
    Z_fem += (InnerProduct(stress_stl(u), strain(w)) - rho_stl * omega**2 * InnerProduct(u, w)) * dx("steel1|steel2")
    Z_fem += (InnerProduct(stress_rbr(u), strain(w)) - rho_rbr * omega**2 * InnerProduct(u, w)) * dx("rubber")
    Z_fem += (InnerProduct(stress_res(u), strain(w)) - rho_res * omega**2 * InnerProduct(u, w)) * dx("resonator")
    # FSI Interface Coupling 
    n_plate_to_air = specialcf.normal(mesh.dim)
    Z_fem += rho_air * omega**2 * (n_plate_to_air * u)* q * ds("fsi_interface")    
    Z_fem += p * (n_plate_to_air *w) * ds("fsi_interface")      
    # Linear Form (RHS) - Surface source on the bottom boundary
    s_fem = LinearForm(fes)
    source_func = exp(-((z-h_tube*0.4)**2)/(h_tube/20)**2) / (h_tube/20 * sqrt(pi))
    s_fem += source_func* q * dx       

    # Assemble 
    gfu = GridFunction(fes)
    with TaskManager():
        Z_fem.Assemble()
        s_fem.Assemble()
        gfu.vec.data = Z_fem.mat.Inverse(fes.FreeDofs()) * s_fem.vec
    print("Solve complete!")
    gfu_p, gfu_u = gfu.components
    p_ref = np.exp(- k**2 * (h_tube/40)**2)/(2*k)
    tau[i_freq] = abs(gfu_p(mesh(r_tube, r_tube, h_tube)))**2 / abs(p_ref)**2

TL = - 10 * np.log10(tau)
import matplotlib.pyplot as plt
plt.semilogx(freq_arr, TL, linewidth=2, linestyle="-", label='MAM')
plt.ylabel('Transmission Loss [dB]', fontsize=12)   # <- changed from plt.set_ylabel
plt.xlabel('Frequency [Hz]', fontsize=12)           # optional, recommended
plt.grid(True, which='both', linestyle='-', alpha=0.3)
plt.xlim(freq_min, freq_max)                        # <- changed from plt.set_xlim
plt.ylim(0, 80)                                
plt.show()

