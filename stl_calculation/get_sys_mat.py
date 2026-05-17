import sys
import os
from ngsolve import *
import math
import scipy.linalg as la
import numpy as np

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from HybridWB_FEM.wbm_top import WBM_Top
from HybridWB_FEM.wbm_bottom import WBM_Bottom
from HybridWB_FEM.hybrid_system import HybridSystem

def get_bare_plate_matrices(freq, theta, phi, Lx, Ly, Lz, c_air, rho_air, E_steel, nu_steel, rho_steel, mesh, curve_order, m_max, n_max):
    # =====================================================================
    # 1. Define physics and finite element space
    # =====================================================================
    k = 2 * math.pi * freq / c_air  
    omega = 2 * math.pi * freq

    # Solid (Steel Plate) Parameters
    mu_steel = E_steel / (2 * (1 + nu_steel))
    lam_steel = E_steel * nu_steel / ((1 + nu_steel) * (1 - 2 * nu_steel))

    # Incident angles (e.g., 45 degrees)
    kx = k * math.sin(theta) * math.cos(phi)
    ky = k * math.sin(theta) * math.sin(phi)
    kz = k * math.cos(theta)
    k_vec = CF((kx, ky, 0)) 

    # Finite Element Space
    fes_air = Periodic(H1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("fluid")))
    fes_plate = Periodic(VectorH1(mesh, order=curve_order, complex=True, definedon=mesh.Materials("solid")))  
    fes = (fes_air * fes_plate)
    (p, u), (q, w) = fes.TnT()  # p,q = acoustic ; u,w = elastic

    # =====================================================================
    # 2. Define variational forms
    # =====================================================================
    # Differential Operators 
    def grad_p(p): return grad(p) + 1j * k_vec * p
    def grad_q(q): return grad(q) - 1j * k_vec * q
    def eps_u(u): return Sym(Grad(u) + 1j * OuterProduct(u, k_vec))
    def eps_w(w): return Sym(Grad(w) - 1j * OuterProduct(w, k_vec))
    def stress(u): return 2*mu_steel*eps_u(u) + lam_steel*Trace(eps_u(u))*Id(3)

    # Uncoupled Bilinear Form (FEM)
    Z_fem = BilinearForm(fes)
    Z_fem += (grad_p(p) * grad_q(q) - k**2 * p * q) * dx("fluid")
    Z_fem += (InnerProduct(stress(u), eps_w(w)) - rho_steel * omega**2 * InnerProduct(u, w)) * dx("solid")

    # FSI Interface Coupling 
    n_plate_to_air = specialcf.normal(mesh.dim)
    Z_fem += rho_air * omega**2 * (n_plate_to_air * u)* q * ds("fsi_interface")    
    Z_fem += p * (n_plate_to_air *w) * ds("fsi_interface")                      

    # Linear Form (RHS) - Surface source on the bottom boundary
    s_fem = LinearForm(fes)
    source_func = exp(-((z-1*Lz/4)**2)/(Lz/20)**2) / (Lz/20 * sqrt(pi))
    s_fem += source_func* q * dx("fluid")

    # Assemble 
    with TaskManager():
        Z_fem.Assemble()
        s_fem.Assemble()
    # =====================================================================
    # 3. Build WBM model and coupling matrices
    # =====================================================================
    wbm_top = WBM_Top(Lx, Ly, Lz, freq, c_air, rho_air, m_max, n_max, theta, phi)
    Z_hyb_top, Z_wbm_top = wbm_top.assemble_matrices(mesh, fes, q, "top")
    wbm_bottom = WBM_Bottom(Lx, Ly, 0, freq, c_air, rho_air, m_max, n_max, theta, phi)
    Z_hyb_bottom, Z_wbm_bottom = wbm_bottom.assemble_matrices(mesh, fes, q, "bottom")

    total_waves = wbm_top.total_waves  + wbm_bottom.total_waves
    Z_hyb = np.hstack([Z_hyb_top, Z_hyb_bottom])
    Z_wbm = la.block_diag(Z_wbm_top, Z_wbm_bottom)
    print("condition number of Z_wbm:", np.linalg.cond(Z_wbm))

    # =====================================================================
    # 4. Build and solve the coupled system
    # =====================================================================
    HBSys = HybridSystem(fes, total_waves)
    Global_Matrix, Global_RHS = HBSys.combine_matrices(Z_fem, s_fem, Z_hyb, Z_wbm)

    return Global_Matrix, Global_RHS, HBSys.free_indices, fes, kx, ky, kz
