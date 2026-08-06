import sys
import os
from ngsolve import *
import math
import scipy.linalg as la
import numpy as np
import scipy.sparse as sps

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from HybridWB_FEM_new.wbm_top import WBM_Top
from HybridWB_FEM_new.wbm_bottom import WBM_Bottom
from HybridWB_FEM_new.hybrid_system import HybridSystem

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
    source_func = exp(-((z - Lz* 0.4)**2)/(Lz/20)**2) / (Lz/20 * sqrt(pi))
    s_fem += source_func* q * dx("fluid")

    # Assemble 
    with TaskManager():
        Z_fem.Assemble()
        s_fem.Assemble()
    # =====================================================================
    # 3. Build WBM model and coupling matrices
    # =====================================================================
    wbm_top = WBM_Top(Lx, Ly, Lz, freq, c_air, rho_air, m_max, n_max, theta, phi)
    Z_hyb_top, Z_wbm_top, s_hyb_top, s_wbm_top = wbm_top.assemble_matrices(mesh, fes, q, "top")
    wbm_bottom = WBM_Bottom(Lx, Ly, 0, freq, c_air, rho_air, m_max, n_max, theta, phi)
    Z_hyb_bottom, Z_wbm_bottom, s_hyb_bottom, s_wbm_bottom = wbm_bottom.assemble_matrices(mesh, fes, q, "bottom")

    total_waves = wbm_top.total_waves  + wbm_bottom.total_waves
    Z_hyb = np.hstack([Z_hyb_top, Z_hyb_bottom])
    Z_wbm = la.block_diag(Z_wbm_top, Z_wbm_bottom)
    s_hyb = np.concatenate([s_hyb_top, s_hyb_bottom])
    s_wbm = np.concatenate([s_wbm_top, s_wbm_bottom])
    print("condition number of Z_wbm:", np.linalg.cond(Z_wbm))

    # =====================================================================
    # 4. Build and solve the coupled system
    # =====================================================================
    HBSys = HybridSystem(fes, total_waves)
    Global_Matrix, Global_RHS = HBSys.combine_matrices(Z_fem, Z_hyb, Z_wbm, s_hyb, s_wbm)

    return Global_Matrix, Global_RHS, HBSys.free_indices, fes, kx, ky, kz

def get_lrm_plate_matrices(freq, theta, phi, Lx, Ly, Lz, Lz_1, Lz_plate, c_air, rho_air, E_steel, nu_steel, rho_steel, mesh, curve_order, m_max, n_max):
    # =====================================================================
    # 1. Define physics and finite element space
    # =====================================================================
    k = 2 * math.pi * freq / c_air  
    omega = 2 * math.pi * freq

    # Solid (Steel Plate) Parameters
    mu_steel = E_steel / (2 * (1 + nu_steel))
    lam_steel = E_steel * nu_steel / ((1 + nu_steel) * (1 - 2 * nu_steel))
    m_reso = rho_steel * Lz_plate * Lx * Ly * 0.8
    k_reso = m_reso * (2 * math.pi * 500)**2

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
    source_func = exp(-((z - Lz* 0.4)**2)/(Lz/20)**2) / (Lz/20 * sqrt(pi))
    s_fem += source_func* q * dx("fluid")

    # Assemble 
    with TaskManager():
        Z_fem.Assemble()
        s_fem.Assemble()
    # =====================================================================
    # 3. Build WBM model and coupling matrices
    # =====================================================================
    wbm_top = WBM_Top(Lx, Ly, Lz, freq, c_air, rho_air, m_max, n_max, theta, phi)
    Z_hyb_top, Z_wbm_top, s_hyb_top, s_wbm_top = wbm_top.assemble_matrices(mesh, fes, q, "top")
    wbm_bottom = WBM_Bottom(Lx, Ly, 0, freq, c_air, rho_air, m_max, n_max, theta, phi)
    Z_hyb_bottom, Z_wbm_bottom, s_hyb_bottom, s_wbm_bottom = wbm_bottom.assemble_matrices(mesh, fes, q, "bottom")

    total_waves = wbm_top.total_waves  + wbm_bottom.total_waves
    Z_hyb = np.hstack([Z_hyb_top, Z_hyb_bottom])
    Z_wbm = la.block_diag(Z_wbm_top, Z_wbm_bottom)
    s_hyb = np.concatenate([s_hyb_top, s_hyb_bottom])
    s_wbm = np.concatenate([s_wbm_top, s_wbm_bottom])
    print("condition number of Z_wbm:", np.linalg.cond(Z_wbm))

    # =====================================================================
    # 4. Build and solve the coupled system
    # =====================================================================
    HBSys = HybridSystem(fes, total_waves)
    Global_Matrix, Global_RHS = HBSys.combine_matrices(Z_fem, Z_hyb, Z_wbm, s_hyb, s_wbm)

    # Add resonator
    center = (Lx/2, Ly/2, Lz_1 + Lz_plate)
    vertices = np.array([tuple(v.point) for v in mesh.vertices])
    dists = np.linalg.norm(vertices - center, axis=1)
    idx_center = np.argmin(dists)
    vertex = mesh.vertices[idx_center]
    dofs = fes.GetDofNrs(vertex)
    dof_z = dofs[3]

    # matrices
    ndof_f = len(HBSys.free_indices)
    Z_fem_empty = sps.coo_matrix(([], ([], [])), shape=(fes.ndof , fes.ndof)).tocsc()
    Z_fem_empty[dof_z, dof_z] = k_reso
    Z_fem_free = Z_fem_empty[HBSys.free_indices, :][:, HBSys.freedofs]
    top_row = sps.hstack([Z_fem_free, np.zeros((ndof_f, total_waves), dtype=complex)])
    bottom_row = np.hstack([np.zeros((total_waves, ndof_f), dtype=complex), np.zeros((total_waves, total_waves), dtype=complex)])
    Z_fem_free = sps.vstack([top_row, bottom_row]).tocsc()

    Z_couple = np.zeros((fes.ndof, 1), dtype=complex)
    Z_couple[dof_z, 0] = - k_reso
    Z_couple_free = Z_couple[HBSys.free_indices, :]
    Z_couple_free = np.vstack([Z_couple_free, np.zeros((total_waves, 1), dtype=complex)])

    z_reso = np.array([[k_reso - m_reso * omega**2]]) 

    # Block Matrix 
    top_row = sps.hstack([Global_Matrix + Z_fem_free, Z_couple_free])
    bottom_row = np.hstack([Z_couple_free.conj().T, z_reso])
    Global_Matrix = sps.vstack([top_row, bottom_row]).tocsc()

    # Global RHS Vector 
    Global_RHS = np.concatenate([Global_RHS, np.zeros((1), dtype=complex)])

    return Global_Matrix, Global_RHS, HBSys.free_indices, fes, kx, ky, kz
