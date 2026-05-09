import numpy as np
from ngsolve import *
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la


class HybridSystem:
    
    def __init__(self, fes, total_waves):
        
        self.fes = fes
        self.total_waves = total_waves
        self.freedofs = self.fes.FreeDofs()
        self.free_indices = [i for i, is_free in enumerate(self.freedofs) if is_free]
        self.slave_indices = [i for i, is_free in enumerate(self.freedofs) if not is_free]

    def combine_matrices(self, Z_fem, s_fem, Z_hyb, Z_wbm):
        """ Combines the WBM and FEM matrices into a single system matrix. """
        print("[WBM at top and bottom surfaces] Combining WBM and FEM matrices...")
        
        # Convert sparse Z_fem to SciPy CSC format
        row, col, val = Z_fem.mat.COO()
        Z_fem_scipy = sp.coo_matrix((val, (row, col)), shape=(self.fes.ndof, self.fes.ndof)).tocsc()
        Z_fem_free = Z_fem_scipy[self.free_indices, :][:, self.free_indices]
        Z_hyb_free = Z_hyb[self.free_indices, :]

        f_f_np = s_fem.vec.FV().NumPy()
        s_fem_free = f_f_np[self.free_indices]

        # Block Matrix 
        top_row = sp.hstack([Z_fem_free, Z_hyb_free])
        bottom_row = np.hstack([Z_hyb_free.conj().T, Z_wbm])
        Global_Matrix = sp.vstack([top_row, bottom_row]).tocsc()

        # Global RHS Vector 
        s_wbm = np.zeros(self.total_waves, dtype=complex)
        Global_RHS = np.concatenate([s_fem_free, s_wbm])
        
        return Global_Matrix, Global_RHS
    
    def solve_coupled_system(self, Global_Matrix, Global_RHS):
        """ Solves the coupled FEM-WBM system using SuperLU. """
        print("[WBM at top and bottom surfaces] Solving the coupled system...")
        # Solve the dense/sparse hybrid system using SuperLU
        solution = spla.spsolve(Global_Matrix, Global_RHS)

        fem_vals = solution[:len(self.free_indices)]
        wbm_factors = solution[len(self.free_indices):]

        # Map back to an NGSolve GridFunction
        fem_gf = GridFunction(self.fes)
        fem_gf.vec.FV().NumPy()[self.free_indices] = fem_vals

        return fem_gf, wbm_factors
        
        
