import math
import numpy as np
from ngsolve import *
import scipy.special as sp

class WBM_Bottom_C:
    def __init__(self, R_max, z_plane, frequency, c_fluid, rho_fluid, m_max=2, n_max=2):
        
        self.R_max = R_max
        self.z_plane = z_plane
        
        self.omega = 2 * np.pi * frequency
        self.rho = rho_fluid
        self.c = c_fluid
        self.k = self.omega / self.c
        
        # Truncation limits
        self.m_indices = np.arange(1, m_max + 1)
        self.n_indices = np.arange(-n_max, n_max + 1)
        self.total_waves = 1 + len(self.m_indices) * len(self.n_indices)
        
        self.waves = []
        self.wave_k_nm = []
        self.wave_n = []
        self.wave_kz = []
        
        self._generate_wave_functions()

    def _generate_wave_functions(self):
        """ Internally generates the analytical wave functions and wavevectors. """
        self.wave_k_nm.append(0)
        self.wave_n.append(0)
        self.wave_kz.append(- self.k)
        self.waves.append(exp(- 1j * self.k * z))
        
        for m in self.m_indices:
            for n in self.n_indices:
                wavefunc, k_nm = self.circular_wave(m, n, self.R_max, x, y)
                
                val = self.k**2 - (k_nm)**2
                kz = - np.sqrt(val) if val.real >= 0 else 1j * np.sqrt(-val)
                
                self.wave_k_nm.append(k_nm)
                self.wave_n.append(n)
                self.wave_kz.append(kz)
                
                # Phase is 0 at z = z_plane
                Theta = atan2(y, x)
                phi_w = wavefunc * exp(1j * (n * Theta + kz * (z - self.z_plane)))
                self.waves.append(phi_w)

    def assemble_matrices(self, mesh, fes, test_function_v, interface_name="bottom"):
        # print(f"[WBM at bottom surface] Assembling coupling matrices for {self.total_waves} wave functions...")
        
        Z_hyb = np.zeros((fes.ndof, self.total_waves), dtype=complex)
        Z_wbm = np.zeros((self.total_waves, self.total_waves), dtype=complex)

        s_hyb = np.zeros(fes.ndof, dtype=complex)
        s_wbm = np.zeros(self.total_waves, dtype=complex)
        
        for i in range(self.total_waves):
            dphi_i_dz = 1j*self.wave_kz[i]*self.waves[i]
            cwf = LinearForm(fes)
            cwf += dphi_i_dz * test_function_v * ds(interface_name)
            with TaskManager():
                cwf.Assemble()
            Z_hyb[:, i] = cwf.vec.FV().NumPy()
            for j in range(self.total_waves):
                phi_j = self.waves[j]
                integrand = - (dphi_i_dz) * phi_j
                # Integrate symbolically over the interface boundary
                val = Integrate(integrand, mesh, definedon=mesh.Boundaries(interface_name))
                Z_wbm[i, j] = val

        return Z_hyb, Z_wbm, s_hyb, s_wbm

    def reconstruct_total_field(self, participation_factors):
        assert len(participation_factors) == self.total_waves, "Size mismatch in participation factors."
        total_field = CF(0.0)
        for i in range(self.total_waves):
            amp = complex(participation_factors[i])
            total_field += CF(amp) * self.waves[i]
        return total_field

    def circular_wave(self, m, n, R_max, X, Y):
        # 1. Use absolute value of n for SciPy roots (requires n >= 0)
        abs_n = abs(n)
        roots = sp.jnp_zeros(abs_n, m)
        k_nm = roots[-1] / R_max

        # 2. Compute spatial radius using NGSolve symbolic variables
        R = (X**2 + Y**2)**0.5
        
        # 3. Setup the argument for the Bessel function: z/2
        z_half = (k_nm * R) / 2.0
        
        # 4. Explicit Taylor series expansion for J_n(z)
        wavefunc = 0
        K_max = 20  # 20 terms is highly accurate for typical waveguide roots
        
        for k in range(K_max):
            # Calculate the scalar coefficient: (-1)^k / (k! * (k + |n|)!)
            coef = ((-1)**k) / (math.factorial(k) * math.factorial(k + abs_n))
            
            # Multiply by the symbolic variable part: (z/2)^(2k + |n|)
            wavefunc += coef * (z_half**(2 * k + abs_n))
            
        # 5. Apply symmetry relation for negative modes
        if n < 0:
            wavefunc = wavefunc * ((-1)**abs_n)
            
        return wavefunc, k_nm