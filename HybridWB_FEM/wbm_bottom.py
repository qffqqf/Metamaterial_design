import numpy as np
from ngsolve import *

class WBM_Bottom:
    def __init__(self, L_x, L_y, z_plane, frequency, c_fluid, rho_fluid, m_max=2, n_max=2, theta=0.0, phi=0.0):
        
        self.Lx = L_x
        self.Ly = L_y
        self.z_plane = z_plane
        
        self.omega = 2 * np.pi * frequency
        self.rho = rho_fluid
        self.c = c_fluid
        self.k = self.omega / self.c
        
        # Incidence Angles
        self.theta = theta
        self.phi = phi
        self.k_x0 = self.k * np.sin(theta) * np.cos(phi)
        self.k_y0 = self.k * np.sin(theta) * np.sin(phi)
        
        # Truncation limits
        self.m_indices = np.arange(-m_max, m_max + 1)
        self.n_indices = np.arange(-n_max, n_max + 1)
        self.total_waves = len(self.m_indices) * len(self.n_indices)
        
        self.waves = []
        self.wave_kx = []
        self.wave_ky = []
        self.wave_kz = []
        
        self._generate_wave_functions()

    def _generate_wave_functions(self):
        """ Internally generates the analytical wave functions and wavevectors. """
        for m in self.m_indices:
            for n in self.n_indices:
                kx = 2 * np.pi * m / self.Lx
                ky = 2 * np.pi * n / self.Ly
                
                val = self.k**2 - (self.k_x0 + kx)**2 - (self.k_y0 + ky)**2
                kz = - np.sqrt(val) if val.real >= 0 else - 1j * np.sqrt(-val)
                
                self.wave_kx.append(kx)
                self.wave_ky.append(ky)
                self.wave_kz.append(kz)
                
                # Phase is 0 at z = z_plane
                phi_w = exp(1j * (kx * x + ky * y + kz * (z - self.z_plane)))
                self.waves.append(phi_w)

    def assemble_matrices(self, mesh, fes, test_function_v, interface_name="bottom"):
        print(f"[WBM at bottom surface] Assembling coupling matrices for {self.total_waves} wave functions...")
        
        Z_hyb = np.zeros((fes.ndof, self.total_waves), dtype=complex)
        Z_wbm = np.zeros((self.total_waves, self.total_waves), dtype=complex)
        
        for i in range(self.total_waves):
            dphi_i_dz = 1j*self.wave_kz[i]*self.waves[i]
            cwf = LinearForm(fes)
            cwf += dphi_i_dz * test_function_v * ds(interface_name)
            with TaskManager():
                cwf.Assemble()
            Z_hyb[:, i] = cwf.vec.FV().NumPy()
            for j in range(self.total_waves):
                phi_j = self.waves[j]
                integrand = - Conj(dphi_i_dz) * phi_j
                # Integrate symbolically over the interface boundary
                val = Integrate(integrand, mesh, definedon=mesh.Boundaries(interface_name))
                Z_wbm[i, j] = val

        return Z_hyb, Z_wbm

    def reconstruct_total_field(self, participation_factors):
        assert len(participation_factors) == self.total_waves, "Size mismatch in participation factors."
        total_field = CF(0.0)
        for i in range(self.total_waves):
            amp = complex(participation_factors[i])
            total_field += CF(amp) * self.waves[i]
        return total_field