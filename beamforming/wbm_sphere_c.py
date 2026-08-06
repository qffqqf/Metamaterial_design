import numpy as np
from ngsolve import *
from scipy.special import spherical_jn, spherical_yn, lpmv
import math

class WBM_sphere:
    def __init__(self, mesh, Radius, frequency, c_fluid, rho_fluid, pts_source, m_max=2):
        
        self.mesh = mesh
        self.Radius = Radius
        self.pts_source = pts_source

        self.omega = 2 * np.pi * frequency
        self.rho = rho_fluid
        self.c = c_fluid
        self.k = self.omega / self.c
        
        # Truncation limits
        self.m_indices = np.arange(1, m_max + 1)
        self.total_waves = len(self.m_indices + 1) **2
        
        self.waves = []
        self.waves_dev = []
        self.wave_m = []
        self.wave_n = []
        
        self._generate_wave_functions()

    def _generate_wave_functions(self):
        # Internally generates the analytical wave functions and wavevectors. 
        for m in self.m_indices:
            for n in np.arange(-m, m + 1):
                self.wave_m.append(m)
                self.wave_n.append(n)
                phi_w = self.evaluate_multipole(m, n)
                self.waves.append(phi_w)
                phi_w_dev = self.multipole_directional_derivative(m, n)
                self.waves_dev.append(phi_w_dev)

        dR = sqrt(((self.pts_source[0] - x)**2 + (self.pts_source[1] - y)**2 + (self.pts_source[2] - z)**2))
        self.wave_par = exp(1j * self.k * dR) / dR

    def assemble_matrices(self, fes, test_function_v, interface_name="Ellipsoid_Boundary"):
        # print(f"[WBM at top surface] Assembling coupling matrices for {self.total_waves} wave functions...")
        
        Z_hyb = np.zeros((fes.ndof, self.total_waves), dtype=complex)
        Z_wbm = np.zeros((self.total_waves, self.total_waves), dtype=complex)
        s_wbm = np.zeros(self.total_waves, dtype=complex)

        hyb_rhs = LinearForm(fes)
        hyb_rhs += self.wave_par * test_function_v * ds(interface_name)
        hyb_rhs.Assemble()
        s_hyb = hyb_rhs.vec.FV().NumPy()

        for i in range(self.total_waves):
            dphi_i_dz = self.waves_dev[i]
            cwf = LinearForm(fes)
            cwf += - dphi_i_dz * test_function_v * ds(interface_name)
            s_wbm[i] = Integrate(- Conj(dphi_i_dz) * self.wave_par, self.mesh, definedon=self.mesh.Boundaries(interface_name))
            with TaskManager():
                cwf.Assemble()
            Z_hyb[:, i] = cwf.vec.FV().NumPy()
            for j in range(self.total_waves):
                phi_j = self.waves[j]
                integrand = Conj(dphi_i_dz) * phi_j
                # Integrate symbolically over the interface boundary
                val = Integrate(integrand, self.mesh, definedon=self.mesh.Boundaries(interface_name))
                Z_wbm[i, j] = val

        return Z_hyb, Z_wbm, s_hyb, s_wbm

    def multipole_directional_derivative(self, u, v):
        """
        Evaluates the directional derivative of the multipole wave function.
        
        Parameters:
        u       : int, order of the function
        v       : int, degree of the function
        k       : float or complex, wave number
        normals : (N, 3) numpy array of Cartesian normal vectors [nx, ny, nz]
        
        Returns:
        (N,) numpy array of complex derivative values.
        """
        # Ensure inputs are 2D arrays to support both single points and vectorized meshes
        normals = specialcf.normal(self.mesh.dim)
        nx, ny, nz = normals[0], normals[1], normals[2]
        
        # 1. Cartesian to Spherical Coordinates
        r = sqrt(x**2 + y**2 + z**2)
        r_safe = IfPos(r - 1e-12, r, 1e-12)
        theta = acos(z / r_safe)
        phi = atan2(y, x)
        
        # 2. Transform Normal Vector to Spherical Basis
        nr = nx * sin(theta) * cos(phi) + ny * sin(theta) * sin(phi) + nz * cos(theta)
        ntheta = nx * cos(theta) * cos(phi) + ny * cos(theta) * sin(phi) - nz * sin(theta)
        nphi = -nx * sin(phi) + ny * cos(phi)
        
        # 3. Radial Function and its Derivative w.r.t 'r'
        kr = self.k * r
        h_val = self.spherical_hn2(u, kr)
        h_prime = self.k * self.spherical_hn2(u, kr, derivative=True) # Chain rule: d/dr = k * d/d(kr)
        
        # 4. Spherical Harmonics & Angular Derivatives
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        
        # Prevent division by zero near the poles (numerical stabilization)
        sin_theta_safe = IfPos(sin_theta**2 - 1e-14, sin_theta, 1e-14)
        
        # Evaluate P_u^v. Note: scipy.special.lpmv signature is (order v, degree u, arg)
        # Evaluate P_u^v using the native NGSolve implementation
        P_u_v = self.ng_lpmv(v, u, cos_theta)
        P_u_minus_1_v = self.ng_lpmv(v, u-1, cos_theta) if u > 0 else cos_theta * 0.0
        
        # Normalization constant N_u^v from Equation 14
        N_uv = sqrt((2*u + 1) / (4 * np.pi) * math.factorial(u - v) / math.factorial(u + v))
        
        # Base spherical harmonic
        exp_phi = exp(1j * v * phi)
        Y_val = N_uv * P_u_v * exp_phi
        
        # Derivatives w.r.t theta and phi
        dY_dtheta = N_uv * ((v * cos_theta * P_u_v - (u + v) * P_u_minus_1_v) / sin_theta_safe) * exp_phi
        dY_dphi = 1j * v * Y_val
        
        # 5. Assemble Gradients natively in Spherical Coordinates
        grad_r = h_prime * Y_val
        grad_theta = (1.0 / r) * h_val * dY_dtheta
        grad_phi = (1.0 / (r * sin_theta_safe)) * h_val * dY_dphi
        
        # 6. Directional Derivative (Dot Product)
        dir_deriv = (grad_r * nr + grad_theta * ntheta + grad_phi * nphi) / self.spherical_hn2(u, self.Radius)
        
        return dir_deriv

    def evaluate_multipole(self, u, v):
        """
        Evaluates the multipole wave function Phi_{uv}.
        
        Parameters:
        u       : int, order of the function
        v       : int, degree of the function
        k       : float or complex, wave number

        Returns:
        (N,) numpy array of complex wave function values.
        """
        
        # 1. Cartesian to Spherical Coordinates
        r = sqrt(x**2 + y**2 + z**2)
        
        # Avoid division by zero at the origin
        r_safe = IfPos(r - 1e-12, r, 1e-12)

        theta = acos(z / r_safe)
        phi = atan2(y, x)
        
        # 2. Radial Function: Spherical Hankel of the 2nd kind
        kr = self.k * r
        h_val = self.spherical_hn2(u, kr)
        
        # 3. Angular Function: Spherical Harmonics
        cos_theta = cos(theta)
        
        # Evaluate Associated Legendre Polynomial P_u^v
        # Note: scipy.special.lpmv signature is (order v, degree u, arg)
        P_u_v = self.ng_lpmv(v, u, cos_theta)
        
        # Normalization constant N_u^v 
        N_uv = sqrt((2*u + 1) / (4 * np.pi) * math.factorial(u - v) / math.factorial(u + v))
        
        # Base spherical harmonic Y_u^v
        exp_phi = exp(1j * v * phi)
        Y_val = N_uv * P_u_v * exp_phi
        
        # 4. Assemble the full wave function
        phi_uv = h_val * Y_val / self.spherical_hn2(u, self.Radius)
        
        return phi_uv
    
    def ng_spherical_jn(self, n, z):
        """Native NGSolve implementation of spherical Bessel function of the 1st kind."""
        if n == 0: return sin(z)/z
        if n == 1: return sin(z)/(z**2) - cos(z)/z
        
        j_prev2 = sin(z)/z
        j_prev1 = sin(z)/(z**2) - cos(z)/z
        for i in range(1, n):
            j_curr = (2*i + 1)/z * j_prev1 - j_prev2
            j_prev2 = j_prev1
            j_prev1 = j_curr
        return j_prev1

    def ng_spherical_yn(self, n, z):
        """Native NGSolve implementation of spherical Bessel function of the 2nd kind."""
        if n == 0: return -cos(z)/z
        if n == 1: return -cos(z)/(z**2) - sin(z)/z
        
        y_prev2 = -cos(z)/z
        y_prev1 = -cos(z)/(z**2) - sin(z)/z
        for i in range(1, n):
            y_curr = (2*i + 1)/z * y_prev1 - y_prev2
            y_prev2 = y_prev1
            y_prev1 = y_curr
        return y_prev1

    def ng_lpmv(self, m, n, x):
        """
        Native NGSolve implementation of Associated Legendre Polynomial P_n^m(x).
        Scipy's lpmv signature is (order m, degree n, arg x).
        """
        m_abs = abs(m)
        if m_abs > n:
            return x * 0.0 # Returns an NGSolve CF of 0
            
        # 1. Compute P_m^m(x)
        if m_abs == 0:
            pmm = x**0  # Evaluates to CF 1.0
        else:
            double_fact = math.prod(range(1, 2*m_abs, 2))
            pmm = double_fact * ( (1 - x**2)**(m_abs / 2.0) )
            
        # 2. Ascend to P_n^m(x)
        if n == m_abs:
            p_val = pmm
        elif n == m_abs + 1:
            p_val = x * (2 * m_abs + 1) * pmm
        else:
            p_prev2 = pmm
            p_prev1 = x * (2 * m_abs + 1) * pmm
            for k in range(m_abs + 2, n + 1):
                p_curr = (x * (2*k - 1) * p_prev1 - (k + m_abs - 1) * p_prev2) / (k - m_abs)
                p_prev2 = p_prev1
                p_prev1 = p_curr
            p_val = p_prev1
            
        # 3. Handle negative order (Scipy lpmv convention)
        if m < 0:
            sign = (-1)**m_abs
            scale = sign * math.factorial(n - m_abs) / math.factorial(n + m_abs)
            p_val = scale * p_val
            
        return p_val
    
    def spherical_hn2(self, n, z, derivative=False):
        # NOTE: You had a 'derivative' kwarg used in multipole_directional_derivative
        # which wasn't defined in your original spherical_hn2 signature. 
        # I added it here for completeness.
        
        h_val = self.ng_spherical_jn(n, z) - 1j * self.ng_spherical_yn(n, z)
        
        if derivative:
            # Derivative of spherical Hankel using standard recurrence:
            # f'_n(z) = f_{n-1}(z) - (n+1)/z * f_n(z)
            if n == 0:
                h_prev = self.ng_spherical_jn(1, z) - 1j * self.ng_spherical_yn(1, z)
                return -h_prev # h'_0 = -h_1
            else:
                h_prev = self.ng_spherical_jn(n-1, z) - 1j * self.ng_spherical_yn(n-1, z)
                return h_prev - (n + 1)/z * h_val
        return h_val