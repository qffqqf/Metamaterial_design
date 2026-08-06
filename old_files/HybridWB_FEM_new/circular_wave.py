import math
import scipy.special as sp


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