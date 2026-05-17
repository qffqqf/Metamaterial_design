import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

from poly_rat import eval_rational_poly, rational_poly_interpolation

# ------------------------------------------------------------
# ------------------------------------------------------------
# Define interval and function
a_int, b_int = 0.0, 6.0
L = (b_int - a_int)     
n_samp = 40

def f_exact(x):
    return 1/(np.sin((x*(1 + 1e-3*1j)-L/3)**2) + 1e-3*1j)*(np.cos((x*(1 + 1e-3*1j)-L/3*2)**2) + 1e-3*1j)   

def generate_sobol_points(a, b, s, seed=42):
    # Next power of 2 >= s
    N_sobol = 1 << (max(1, s - 1)).bit_length()
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    points = sampler.random(n=N_sobol).flatten()  # in [0,1]
    # Shuffle and keep only s points (better uniformity than taking first s)
    rng = np.random.default_rng(seed + 1)  # separate seed for selection
    rng.shuffle(points)
    points = points[:s-2]
    points = np.concatenate(([1e-4], points, [1-1e-4]))
    # Map to [a, b]
    points = a + (b - a) * points
    points.sort()
    return points

# Generate exactly 5 sample points
x_sample = generate_sobol_points(a_int, b_int, n_samp, seed=42) 
y_sample = f_exact(x_sample)
N = (n_samp - 1) // 2

# Perform rational polynomial interpolation
a, c = rational_poly_interpolation(x_sample, y_sample, N, L)

# Dense grid for plotting
x_dense = np.linspace(a_int, b_int, 1000)
y_exact = f_exact(x_dense)
y_interp = eval_rational_poly(x_dense, a, c, L)

# Relative error (avoid division by zero)
eps = 1e-2
rel_error = np.abs(y_interp - y_exact) / (np.abs(y_exact) + eps)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Top: interpolant vs exact
ax1.semilogy(x_dense, np.abs(y_exact), 'k-', label='Exact')
ax1.semilogy(x_dense, np.abs(y_interp), 'r--', label=f'Rational Polynomial (N={N})')
ax1.plot(x_sample, np.abs(y_sample), 'b.', label='Sample points')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.set_title(f'Rational Polynomial Interpolation with {n_samp} Points')
ax1.grid(True, alpha=0.3)

# Bottom: relative error
ax2.semilogy(x_dense, rel_error, 'm-')
ax2.set_xlabel('x')
ax2.set_ylabel('Relative error')
ax2.set_title('Relative Error')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()