import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp

# Set up the mesh in polar coordinates for a unit circle
# Using higher resolution (150) for a smoother surface
r = np.linspace(0, 1, 150)
theta = np.linspace(0, 2*np.pi, 150)
R, Theta = np.meshgrid(r, theta)
X, Y = R * np.cos(Theta), R * np.sin(Theta)

n = 0
m = 1

# 1. Calculate the wavenumber k: the m-th non-zero root of J_n'(z) = 0
# jnp_zeros(n, m) returns the first m non-zero roots.
roots = sp.jnp_zeros(n, m)
k = roots[-1]

# 2. Calculate the wave field amplitude Z: J_n(k*r) * cos(n*theta)
Z = sp.jv(n, k * R) * np.cos(n * Theta)

# --- Define the single, high-quality 3D Plot ---
# Do not use plt.subplots(). Initialize a single figure directly.
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 3. Plot the surface with high density and alpha for transparency
# Use a diverging colormap ('RdBu_r') to represent positive/negative wave phases
norm = plt.Normalize(vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)))
surf = ax.plot_surface(X, Y, Z, cmap='RdBu_r', norm=norm,
                       edgecolor='none', linewidth=0, antialiased=True, alpha=0.9)

# 4. Add visual anchor: a dashed boundary ring at the base for grounding
theta_boundary = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta_boundary), np.sin(theta_boundary), np.min(Z),
        color='black', lw=1.5, ls='--')

# 5. Optimization for viewing angle and scale
ax.view_init(elev=35, azim=45) # Classic perspective view of dipoles
ax.set_zlim(np.min(Z)*1.2, np.max(Z)*1.2) # Ensure peaks/valleys fit comfortably

# 6. Formatting and Titles
ax.set_title(f'Bessel Mode (n={n}, m={m}) in a Circular Domain\nWavenumber $k_{{11}} = {k:.3f}$\n(Condition: $\\partial u / \\partial n = 0$)', fontsize=16, pad=20)
ax.set_aspect('equal')
# 7. Add a matching colorbar
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.05, label='Wave Amplitude')

plt.tight_layout()
plt.show()