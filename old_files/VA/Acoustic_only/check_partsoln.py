import numpy as np
import pyvista as pv

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
r_tube = 61.25e-3 # tube
h_tube = 600e-3
rho_air = 1.21
c_air = 343.0 # speed of sound in air
h_plate = 140e-3 # plate position
freq = 1000.0
omega = 2 * np.pi * freq
k = 2 * np.pi * freq / c_air
zs = h_tube*0.4

nr, ntheta, nz = 20, 40, 60
r = np.linspace(0.0, r_tube, nr)
theta = np.linspace(0, 2 * np.pi, ntheta)
z = np.linspace(0, h_tube, nz)

R, Theta, Z = np.meshgrid(r, theta, z, indexing='ij')
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
grid = pv.StructuredGrid(X, Y, Z)

max_amplitude = 1.0
initial_wave = np.sin(k * np.abs(Z - zs))
grid.point_data["Wave Amplitude"] = initial_wave.flatten(order='F')

plotter = pv.Plotter()
# Add the grid mesh
plotter.add_mesh(
    grid, 
    cmap='coolwarm', 
    show_scalar_bar=True, 
    clim=[-max_amplitude, max_amplitude],  # CRITICAL: Keep color limits locked
    opacity=1.0
)
plotter.set_background('white')
plotter.view_isometric()
plotter.add_axes()
# plotter.show()

print("Rendering animation frames...")
plotter.open_gif("VA/Acoustic_only/particular_solution.gif", fps=24)
n_frames = 60    
time_steps = np.linspace(0, 1 / freq, n_frames, endpoint=False)
for t in time_steps:
    wave_real = np.sin(k * np.abs(Z - zs) - omega * t)
    grid.point_data["Wave Amplitude"] = wave_real.flatten(order='F')
    plotter.write_frame()
plotter.close()
print("Success! Animation saved as 'VA/Acoustic_only/particular_solution.gif'")
