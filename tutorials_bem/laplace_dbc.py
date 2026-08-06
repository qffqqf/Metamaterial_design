import numpy as np
import pyvista as pv
import bempp_cl.api

# 1. Meshing and Function Spaces
grid = bempp_cl.api.shapes.sphere(h=0.1)
dp0_space = bempp_cl.api.function_space(grid, "DP", 0)
p1_space = bempp_cl.api.function_space(grid, "P", 1)

print("Defining boundary operators...")
# 2. Defining the Boundary Operators
identity = bempp_cl.api.operators.boundary.sparse.identity(p1_space, p1_space, dp0_space)
dlp = bempp_cl.api.operators.boundary.laplace.double_layer(p1_space, p1_space, dp0_space)
slp = bempp_cl.api.operators.boundary.laplace.single_layer(dp0_space, p1_space, dp0_space)

print("Setting up Dirichlet boundary data...")
# 3. Setting the Dirichlet Data
@bempp_cl.api.real_callable
def dirichlet_data(x, n, domain_index, result):
    # Point source at (0.8, 0, 0)
    result[0] = x[0]**2 - x[1]**2

dirichlet_fun = bempp_cl.api.GridFunction(p1_space, fun=dirichlet_data)

print("Solving the boundary integral equation (this may take a moment)...")
# 4. Solving the Boundary Integral Equation
rhs = (0.5 * identity + dlp) * dirichlet_fun
neumann_fun, info = bempp_cl.api.linalg.cg(slp, rhs, tol=1e-3)
print(f"CG Solver finished with info: {info}")

##########################################################################

resolution = 200
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0

# 1. Let PyVista create a perfectly oriented 2D grid directly
# Dimensions are (Nx, Ny, Nz). We want a flat plane in Z, so Nz = 1
dx = (x_max - x_min) / (resolution - 1)
dy = (y_max - y_min) / (resolution - 1)

# FIX: Changed dimensions from (res, 1, res) to (res, res, 1)
grid = pv.ImageData(
    dimensions=(resolution, resolution, 1),
    spacing=(dx, dy, 1.0),       # 1.0 is a dummy spacer for the z-axis
    origin=(x_min, y_min, 0.0)   # Center the plane at z = 0
)

# 2. PyVista exposes a flat list of (x, y, z) points
# FIX: Transpose the array so BEM++ gets shape (3, N) instead of (N, 3)
points = grid.points.T 

print("Evaluating pressure field on the new grid...")
slp_pot = bempp_cl.api.operators.potential.laplace.single_layer(dp0_space, points)
dlp_pot = bempp_cl.api.operators.potential.laplace.double_layer(p1_space, points)

# BEM++ returns an array of shape (1, N)
pressure_values = slp_pot * neumann_fun - dlp_pot * dirichlet_fun
 
# 3. Calculate Analytical Solution and Error
# Source location used in your boundary conditions
source_pos = np.array([[0.8], [0.0], [0.0]])

# Calculate distance from source to every point in the (3, N) array
distances = np.linalg.norm(points - source_pos, axis=0)

# Exact equation: 1 / (4 * pi * r)
analytical_values = points[0, :]**2 - points[1, :]**2

# Calculate absolute error
numerical_values = pressure_values.flatten()
error_values = np.abs(numerical_values - analytical_values)

# Assign arrays to the grid
grid.point_data["Numerical"] = numerical_values
grid.point_data["Analytical"] = analytical_values
grid.point_data["Absolute Error"] = error_values

# 4. Render the results side-by-side
# Create a 1x3 subplot layout
plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

# Plot 1: Numerical Solution
plotter.subplot(0, 0)
plotter.add_text("Numerical Solution", font_size=12)
plotter.add_mesh(grid, scalars="Numerical", cmap="coolwarm", show_edges=False)
plotter.camera_position = 'xy'
plotter.add_axes(line_width=2)

# Plot 2: Analytical Solution
plotter.subplot(0, 1)
plotter.add_text("Analytical Solution", font_size=12)
plotter.add_mesh(grid, scalars="Analytical", cmap="coolwarm", show_edges=False)
plotter.camera_position = 'xy'
plotter.add_axes(line_width=2)

# Plot 3: Absolute Error
plotter.subplot(0, 2)
plotter.add_text("Absolute Error", font_size=12)
plotter.add_mesh(grid, scalars="Absolute Error", cmap="viridis", show_edges=False)
plotter.camera_position = 'xy'
plotter.add_axes(line_width=2)

# Formatting
plotter.set_background("white")
plotter.show()