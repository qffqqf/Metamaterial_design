import numpy as np
import pyvista as pv
import bempp_cl.api
from bempp_cl.api.linalg import gmres

# 1. Setup the Problem and Function Spaces
k = 3.0  # Wavenumber
grid = bempp_cl.api.shapes.regular_sphere(3)
space = bempp_cl.api.function_space(grid, "DP", 0)

print("Defining boundary operators...")
# 2. Defining the Boundary Operators for CFIE (Combined Field Integral Equation)
identity = bempp_cl.api.operators.boundary.sparse.identity(space, space, space)
adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, k)
slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(space, space, space, k)

# Left hand side of the Burton-Miller / CFIE formulation
lhs = 0.5 * identity + adlp - 1j * k * slp

print("Setting up combined boundary data...")
# 3. Setting the Right Hand Side Data
@bempp_cl.api.complex_callable
def combined_data(x, n, domain_index, result):
    # Incident plane wave data: 1j * k * exp(1j*k*x) * (n_x - 1)
    result[0] = 1j * k * np.exp(1j * k * x[0]) * (n[0] - 1)

grid_fun = bempp_cl.api.GridFunction(space, fun=combined_data)

print("Solving the boundary integral equation (this may take a moment)...")
# 4. Solving the System
neumann_fun, info = gmres(lhs, grid_fun, tol=1e-5)
print(f"GMRES Solver finished with info: {info}")

##########################################################################

# 5. PyVista Visualization Grid Setup
resolution = 250
x_min, x_max = -3.0, 3.0
y_min, y_max = -3.0, 3.0

dx = (x_max - x_min) / (resolution - 1)
dy = (y_max - y_min) / (resolution - 1)

viz_grid = pv.ImageData(
    dimensions=(resolution, resolution, 1),
    spacing=(dx, dy, 1.0),
    origin=(x_min, y_min, 0.0)
)

# Extract flat list of (x, y, z) points and transpose for BEM++ (shape: 3, N)
points = viz_grid.points.T 

# 6. Masking and Exterior Evaluation
# Find radii to isolate points outside the sphere
radii = np.sqrt(points[0, :]**2 + points[1, :]**2)

# Mask for strictly exterior points (1.05 safely avoids boundary singularities)
exterior_mask = radii > 1.05
eval_points = points[:, exterior_mask]

print("Evaluating scattered field on the exterior grid...")
slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(space, eval_points, k)

# From Green's representation, u = u_inc - SLP(u_nu)
scattered_field = -1.0 * (slp_pot * neumann_fun).flatten()
incident_field = np.exp(1j * k * eval_points[0, :])
total_field = scattered_field + incident_field

# 7. Map Back to the Grid
# Create an array filled with NaNs for the whole grid
total_field_real = np.full(resolution * resolution, np.nan)

# Assign the REAL part of the computed total field to the exterior mask indices
total_field_real[exterior_mask] = np.real(total_field)

viz_grid.point_data["Total Field (Real)"] = total_field_real

# 8. Render the results
plotter = pv.Plotter(window_size=(1000, 800))
plotter.add_text("Helmholtz Exterior Scattering: Total Field (Real Part)", font_size=12)

# Render the grid. The NaN values (inside the sphere) will be colored black
plotter.add_mesh(
    viz_grid, 
    scalars="Total Field (Real)", 
    cmap="RdBu", 
    nan_color="black", 
    show_edges=False
)

plotter.camera_position = 'xy'
plotter.add_axes(line_width=2)
plotter.set_background("white")
plotter.show()