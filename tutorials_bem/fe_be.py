import numpy as np
import scipy.sparse
from scipy.sparse.linalg import gmres, LinearOperator

import dolfinx
from dolfinx.fem import functionspace, Function, form, petsc
from dolfinx.mesh import create_unit_cube
from dolfinx import geometry
import ufl

from mpi4py import MPI
import bempp_cl.api
from bempp_cl.api.external import fenicsx
import pyvista as pv

# =============================================================================
# 1. SETUP PARAMETERS & FUNCTION SPACES
# =============================================================================
k = 6.0
n_refractive = 0.5
d = np.array([1.0, 1.0, 1.0])
d /= np.linalg.norm(d)

print("Creating FEniCSx Mesh and Spaces...")
# Create a unit cube [0,1]x[0,1]x[0,1]
mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

# FEniCSx FEM space
fenics_space = functionspace(mesh, ("CG", 1))

# Bempp-cl BEM trace spaces
trace_space, trace_matrix = fenicsx.fenics_to_bempp_trace_data(fenics_space)
bempp_space = bempp_cl.api.function_space(trace_space.grid, "DP", 0)

fem_size = fenics_space.dofmap.index_map.size_global
bem_size = bempp_space.global_dof_count

print(f"FEM dofs: {fem_size}")
print(f"BEM dofs: {bem_size}")

# =============================================================================
# 2. DEFINING BEM & FEM OPERATORS
# =============================================================================
print("Constructing Boundary Operators...")
id_op = bempp_cl.api.operators.boundary.sparse.identity(trace_space, bempp_space, bempp_space)
mass = bempp_cl.api.operators.boundary.sparse.identity(bempp_space, bempp_space, trace_space)
dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(trace_space, bempp_space, bempp_space, k)
slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(bempp_space, bempp_space, bempp_space, k)

print("Assembling FEM Operators...")
u = ufl.TrialFunction(fenics_space)
v = ufl.TestFunction(fenics_space)

# FEM volume form: A - k^2 M
fem_form = form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - (k**2) * (n_refractive**2) * u * v * ufl.dx)
A_fem_petsc = petsc.assemble_matrix(fem_form)
A_fem_petsc.assemble()

# Convert PETSc matrix to SciPy sparse matrix for the blocked system
ai, aj, av = A_fem_petsc.getValuesCSR()
A_fem = scipy.sparse.csr_matrix((av, aj, ai), shape=(fem_size, fem_size))

# =============================================================================
# 3. CONSTRUCTING THE BLOCKED SYSTEM & RHS
# =============================================================================
@bempp_cl.api.complex_callable
def u_inc_callable(x, n, domain_index, result):
    result[0] = np.exp(1j * k * np.dot(x, d))

u_inc = bempp_cl.api.GridFunction(bempp_space, fun=u_inc_callable)

# Form RHS
rhs_fem = np.zeros(fem_size, dtype=np.complex128)
rhs_bem = u_inc.projections(bempp_space)
rhs = np.concatenate([rhs_fem, rhs_bem])

# Precompute weak form matrices for BEM
mass_mat = mass.weak_form().A
slp_mat = slp.weak_form().A
dlp_mat = dlp.weak_form().A
id_mat = id_op.weak_form().A

# BEM Matrix Term: 0.5 * Id - K (Double Layer)
half_id_minus_dlp = 0.5 * id_mat - dlp_mat

def block_mat_vec(x):
    """ Matrix-vector product for the coupled FEM-BEM blocked system """
    u_vec = x[:fem_size]
    lambda_vec = x[fem_size:]
    
    # Row 1 (FEM): (A - k^2 M)*u - M_Gamma*lambda
    # trace_matrix maps FEM nodes to the boundary trace
    fem_row = A_fem.dot(u_vec) - trace_matrix.T.dot(mass_mat.dot(lambda_vec))
    
    # Row 2 (BEM): (0.5*Id - K)*trace(u) + V*lambda
    trace_u = trace_matrix.dot(u_vec)
    bem_row = half_id_minus_dlp.dot(trace_u) + slp_mat.dot(lambda_vec)
    
    # Use np.ravel() to strictly enforce 1D standard ndarrays
    return np.concatenate([np.ravel(fem_row), np.ravel(bem_row)])

system_op = LinearOperator((fem_size + bem_size, fem_size + bem_size), 
                           matvec=block_mat_vec, dtype=np.complex128)

# =============================================================================
# 4. SOLVING THE SYSTEM
# =============================================================================
print("Solving the coupled system via GMRES...")
sol, info = gmres(system_op, rhs, rtol=1e-5)
print(f"GMRES finished with info code: {info}")

# Extract solutions
u_vec = sol[:fem_size]
lambda_vec = sol[fem_size:]

# Wrap FEM solution in FEniCSx Function
u_fem = Function(fenics_space)
u_fem.x.array[:] = u_vec

# Wrap BEM solutions in Bempp GridFunctions
dirichlet_data = trace_matrix.dot(u_vec)
dirichlet_trace = bempp_cl.api.GridFunction(trace_space, coefficients=dirichlet_data)
neumann_trace = bempp_cl.api.GridFunction(bempp_space, coefficients=lambda_vec)


# =============================================================================
# 5. EVALUATION AND PYVISTA VISUALIZATION
# =============================================================================
print("Setting up PyVista visualization grid...")
resolution = 250
# Window coordinates expanded slightly around the [0, 1] unit cube
x_min, x_max = -1.5, 2.5
y_min, y_max = -1.5, 2.5
z_val = 0.5  # Slice directly through the middle of the cube

dx = (x_max - x_min) / (resolution - 1)
dy = (y_max - y_min) / (resolution - 1)

viz_grid = pv.ImageData(
    dimensions=(resolution, resolution, 1),
    spacing=(dx, dy, 1.0),
    origin=(x_min, y_min, z_val)
)

points = viz_grid.points.T # Shape: (3, N)
x_c, y_c, z_c = points[0, :], points[1, :], points[2, :]

# Masking logic for the Unit Cube [0, 1]^3
margin = 0.02 # Buffer to avoid evaluating perfectly on the boundary layer
interior_mask = (x_c > margin) & (x_c < 1 - margin) & \
                (y_c > margin) & (y_c < 1 - margin) & \
                (z_c > margin) & (z_c < 1 - margin)

exterior_mask = (x_c < -margin) | (x_c > 1 + margin) | \
                (y_c < -margin) | (y_c > 1 + margin) | \
                (z_c < -margin) | (z_c > 1 + margin)

# --- Evaluate Exterior Field (BEM) ---
print("Evaluating scattered field on the exterior grid...")
ext_points = points[:, exterior_mask]

slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(
    neumann_trace.space, ext_points, k
)
dlp_pot = bempp_cl.api.operators.potential.helmholtz.double_layer(
    dirichlet_trace.space, ext_points, k
)

scattered_field = (slp_pot * neumann_trace).flatten() - (dlp_pot * dirichlet_trace).flatten()
incident_field = np.exp(1j * k * np.dot(d, ext_points))
total_exterior_field = scattered_field + incident_field

# --- Evaluate Interior Field (FEM) ---
print("Evaluating FEM field on the interior grid...")
int_points = points[:, interior_mask].T  # Shape: (N_int, 3) for dolfinx

bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, int_points)
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, int_points)

cells = []
points_on_proc = []
valid_indices = []

for i, p in enumerate(int_points):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(p)
        cells.append(colliding_cells.links(i)[0])
        valid_indices.append(i)

total_interior_field = np.zeros(len(int_points), dtype=np.complex128)
if len(points_on_proc) > 0:
    u_evaluated = u_fem.eval(points_on_proc, cells)
    total_interior_field[valid_indices] = u_evaluated.flatten()

# --- Merge and Render ---
print("Mapping data to PyVista and rendering...")
total_field_real = np.full(resolution * resolution, np.nan)

total_field_real[exterior_mask] = np.real(total_exterior_field)
total_field_real[interior_mask] = np.real(total_interior_field)

viz_grid.point_data["Total Field (Real)"] = total_field_real

plotter = pv.Plotter(window_size=(1000, 800))
plotter.add_text("FEM-BEM Helmholtz Coupling: Total Field (Real Part)", font_size=12)

# NaN values act as a boundary outline where masking margins omitted points
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