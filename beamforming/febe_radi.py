import numpy as np
from ngsolve import *
from ngsolve import bem  # Use the built-in BEM module
import netgen.read_gmsh
import pyvista as pv
from netgen.read_gmsh import ReadGmsh

mesh_filepath = "./tutorials/mesh_files/t10.msh"
# --- 1. Load the Mesh ---
print("Loading mesh into Netgen...")
ngmesh = ReadGmsh(mesh_filepath)
mesh = Mesh(ngmesh)

# --- 2. Define Acoustic Parameters ---
c_0 = 343.0
freq = 1200.0
omega = 2 * np.pi * freq
k = omega / c_0
rho_air = 1.225

pts_source = np.array([0.6, 0.0, 0.0])

# --- 3. Define the FE and BE Spaces ---
fes_p = H1(mesh, order=5, complex=True)
# SurfaceL2 is the standard BEM space for the boundary
fes_lam = SurfaceL2(mesh, order=4, complex=True, definedon=mesh.Boundaries("Ellipsoid_Boundary"))

p, q = fes_p.TnT()
lam, mu = fes_lam.TnT()

# --- 4. Weak Formulation & Operators ---
# FEM volume matrix
a_fem = BilinearForm(fes_p)
a_fem += (grad(p)*grad(q) - k**2 * p * q) * dx("Fluid")
a_fem.Assemble()

# Coupling Mass Matrices
M_lu = BilinearForm(trialspace=fes_p, testspace=fes_lam)
M_lu += p * mu * ds("Ellipsoid_Boundary")
M_lu.Assemble()

M_ul = BilinearForm(trialspace=fes_lam, testspace=fes_p)
M_ul += lam * q * ds("Ellipsoid_Boundary")
M_ul.Assemble()

# Use 'kappa=k' so the float is passed to the correct parameter
V = bem.HelmholtzSingleLayerPotentialOperator(fes_lam, kappa=k)

# Ensure you also update K to use 'kappa' instead of 'k'
K = bem.HelmholtzDoubleLayerPotentialOperator(trial_space=fes_p, test_space=fes_lam, kappa=k)

# --- 5. Assemble Block Matrix ---
# Costabel formulation:
# [ A_fem,         -M_ul ] [ p   ] = [ 0 ]
# [ 0.5*M_lu + K,  -V    ] [ lam ] = [ f_lam ]
Global_Matrix = BlockMatrix([
    [a_fem.mat, -M_ul.mat],
    [0.5 * M_lu.mat + K.mat, -V.mat]
])

# --- 6. Incident Wave & RHS ---
xs, ys, zs = pts_source
# Defining incident field as CoefficientFunction
r = sqrt((x-xs)**2 + (y-ys)**2 + (z-zs)**2)
p_inc = exp(-1j * k * r) / (4 * np.pi * r)

# Normal derivative
n = specialcf.normal(3)
dr = CF(((x-xs)/r, (y-ys)/r, (z-zs)/r))
dp_inc_dr = exp(-1j * k * r) / (4 * np.pi) * (-1j * k / r - 1 / r**2)
dp_inc_dn = dp_inc_dr * InnerProduct(dr, n)

# Prepare RHS
gfu_p_inc = GridFunction(fes_p)
gfu_p_inc.Set(p_inc, definedon=mesh.Boundaries("Ellipsoid_Boundary"))

gfu_dp_inc = GridFunction(fes_lam)
gfu_dp_inc.Set(dp_inc_dn, definedon=mesh.Boundaries("Ellipsoid_Boundary"))

# Safely assemble f_lam
f_lam = (0.5 * M_lu.mat + K.mat) * gfu_p_inc.vec - V.mat * gfu_dp_inc.vec

# FIX 1: Properly initialize an empty NGSolve vector for f_p
f_p = LinearForm(fes_p).Assemble()
Global_RHS = BlockVector([f_p.vec, f_lam])

# --- 7. Solve ---
gfu_p = GridFunction(fes_p)
gfu_lam = GridFunction(fes_lam)
sol = BlockVector([gfu_p.vec, gfu_lam.vec])

print("Solving using GMRes...")

# FIX 2: Actually import and use the GMRes iterative solver
from ngsolve.krylovspace import GMRes

# Use GMRes (Costabel coupling may take several iterations to converge)
sol.data = GMRes(A=Global_Matrix, b=Global_RHS, pre=None, tol=1e-8, maxsteps=2000, printrates=True)

print("System solved.")

# --- 8. Visualization (remains the same) ---
# ... [Insert your existing PyVista code here] ...
# --- 7. Post-Processing and Visualization ---
print("\n--- Plotting in PyVista (High-Resolution Grid) ---")

resolution = 200
x_min, x_max = -1.0, 1.0  # Widened sightly to see the point source well
z_min, z_max = -1.0, 1.0

dx_step = (x_max - x_min) / (resolution - 1)
dz_step = (z_max - z_min) / (resolution - 1)

grid = pv.ImageData(
    dimensions=(resolution, 1, resolution),
    spacing=(dx_step, 1.0, dz_step),
    origin=(x_min, 0.0, z_min)
)

points = grid.points
pressure_values = np.zeros(grid.n_points)

print("Evaluating pressure field on the new grid...")
for i, (x_val, y_val, z_val) in enumerate(points):
    try:
        pressure_values[i] = gfu_p(mesh(x_val, y_val, z_val)).real
    except Exception:
        pressure_values[i] = np.nan

grid.point_data["Real Pressure"] = pressure_values

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="Real Pressure", cmap="coolwarm", show_edges=False)
plotter.set_background("white")
plotter.camera_position = 'xz'
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()