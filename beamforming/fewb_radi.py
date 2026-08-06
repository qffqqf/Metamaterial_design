import numpy as np
from ngsolve import *
import netgen.read_gmsh
import meshio
import pyvista as pv
import netgen.gui 
from netgen.read_gmsh import ReadGmsh
from wbm_sphere_c import WBM_sphere
from hybrid_sys import HybridSystem

mesh_filepath = "./tutorials/mesh_files/t10.msh"
# --- 1. Load the Mesh ---
print("Loading mesh into Netgen...")
ngmesh = ReadGmsh(mesh_filepath)
mesh = Mesh(ngmesh)
print(f"Number of 3D elements: {mesh.ne}")
print(f"Number of vertices: {mesh.nv}")

# --- 2. Define Acoustic Parameters ---
c_0 = 343.0            # Speed of sound in air (m/s)
freq = 1200.0         # Excitation frequency (Hz)
omega = 2 * np.pi * freq
k = omega / c_0        # Wavenumber
rho_air = 1.225        # Density of air (kg/m^3)

Radius = 0.5
pts_source = np.array([0.6, 0.0, 0.0])  

# --- 3. Define the Finite Element Space ---
# complex=True is critical because the Helmholtz equation involves phase (imaginary numbers)
fes = H1(mesh, order=5, complex=True)
p, q = fes.TnT()

print(f"Degrees of freedom: {fes.ndof}")

# --- 4. Define the Weak Formulation ---
# Bilinear form (Left-Hand Side)
Z_fem = BilinearForm(fes, symmetric=True)
Z_fem += (grad(p)*grad(q) - k**2 * p * q) * dx("Fluid")
print("Assembling matrices...")
Z_fem.Assemble()

m_max = 7
wbm_model = WBM_sphere(mesh, Radius, freq, c_0, rho_air, pts_source, m_max)
Z_hyb, Z_wbm, s_hyb, s_wbm = wbm_model.assemble_matrices(fes, q, "Ellipsoid_Boundary")
total_waves = wbm_model.total_waves
cond_num = np.linalg.cond(Z_wbm)
print(f"Condition number of Z_wbm: {cond_num:.2e}")

HBSys = HybridSystem(fes, total_waves)
Global_Matrix, Global_RHS = HBSys.combine_matrices(Z_fem, Z_hyb, Z_wbm, s_hyb, s_wbm)
gfu_p, wbm_factors = HBSys.solve_coupled_system(Global_Matrix, Global_RHS)
print("Coupled system solved successfully!")

# ---- 6. Post-Processing and Visualization ---
print("\n--- Plotting in PyVista (High-Resolution Grid) ---")

resolution = 200
x_min, x_max = -0.5, 0.5
z_min, z_max = -0.5, 0.5

# 1. Let PyVista create a perfectly oriented 2D grid directly
# Dimensions are (Nx, Ny, Nz). We want a flat plane in Y, so Ny = 1
dx = (x_max - x_min) / (resolution - 1)
dz = (z_max - z_min) / (resolution - 1)

grid = pv.ImageData(
    dimensions=(resolution, 1, resolution),
    spacing=(dx, 1.0, dz),       # 1.0 is a dummy spacer for the Y-axis
    origin=(x_min, 0.0, z_min)   # Center the plane at y = 0
)

# 2. PyVista already exposes a flat list of (x, y, z) points
points = grid.points
pressure_values = np.zeros(grid.n_points)

print("Evaluating pressure field on the new grid...")
for i, (x_val, y_val, z_val) in enumerate(points):
    try:
        pressure_values[i] = gfu_p(mesh(x_val, y_val, z_val)).real
    except Exception:
        # Assign NaN if a point falls outside the fluid domain bounds
        pressure_values[i] = np.nan

# 3. Assign the calculated values back to the grid
grid.point_data["Real Pressure"] = pressure_values

# 4. Render the result
plotter = pv.Plotter()

plotter.add_mesh(grid, scalars="Real Pressure", cmap="coolwarm", show_edges=False)

# Formatting
plotter.set_background("white")
plotter.camera_position = 'xz'  # X will be strictly horizontal, Z strictly vertical
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()