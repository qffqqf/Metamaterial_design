import numpy as np
from ngsolve import *
import netgen.read_gmsh
import meshio
import pyvista as pv
import netgen.gui 
from netgen.read_gmsh import ReadGmsh

mesh_filepath = "./tutorials/mesh_files/t9.msh"
# --- 1. Load the Mesh ---
# NGSolve can read Gmsh files directly. 
# 1. Use the dedicated Gmsh reader to create a Netgen mesh object first
print("Loading mesh into Netgen...")
ngmesh = ReadGmsh(mesh_filepath)

# 2. Wrap it into an NGSolve mesh
mesh = Mesh(ngmesh)

# Debug check: verify elements were actually loaded
print(f"Number of 3D elements: {mesh.ne}")
print(f"Number of vertices: {mesh.nv}")

# --- 2. Define Acoustic Parameters ---
c = 343.0            # Speed of sound in air (m/s)
freq = 1200.0         # Excitation frequency (Hz)
omega = 2 * np.pi * freq
k = omega / c        # Wavenumber

# --- 3. Define the Finite Element Space ---
# complex=True is critical because the Helmholtz equation involves phase (imaginary numbers)
fes = H1(mesh, order=4, complex=True)
u, v = fes.TnT()

print(f"Degrees of freedom: {fes.ndof}")

# --- 4. Define the Weak Formulation ---
# Bilinear form (Left-Hand Side)
a = BilinearForm(fes, symmetric=True)

# Volume terms: \int (\nabla u \cdot \nabla v - k^2 u v) dx
a += (grad(u)*grad(v) - k**2 * u * v) * dx("Fluid")

# radiation boundary conditions
a += -1j * k * u * v * ds("Sphere_Boundary")

# Linear form (Right-Hand Side) - Excitation Source
# A Gaussian volume source placed at the origin (center of the sphere)
source_term = exp(-200 * ((x+0.1)**2 + y**2 + z**2))

f = LinearForm(fes)
f += source_term * v * dx("Fluid")

# --- 5. Assemble and Solve ---
print("Assembling matrices...")
a.Assemble()
f.Assemble()

# GridFunction to store the solution
gfu = GridFunction(fes, name="acoustic_pressure")

print("Solving the linear system...")
# We use a sparse direct solver (UMFPACK) for the complex system
gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="umfpack") * f.vec

"""
# ---- 6. Post-Processing and Visualization ---
print("\n--- Plotting in PyVista ---")

# Load geometry into PyVista
mesh_raw = meshio.read(mesh_filepath)
mesh_pv = pv.from_meshio(mesh_raw)
tetra_mesh = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TETRA)

# Evaluate NGSolve solution at PyVista nodes
points = tetra_mesh.points
pressure_values = np.zeros(len(points))
for i, pt in enumerate(points):
    pressure_values[i] = gfu(mesh(*pt)).real

tetra_mesh.clear_cell_data()
tetra_mesh.clear_point_data()
tetra_mesh.point_data["Real Pressure"] = pressure_values

# Render the result
plotter = pv.Plotter()

plotter.add_mesh(tetra_mesh, cmap="coolwarm", show_edges=False, opacity=0.9)

# Formatting
plotter.set_background("white")
plotter.camera_position = "iso"
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()
"""
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
        pressure_values[i] = gfu(mesh(x_val, y_val, z_val)).real
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