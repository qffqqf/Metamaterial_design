import gmsh
import meshio
import pyvista as pv
import os
import numpy as np

os.makedirs("./tutorials/mesh_files", exist_ok=True)

gmsh.initialize()
gmsh.model.add("air_sphere")

# --- 1. Define Parameters ---
R = 5e-1  # Sphere radius

# --- 2. Create Geometry ---
# Create a sphere at the origin
sphere = gmsh.model.occ.addSphere(0, 0, 0, R)
gmsh.model.occ.synchronize()

# --- 3. Create Physical Groups ---
# Volume: air sphere
fluid_id = gmsh.model.addPhysicalGroup(3, [sphere], name="Fluid")

# --- Exterior Spherical Boundary Surface ---
# Get all surfaces (2D entities) from the sphere
all_surfs = gmsh.model.getEntities(2)

# All surfaces of the sphere are exterior boundary surfaces
# Group them all under a single physical group for the spherical boundary
sphere_surf_tags = [tag for dim, tag in all_surfs]
boundary_id = gmsh.model.addPhysicalGroup(2, sphere_surf_tags, name="Sphere_Boundary")

# --- 4. Generate Mesh and Save ---
gmsh.option.setNumber("Mesh.MeshSizeMin", R / 20)
gmsh.option.setNumber("Mesh.MeshSizeMax", R / 5)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("./tutorials/mesh_files/t9.msh")
gmsh.finalize()

# --- 5. PyVista Plotting ---
mesh = meshio.read("./tutorials/mesh_files/t9.msh")
mesh_pv = pv.from_meshio(mesh)

tetra_mesh = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TETRA)
tri_mesh   = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TRIANGLE)

# Volume
fluid_mesh = tetra_mesh.threshold([fluid_id, fluid_id], scalars="gmsh:physical")

# Boundary surface
sphere_boundary_mesh = tri_mesh.threshold([boundary_id, boundary_id], scalars="gmsh:physical")

# --- Plot ---
plotter = pv.Plotter()

plotter.add_mesh(fluid_mesh, show_edges=True, color="lightblue", edge_color="lightblue", line_width=0.1, opacity=0.3, style="surface")
# plotter.add_mesh(sphere_boundary_mesh, color="green", show_edges=True, edge_color="darkgreen", line_width=1.5, opacity=0.8, style="surface", label="Sphere Boundary")

plotter.set_background("white")
plotter.camera_position = "iso"
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()
