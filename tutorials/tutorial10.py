import gmsh
import meshio
import pyvista as pv
import os

os.makedirs("./tutorials/mesh_files", exist_ok=True)

gmsh.initialize()
gmsh.model.add("air_ellipsoid")

# --- 1. Define Parameters ---
R = 5e-1  # Base radius

# Define scaling factors for the ellipsoid axes
scale_x = 1.0  # Stretches the X-axis to 2*R
scale_y = 0.3  # Keeps the Y-axis at 1*R
scale_z = 0.3  # Squashes the Z-axis to 0.5*R

# --- 2. Create Geometry ---
sphere = gmsh.model.occ.addSphere(0, 0, 0, R)

# Note: We no longer need to add tip1 and tip2 manually here for OCC.

gmsh.model.occ.affineTransform(
    [(3, sphere)], 
    [scale_x, 0.0, 0.0, 0.0,
     0.0, scale_y, 0.0, 0.0,
     0.0, 0.0, scale_z, 0.0,
     0.0, 0.0, 0.0, 1.0]
)

gmsh.model.occ.synchronize()

# --- 3. Create Physical Groups ---
fluid_id = gmsh.model.addPhysicalGroup(3, [sphere], name="Fluid")

all_surfs = gmsh.model.getEntities(2)
ellipsoid_surf_tags = [tag for dim, tag in all_surfs]
boundary_id = gmsh.model.addPhysicalGroup(2, ellipsoid_surf_tags, name="Ellipsoid_Boundary")

# --- 4. Generate Mesh with Fields ---
# Calculate expected tip coordinates
tip_x = R * scale_x

# Add dummy points to use as reference coordinates for our distance field
tip1_tag = gmsh.model.addDiscreteEntity(0)
gmsh.model.setCoordinates(tip1_tag, tip_x, 0, 0)
tip2_tag = gmsh.model.addDiscreteEntity(0)
gmsh.model.setCoordinates(tip2_tag, -tip_x, 0, 0)

# Field 1: Distance to the tips
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [tip1_tag, tip2_tag])

# Field 2: Threshold based on distance
# If distance < DistMin, mesh size is SizeMin
# If distance > DistMax, mesh size is SizeMax
gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", R / 100) # Extremely fine mesh at the tips
gmsh.model.mesh.field.setNumber(2, "SizeMax", R / 10)  # Coarse mesh far away
gmsh.model.mesh.field.setNumber(2, "DistMin", R / 20)  # Keep it fine very close to the tip
gmsh.model.mesh.field.setNumber(2, "DistMax", R / 2)   # Transition out to the coarse size

# Set this Threshold field as the background mesh
gmsh.model.mesh.field.setAsBackgroundMesh(2)

# Tell Gmsh to respect the background mesh strictly
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# Generate mesh
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("./tutorials/mesh_files/t10.msh")
gmsh.finalize()

# --- 5. PyVista Plotting ---
mesh = meshio.read("./tutorials/mesh_files/t10.msh")
mesh_pv = pv.from_meshio(mesh)

tetra_mesh = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TETRA)
tri_mesh   = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TRIANGLE)

fluid_mesh = tetra_mesh.threshold([fluid_id, fluid_id], scalars="gmsh:physical")
ellipsoid_boundary_mesh = tri_mesh.threshold([boundary_id, boundary_id], scalars="gmsh:physical")

# --- Plot ---
plotter = pv.Plotter()

# Setting edge_color to something darker so the refinement is visible
plotter.add_mesh(fluid_mesh, show_edges=True, color="lightblue", edge_color="blue", line_width=0.5, opacity=0.3, style="surface")

plotter.set_background("white")
plotter.camera_position = "iso"
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()