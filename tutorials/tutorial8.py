import gmsh
import meshio
import pyvista as pv
import os

os.makedirs("./tutorials/mesh_files", exist_ok=True)

gmsh.initialize()
gmsh.model.add("air_volume")

# --- 1. Define Parameters ---
L, W = 136e-3, 139e-3
H = 30e-3

# --- 2. Create Geometry ---
fluid_box = gmsh.model.occ.addBox(0, 0, -H/3, L, W, H)
gmsh.model.occ.synchronize()

# --- 3. Create Physical Groups ---
# FEniCSx requires explicit integer tags. 
# We assign tag=1 to the 3D volume.
fluid_id = gmsh.model.addPhysicalGroup(3, [fluid_box], tag=1, name="Fluid")

# --- Exterior Boundary Surfaces ---
tol = 1e-6
bounds = {
    "Boundary_Z_Min": (-H/3, 2),
    "Boundary_Z_Max": (-H/3 + H, 2),
    "Boundary_X_Min": (0, 0),
    "Boundary_X_Max": (L, 0),
    "Boundary_Y_Min": (0, 1),
    "Boundary_Y_Max": (W, 1),
}

all_surfs = gmsh.model.getEntities(2)
boundary_ids = {}

# We start boundary tags at 2 to keep them distinct from the volume tag
tag_counter = 2 

for name, (val, axis) in bounds.items():
    matching = []
    for dim, tag in all_surfs:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        low  = bbox[axis]
        high = bbox[axis + 3]
        if abs(low - val) < tol and abs(high - val) < tol:
            matching.append(tag)
    if matching:
        boundary_ids[name] = gmsh.model.addPhysicalGroup(2, matching, tag=tag_counter, name=name)
        tag_counter += 1

translation_x = [1, 0, 0, L,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1]

translation_y = [1, 0, 0, 0,
                 0, 1, 0, W,
                 0, 0, 1, 0,
                 0, 0, 0, 1]

gmsh.model.mesh.setPeriodic(2, [boundary_ids["Boundary_X_Max"]], [boundary_ids["Boundary_X_Min"]], translation_x)
gmsh.model.mesh.setPeriodic(2, [boundary_ids["Boundary_Y_Max"]], [boundary_ids["Boundary_Y_Min"]], translation_y)

# --- 4. Generate Mesh and Save ---
gmsh.option.setNumber("Mesh.MeshSizeMin", 30e-3)
gmsh.option.setNumber("Mesh.MeshSizeMax", 30e-3) 
gmsh.model.mesh.generate(3)

# CRITICAL FOR FENICSX: Must be version 4.1
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

gmsh.write("./tutorials/mesh_files/t8.msh")
gmsh.finalize()

print("Mesh successfully generated as version 4.1 and saved to ./tutorials/mesh_files/t8.msh")

# --- 5. PyVista Plotting ---
mesh = meshio.read("./tutorials/mesh_files/t8.msh")
mesh_pv = pv.from_meshio(mesh)

tetra_mesh = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TETRA)
tri_mesh   = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TRIANGLE)

# Volume
fluid_mesh = tetra_mesh.threshold([fluid_id, fluid_id], scalars="gmsh:physical")

# Boundary surfaces
z_min_mesh = tri_mesh.threshold([boundary_ids["Boundary_Z_Min"], boundary_ids["Boundary_Z_Min"]], scalars="gmsh:physical")
z_max_mesh = tri_mesh.threshold([boundary_ids["Boundary_Z_Max"], boundary_ids["Boundary_Z_Max"]], scalars="gmsh:physical")
x_min_mesh = tri_mesh.threshold([boundary_ids["Boundary_X_Min"], boundary_ids["Boundary_X_Min"]], scalars="gmsh:physical")
x_max_mesh = tri_mesh.threshold([boundary_ids["Boundary_X_Max"], boundary_ids["Boundary_X_Max"]], scalars="gmsh:physical")
y_min_mesh = tri_mesh.threshold([boundary_ids["Boundary_Y_Min"], boundary_ids["Boundary_Y_Min"]], scalars="gmsh:physical")
y_max_mesh = tri_mesh.threshold([boundary_ids["Boundary_Y_Max"], boundary_ids["Boundary_Y_Max"]], scalars="gmsh:physical")

# --- Plot ---
plotter = pv.Plotter()

plotter.add_mesh(fluid_mesh, show_edges=True, color="lightblue", edge_color="lightblue", line_width=0.1, opacity=0.3, style="surface")

# plotter.add_mesh(z_max_mesh, color="green",   show_edges=True, edge_color="green",   line_width=1.5, opacity=1.0, style="surface", label="Max Z Boundary")
# plotter.add_mesh(x_min_mesh, color="purple",  show_edges=True, edge_color="purple",  line_width=1.5, opacity=1.0, style="surface", label="Min X Boundary")
# plotter.add_mesh(y_min_mesh, color="yellow",  show_edges=True, edge_color="yellow",  line_width=1.5, opacity=1.0, style="surface", label="Min Y Boundary")

plotter.set_background("white")
plotter.camera_position = "iso"  
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()