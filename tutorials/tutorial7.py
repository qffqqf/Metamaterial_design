import gmsh
import meshio
import pyvista as pv
import os

os.makedirs("./tutorials/mesh_files", exist_ok=True)

gmsh.initialize()
gmsh.model.add("bare_plates")

# --- 1. Define Parameters ---
L, W = 136e-3, 139e-3
H = 30e-3
H_s1 = 3e-3

# --- 2. Create Geometry ---
steel_box = gmsh.model.occ.addBox(0, 0, 0, L, W, H_s1)
fluid_box = gmsh.model.occ.addBox(0, 0, -H/3, L, W, H)
out_frag, out_frag_map = gmsh.model.occ.fragment(
    [(3, fluid_box)], [(3, steel_box)],
    removeObject=True, removeTool=True
)
gmsh.model.occ.synchronize()

# --- 3. Physical Groups ---
all_vols = gmsh.model.getEntities(3)
fluid_candidates = {tag for dim, tag in out_frag_map[0]}
vols_steel = [tag for dim, tag in out_frag_map[1]]
vols_fluid = list(fluid_candidates - set(vols_steel))

steel_id = gmsh.model.addPhysicalGroup(3, vols_steel, name="Steel")
fluid_id = gmsh.model.addPhysicalGroup(3, vols_fluid, name="Fluid")

# --- Fluid-Structure Interface (SIMPLIFIED via set intersection) ---
struct_boundary = gmsh.model.getBoundary([(3, t) for t in vols_steel], combined=True, oriented=False)
fluid_boundary  = gmsh.model.getBoundary([(3, t) for t in vols_fluid], combined=True, oriented=False)

struct_surfs = {tag for dim, tag in struct_boundary}
fluid_surfs  = {tag for dim, tag in fluid_boundary}
interface_tags = list(struct_surfs & fluid_surfs)

fsi_id = gmsh.model.addPhysicalGroup(2, interface_tags, name="fluid_structure_interface")

# --- Exterior Boundary Surfaces (SIMPLIFIED via dict-driven loop) ---
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

for name, (val, axis) in bounds.items():
    matching = []
    for dim, tag in all_surfs:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        low  = bbox[axis]
        high = bbox[axis + 3]
        if abs(low - val) < tol and abs(high - val) < tol:
            matching.append(tag)
    if matching:
        boundary_ids[name] = gmsh.model.addPhysicalGroup(2, matching, name=name)

# --- 4. Generate Mesh and Save ---
gmsh.option.setNumber("Mesh.MeshSizeMin", 30e-3)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("./tutorials/mesh_files/t7.msh")
gmsh.finalize()

# --- 5. PyVista Plotting ---
mesh = meshio.read("./tutorials/mesh_files/t7.msh")
mesh_pv = pv.from_meshio(mesh)

tetra_mesh = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TETRA)
tri_mesh   = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TRIANGLE)

# Volume components
steel_mesh = tetra_mesh.threshold([steel_id, steel_id], scalars="gmsh:physical")
fluid_mesh = tetra_mesh.threshold([fluid_id, fluid_id], scalars="gmsh:physical")

# Surface components
fsi_mesh   = tri_mesh.threshold([fsi_id, fsi_id], scalars="gmsh:physical")
z_max_mesh = tri_mesh.threshold([boundary_ids["Boundary_Z_Max"], boundary_ids["Boundary_Z_Max"]], scalars="gmsh:physical")
x_min_mesh = tri_mesh.threshold([boundary_ids["Boundary_X_Min"], boundary_ids["Boundary_X_Min"]], scalars="gmsh:physical")
y_max_mesh = tri_mesh.threshold([boundary_ids["Boundary_Y_Max"], boundary_ids["Boundary_Y_Max"]], scalars="gmsh:physical")

# --- Plot ---
plotter = pv.Plotter()

plotter.add_mesh(steel_mesh, show_edges=True, color="darkgray",  edge_color="black",    line_width=0.5, opacity=1.0, style="surface")
plotter.add_mesh(fluid_mesh, show_edges=True, color="lightblue", edge_color="lightblue", line_width=0.1, opacity=0.2, style="surface")

# FSI interface
# plotter.add_mesh(fsi_mesh, color="red", show_edges=True, edge_color="darkred", line_width=1.5, opacity=1.0, style="surface", label="FSI Interface")

# Exterior boundary surfaces
# plotter.add_mesh(z_max_mesh, color="green",  show_edges=True, edge_color="darkgreen", line_width=1.5, opacity=0.8, style="surface", label="Max Z Boundary")
# plotter.add_mesh(x_min_mesh, color="purple", show_edges=True, edge_color="indigo",    line_width=1.5, opacity=0.8, style="surface", label="Min X Boundary")
# plotter.add_mesh(y_max_mesh, color="yellow", show_edges=True, edge_color="indigo",    line_width=1.5, opacity=0.8, style="surface", label="Max Y Boundary")

plotter.set_background("white")
plotter.camera_position = "iso"
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()
