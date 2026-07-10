import gmsh
import meshio
import pyvista as pv
import os
import numpy as np

os.makedirs("./tutorial/mesh_files", exist_ok=True)

gmsh.initialize()
gmsh.model.add("sandwich_plates")

# --- 1. Define Parameters ---
L, W = 136e-3, 139e-3
H = 30e-3
H_s1, H_r, H_s2 = 1.5e-3, 1e-3, 1.5e-3
hole_radius = 51e-3
r_res, h_res = 5e-3, 3e-3

# --- 2. Create Geometry ---
bot_steel = gmsh.model.occ.addBox(0, 0, 0, L, W, H_s1)
rubber    = gmsh.model.occ.addBox(0, 0, H_s1, L, W, H_r)
top_steel = gmsh.model.occ.addBox(0, 0, H_s1 + H_r, L, W, H_s2)

# Cut holes through the plates
cyl1 = gmsh.model.occ.addCylinder(L/2, W/2, 0, 0, 0, H_s1, hole_radius)
cyl2 = gmsh.model.occ.addCylinder(L/2, W/2, H_s1 + H_r, 0, 0, H_s2, hole_radius)
out_cut, out_cut_map = gmsh.model.occ.cut([(3, bot_steel), (3, rubber), (3, top_steel)],
                                 [(3, cyl1), (3, cyl2)],
                                 removeObject=True, removeTool=True)

# Fragment fluid box with cut plates + resonator
resonator = gmsh.model.occ.addCylinder(L/2, W/2, H_s1 + H_r, 0, 0, h_res, r_res)
fluid_box = gmsh.model.occ.addBox(0, 0, -H/4, L, W, H)

out_frag, out_frag_map = gmsh.model.occ.fragment(
    [(3, fluid_box)], out_cut + [(3, resonator)],
    removeObject=True, removeTool=True
)
gmsh.model.occ.synchronize()

# --- 3. Physical Groups ---
num_bottom, num_rubber, num_top = len(out_cut_map[0]), len(out_cut_map[1]), len(out_cut_map[2])

vols_bottom_steel, vols_rubber, vols_top_steel = [], [], []
vols_resonator, vols_fluid = [], []

# out_frag_map[0] = fluid box fragments
fluid_candidates = {tag for dim, tag in out_frag_map[0]}

# Now assign structural fragments: out_frag_map[1..3] = cut plates, out_frag_map[4] = resonator
vols_bottom_steel = [tag for dim, tag in out_frag_map[1]]
vols_rubber       = [tag for dim, tag in out_frag_map[2]]
vols_top_steel    = [tag for dim, tag in out_frag_map[3]]
vols_resonator    = [tag for dim, tag in out_frag_map[4]]
struct_set = set(vols_bottom_steel + vols_rubber + vols_top_steel + vols_resonator)
vols_fluid = list(fluid_candidates - struct_set)

bot_id  = gmsh.model.addPhysicalGroup(3, vols_bottom_steel, name="Bottom_Steel")
rub_id  = gmsh.model.addPhysicalGroup(3, vols_rubber, name="Rubber")
top_id  = gmsh.model.addPhysicalGroup(3, vols_top_steel, name="Top_Steel")
res_id  = gmsh.model.addPhysicalGroup(3, vols_resonator, name="Resonator")
flu_id  = gmsh.model.addPhysicalGroup(3, vols_fluid, name="Fluid")

# --- Fluid-Structure Interface (SIMPLIFIED via set intersection) ---
fluid_set = set(vols_fluid)

# Requirement 2: Fluid-Structure Interface (FSI)
surfs = gmsh.model.getEntities(2)
target_surfaces = []

for dim, tag in surfs:
    upward, downward = gmsh.model.getAdjacencies(dim, tag)
    # A true interface between two different materials has exactly 2 parent volumes
    if len(upward) == 2:
        v1, v2 = upward[0], upward[1]
        # Check if it bridges a structure volume and a fluid volume
        if (v1 in struct_set and v2 in fluid_set) or (v2 in struct_set and v1 in fluid_set):
            target_surfaces.append(tag)

fsi_id = gmsh.model.addPhysicalGroup(2, target_surfaces, name="fluid_structure_interface")

# --- Exterior Boundary Surfaces (SIMPLIFIED via dict-driven loop) ---
tol = 1e-6
bounds = {
    "Boundary_Z_Min": (-H/4, 2),
    "Boundary_Z_Max": (-H/4 + H, 2),
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
gmsh.option.setNumber("Mesh.MeshSizeMax", 1e0)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("./tutorial/mesh_files/t6.msh")
gmsh.finalize()

# --- 5. PyVista Plotting ---
mesh = meshio.read("./tutorial/mesh_files/t6.msh")
mesh_pv = pv.from_meshio(mesh)

tetra_mesh = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TETRA)
tri_mesh   = mesh_pv.extract_cells(mesh_pv.celltypes == pv.CellType.TRIANGLE)

# Volume components
bottom_steel_mesh = tetra_mesh.threshold([bot_id, bot_id], scalars="gmsh:physical")
top_steel_mesh    = tetra_mesh.threshold([top_id, top_id], scalars="gmsh:physical")
rubber_mesh       = tetra_mesh.threshold([rub_id, rub_id], scalars="gmsh:physical")
resonator_mesh    = tetra_mesh.threshold([res_id, res_id], scalars="gmsh:physical")
fluid_mesh        = tetra_mesh.threshold([flu_id, flu_id], scalars="gmsh:physical")

# Surface components
fsi_mesh     = tri_mesh.threshold([fsi_id, fsi_id], scalars="gmsh:physical")
z_max_mesh   = tri_mesh.threshold([boundary_ids["Boundary_Z_Max"], boundary_ids["Boundary_Z_Max"]], scalars="gmsh:physical")
x_min_mesh   = tri_mesh.threshold([boundary_ids["Boundary_X_Min"], boundary_ids["Boundary_X_Min"]], scalars="gmsh:physical")
y_max_mesh   = tri_mesh.threshold([boundary_ids["Boundary_Y_Max"], boundary_ids["Boundary_Y_Max"]], scalars="gmsh:physical")

# --- Plot ---
plotter = pv.Plotter()

# Volume bodies
plotter.add_mesh(bottom_steel_mesh, show_edges=True, color="darkgray",  edge_color="black",    line_width=0.5, opacity=1.0, style="surface")
plotter.add_mesh(top_steel_mesh,    show_edges=True, color="darkgray",  edge_color="black",    line_width=0.5, opacity=1.0, style="surface")
plotter.add_mesh(rubber_mesh,       show_edges=True, color="#654321",   edge_color="black",    line_width=0.5, opacity=1.0, style="surface")
plotter.add_mesh(resonator_mesh,    show_edges=True, color="#654321",   edge_color="black",    line_width=0.5, opacity=1.0, style="surface")
plotter.add_mesh(fluid_mesh,        show_edges=True, color="lightblue", edge_color="lightblue", line_width=0.1, opacity=0.2, style="surface")

# FSI interface (red overlay on the fluid-structure boundary)
# plotter.add_mesh(fsi_mesh, color="red", show_edges=True, edge_color="darkred", line_width=1.5, opacity=1.0, style="surface", label="FSI Interface")

# Exterior boundary surfaces
# plotter.add_mesh(z_max_mesh, color="green",  show_edges=True, edge_color="darkgreen", line_width=1.5, opacity=0.8, style="surface", label="Max Z Boundary")
# plotter.add_mesh(x_min_mesh, color="purple", show_edges=True, edge_color="indigo",    line_width=1.5, opacity=0.8, style="surface", label="Min X Boundary")
# plotter.add_mesh(y_max_mesh, color="yellow", show_edges=True, edge_color="indigo",    line_width=1.5, opacity=0.8, style="surface", label="Max Y Boundary")

plotter.set_background("white")
plotter.camera_position = "iso"
plotter.add_axes(line_width=2, color="black", x_color="gray", y_color="gray", z_color="gray")
plotter.show()
