from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import *
import netgen.gui  
import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

import sys
import os
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from HybridWB_FEM.wbm_top import WBM_Top
from HybridWB_FEM.wbm_bottom import WBM_Bottom

# =====================================================================
# 1. Geometry and mesh setup
# =====================================================================
print("\n 1. Geometry and mesh setup...\n")
# 1. Geometry Setup
L = 4.0

Lx = L
Ly = L/64
Lz = L
Lz_1 = 0.98 * L/2
Lz_2 = 0.98 * L/2      

Lx_plate = 0.9* Lx/2  
Lz_plate = L - Lz_1 - Lz_2

# 1. Bottom Air Volume (from z=0 to z=Lz_1)
air_bottom = Box((0, 0, 0), (Lx, Ly, Lz_1))
air_bottom.mat("air")
air_bottom.faces.Min(Z).name = "bottom"

# 2. Top Air Volume (from z=Lz_1+Lz_plate to z=L)
air_top = Box((0, 0, Lz_1 + Lz_plate), (Lx, Ly, Lz))
air_top.mat("air")
air_top.faces.Max(Z).name = "top"

# 3. Plate Volume left (from z=Lz_1 to z=Lz_1+Lz_plate, x=0 to x=Lx_plate)
plate_left = Box((0, 0, Lz_1), (Lx_plate, Ly, Lz_1 + Lz_plate))
plate_left.mat("plate")
plate_left.faces.Min(Z).name = "plate1_bottom"
plate_left.faces.Max(Z).name = "plate1_top"
plate_left.faces.Min(X).name = "plate1_left"
plate_left.faces.Max(X).name = "plate1_right"

# 4. Plate Volume right (from z=Lz_1 to z=Lz_1+Lz_plate, x=Lx-Lx_plate to x=Lx)
plate_right = Box((Lx - Lx_plate, 0, Lz_1), (Lx, Ly, Lz_1 + Lz_plate))
plate_right.mat("plate")
plate_right.faces.Min(Z).name = "plate2_bottom"
plate_right.faces.Max(Z).name = "plate2_top"
plate_right.faces.Min(X).name = "plate2_left"
plate_right.faces.Max(X).name = "plate2_right"

# 5. Air Volume in the middle (from z=Lz_1 to z=Lz_1+Lz_plate, x=Lx_plate to x=Lx-Lx_plate)
air_middle = Box((Lx_plate, 0, Lz_1), (Lx - Lx_plate, Ly, Lz_1 + Lz_plate))
air_middle.mat("air")   

# Glue them together to ensure conforming mesh nodes at the interfaces
geometry = Glue([air_bottom, plate_left, plate_right, air_top, air_middle])

# 4. Generate Mesh
geo = OCCGeometry(geometry)
mp = MeshingParameters(maxh=1.0, minh=0.2)
mesh = Mesh(geo.GenerateMesh(mp=mp))
Draw(mesh)
input("Mesh generated successfully! Press Enter to continue...")
print("Mesh generated successfully!")

# =====================================================================
# 2. Define physics and finite element space
# =====================================================================
print("\n 2. Define physics and finite element space...\n")
# 1. Physics Parameters
freq = 343.0     
c_0 = 343.0* (1 - 1j * 0.001)  
k = 2 * math.pi * freq / c_0  
rho_air = 1.21
omega = 2 * math.pi * freq

# Incident angles (e.g., 45 degrees)
theta = 0.0* (math.pi / 4)  # Polar angle (0 for normal incidence)
phi = 0.0

# Wave vector components
kx = k * math.sin(theta) * math.cos(phi)
ky = k * math.sin(theta) * math.sin(phi)
kz = k * math.cos(theta)

# Transverse wave vector for the Floquet shift
k_vec = CF((kx, ky, 0)) 

# Finite Element Space
fes = Periodic(H1(mesh, order=4, complex=True))

p,q = fes.TnT()

print(f"Degrees of freedom: {fes.ndof}")
print(f"wave vector k: ({kx:.2f}, {ky:.2f}, {kz:.2f})")

# =====================================================================
# 3. Define variational forms and assemble FE model
# =====================================================================
print("\n 3. Define variational forms and assemble FE model...\n")
# 1. Modified Gradients (Floquet Trick)
grad_p = grad(p) + 1j * k_vec * p
grad_q = grad(q) - 1j * k_vec * q 

# 2. Bilinear Form (LHS)
Z_fem = BilinearForm(fes)
Z_fem += (grad_p * grad_q - k**2 * p * q) * dx

# 3. Linear Form (RHS) - Surface source on the bottom boundary
s_fem = LinearForm(fes)
# Gaussian point source
source_func = exp(-((z-1*L/6)**2 + (x-Lx/2)**2 + (y-Ly/2)**2)/(L/20)**2)
s_fem += source_func* q * dx

# 4. Assemble 
with TaskManager():
    Z_fem.Assemble()
    s_fem.Assemble()
    
# 5. Extract free DOFs
freedofs = fes.FreeDofs()
free_indices = [i for i, is_free in enumerate(freedofs) if is_free]
slave_indices = [i for i, is_free in enumerate(freedofs) if not is_free]
print("Assembly complete!")