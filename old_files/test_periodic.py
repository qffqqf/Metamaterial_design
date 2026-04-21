from ngsolve import *
from netgen.occ import *
from ngsolve.webgui import Draw

# ---------------------------------------------------------
# 1. Geometry and Boundary Naming
# ---------------------------------------------------------
box = Box((0,0,0), (1,1,1))

# Name the periodic faces
box.faces.Max(X).name = "right"
box.faces.Min(X).name = "left"

# Name all other faces "abc" (Absorbing Boundary Condition)
for f in box.faces:
    if f.name not in ["right", "left"]:
        f.name = "abc"

# Identify the left and right faces as periodic
box.faces.Max(X).Identify(box.faces.Min(X), "periodic_x")

# ---------------------------------------------------------
# 2. Mesh Generation
# ---------------------------------------------------------
geo = OCCGeometry(box)
# maxh needs to be small enough to resolve the wave (usually 6-10 elements per wavelength)
mesh = Mesh(geo.GenerateMesh(maxh=0.1))

# ---------------------------------------------------------
# 3. Finite Element Space
# ---------------------------------------------------------
# The Helmholtz equation requires complex numbers to represent wave phase.
# We wrap it in Periodic() to enforce the left=right mapping.
fes = Periodic(H1(mesh, order=3, complex=True))
print(f"Total degrees of freedom: {fes.ndof}")

u, v = fes.TnT()
k = 20  # Wave number

# ---------------------------------------------------------
# 4. Variational Forms
# ---------------------------------------------------------
a = BilinearForm(fes)
# Domain integral: (grad u * grad v) - (k^2 * u * v)
a += grad(u) * grad(v) * dx - (k**2) * u * v * dx
# Boundary integral for ABC: - i * k * u * v on the "abc" faces
a += -1j * k * u * v * ds("abc")
a.Assemble()

# Gaussian point source at the center
sigma = 0.02
source_expr = exp(-((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2) / sigma**2)

f = LinearForm(fes)
f += source_expr * v * dx
f.Assemble()

# ---------------------------------------------------------
# 5. Solve the System
# ---------------------------------------------------------
gfu = GridFunction(fes, name="wave_field")

# Solve using the default sparse direct solver
with TaskManager():
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# ---------------------------------------------------------
# 6. Visualization (Optional - requires Netgen GUI or Jupyter)
# ---------------------------------------------------------
# Draw the real part of the wave field
Draw(gfu)