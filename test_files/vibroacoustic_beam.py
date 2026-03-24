import math
from ngsolve import *
from netgen.occ import *

# -----------------------
# 1) Geometry + mesh
# -----------------------
L = 0.20          # cube size [m]
beam_L = 0.16
beam_w = 0.02
beam_h = 0.02

# Beam centered in y,z, located inside cube
x0 = 0.02
x1 = x0 + beam_L
yc = 0.5 * L
zc = 0.5 * L

cube = Box((0, 0, 0), (L, L, L))
cube.faces.name = "outer"  # all 6 faces

beam = Box((x0, yc - beam_w/2, zc - beam_h/2),
           (x1, yc + beam_w/2, zc + beam_h/2))

beam.faces.name = "interface"
beam.faces.Min(X).name = "clamp"
beam.faces.Max(X).name = "load"

fluid = cube - beam
fluid.mat("fluid")
beam.mat("beam")

geo = Glue([fluid, beam])
mesh = Mesh(OCCGeometry(geo, dim=3).GenerateMesh(maxh=0.02))
mesh.Curve(2)

print("Materials:", mesh.GetMaterials())
print("Boundaries:", mesh.GetBoundaries())

# -----------------------
# 2) Physics parameters
# -----------------------
# Solid (steel-like)
E = 210e9
nu = 0.30
rho_s = 7800

mu  = E/(2*(1+nu))
lam = E*nu/((1+nu)*(1-2*nu))

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(3)

# Fluid (air-like)
rho0 = 1.2
c0   = 343.0

f0 = 500.0
omega = 2*math.pi*f0
k = omega/c0

# -----------------------
# 3) FE spaces (complex harmonic response)
# -----------------------
order_u = 2
order_p = 2

fes_u = VectorH1(mesh, order=order_u, complex=True,
                 dirichlet="clamp",
                 definedon=mesh.Materials("beam"))

fes_p = H1(mesh, order=order_p, complex=True,
           definedon=mesh.Materials("fluid"))

fes = FESpace([fes_u, fes_p])
(u, p) = fes.TrialFunction()
(v, q) = fes.TestFunction()

n = specialcf.normal(mesh.dim)

dx_beam  = dx(definedon=mesh.Materials("beam"))
dx_fluid = dx(definedon=mesh.Materials("fluid"))

# Boundary regions (your printed names match these)
OuterBnd = mesh.Boundaries("outer")
GammaFSI = mesh.Boundaries("interface|clamp|load")

# -----------------------
# 4) Variational forms (monolithic coupling)
# -----------------------
a = BilinearForm(fes, symmetric=False)

# Solid: ∫ σ(u):ε(v) dΩ  - ω² ρ ∫ u·v dΩ
a += InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v))) * dx_beam
a += (-omega**2 * rho_s) * InnerProduct(u, v) * dx_beam

# Fluid: ∫ ∇p·∇q dΩ - k² ∫ p q dΩ
a += (grad(p)*grad(q) - k**2*p*q) * dx_fluid

# Non-reflecting outer BC (Sommerfeld/impedance): ∂p/∂n - i k p = 0
a += (-1j * k) * p * q * ds(definedon=OuterBnd)

# Coupling on beam surface:
# (1) pressure traction on solid
a += (-p) * (v*n) * ds(definedon=GammaFSI)
# (2) normal acceleration drives fluid Neumann: ∂p/∂n = ρ ω² u_n
a += (rho0 * omega**2) * (u*n) * q * ds(definedon=GammaFSI)

# Load on beam end: resultant shear force Fy (along +y) applied on end face
Fy = 1.0                     # [N] desired shear force amplitude
Aend = beam_w * beam_h        # end-face area [m²]
t_shear = CF((0, 0, Fy/Aend)) # [Pa] traction giving net shear Fy

f = LinearForm(fes)
f += InnerProduct(t_shear, v) * ds("load")


# -----------------------
# 5) Solve (NO UMFPACK)
# -----------------------
gfu = GridFunction(fes)

with TaskManager():
    a.Assemble()
    f.Assemble()

    inv = a.mat.Inverse(freedofs=fes.FreeDofs())

    gfu.vec.data = inv * f.vec

gfu_u = gfu.components[0]
gfu_p = gfu.components[1]

# -----------------------
# 6) Export to ParaView (avoid complex fields directly)
# -----------------------
u_re = gfu_u.real
u_im = gfu_u.imag
u_mag = sqrt(InnerProduct(u_re, u_re) + InnerProduct(u_im, u_im))

p_re = gfu_p.real
p_im = gfu_p.imag
p_abs = sqrt(p_re*p_re + p_im*p_im)

vtk = VTKOutput(mesh,
                coefs=[u_re, u_im, u_mag, p_re, p_im, p_abs],
                names=["u_re", "u_im", "u_mag", "p_re", "p_im", "p_abs"],
                filename="va_beam_cube",
                subdivision=1)
vtk.Do()


print("Wrote: va_beam_cube.vtu")
