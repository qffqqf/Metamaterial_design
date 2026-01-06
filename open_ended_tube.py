import math
from ngsolve import *
from netgen.occ import *

# ============================================================
# 1) Geometry: rigid tube + (rubber + steel) plate with sealed center hole
# ============================================================
Lt = 0.40         # tube length [m] (x-direction)
Hy = 0.08         # tube height [m] (y-direction)
Hz = 0.08         # tube width  [m] (z-direction)

# Plate location and thicknesses
xp = 0.20         # steel starts at x = xp
ts = 0.002        # steel thickness
tr = 0.001        # rubber coating thickness (upstream / inlet side)

# Hole (in steel) + rubber plug sealing it
r_hole = 0.012
yc = 0.5 * Hy
zc = 0.5 * Hz

tube = Box((0, 0, 0), (Lt, Hy, Hz))
tube.faces.name = "wall"
tube.faces.Min(X).name = "inlet"
tube.faces.Max(X).name = "outlet"

# Rubber coating (inlet side): [xp-tr, xp]
rubber = Box((xp-tr, 0, 0), (xp, Hy, Hz))
rubber.mat("rubber")
rubber.faces.name = "plate_rim"
rubber.faces.Min(X).name = "fsi_in"
rubber.faces.Max(X).name = "bond"

# Steel plate: [xp, xp+ts]
steel = Box((xp, 0, 0), (xp+ts, Hy, Hz))
steel.mat("steel")
steel.faces.name = "plate_rim"
steel.faces.Min(X).name = "bond"
steel.faces.Max(X).name = "fsi_out"

# Hole cylinder through STEEL only
hole_steel = Cylinder((xp, yc, zc), X, r_hole, ts)

# Cut hole out of steel
steel = steel - hole_steel

# Rubber plug that fills the steel hole (seals it; no fluid cavity)
plug = Cylinder((xp, yc, zc), X, r_hole, ts)
plug.mat("rubber")
# Make sure the downstream plug face participates in FSI with the same name:
plug.faces.Max(X).name = "fsi_out"
plug.faces.Min(X).name = "bond"

# Solid assembly
solid = steel + rubber + plug

# Fluid domain is tube minus solid
fluid = tube - solid
fluid.mat("fluid")

geo = Glue([fluid, steel, rubber, plug])
mesh = Mesh(OCCGeometry(geo, dim=3).GenerateMesh(maxh=0.02))
mesh.Curve(2)

print("Materials:", mesh.GetMaterials())
print("Boundaries:", mesh.GetBoundaries())

# ============================================================
# 2) Parameters (frequency domain)
# ============================================================
# Air
rho0 = 1.2
c0   = 343.0

f0 = 500.0
omega = 2*math.pi*f0
k = omega / c0

# Incoming plane wave amplitude at inlet (Pa)
p0 = 1.0

# Steel
E_steel, nu_steel, rho_steel = 210e9, 0.30, 7800.0
mu_steel  = E_steel/(2*(1+nu_steel))
lam_steel = E_steel*nu_steel/((1+nu_steel)*(1-2*nu_steel))

# Rubber (example values; tune as needed)
E_rub, nu_rub, rho_rub = 5e6, 0.45, 1100.0
eta = 0.10                        # simple loss factor
E_rub = E_rub * (1 + 1j*eta)      # complex modulus for damping

mu_rub  = E_rub/(2*(1+nu_rub))
lam_rub = E_rub*nu_rub/((1+nu_rub)*(1-2*nu_rub))

# Piecewise Lamé parameters + density by material name
mu  = mesh.MaterialCF({"steel": mu_steel,  "rubber": mu_rub},  default=0)
lam = mesh.MaterialCF({"steel": lam_steel, "rubber": lam_rub}, default=0)
rho_s = mesh.MaterialCF({"steel": rho_steel, "rubber": rho_rub}, default=0)

def Stress(eps):
    return 2*mu*eps + lam*Trace(eps)*Id(3)

# ============================================================
# 3) FE spaces (complex harmonic response)
# ============================================================
order_u = 2
order_p = 2

solid_dom = mesh.Materials("steel|rubber")
fluid_dom = mesh.Materials("fluid")

dx_solid = dx(definedon=solid_dom)
dx_fluid = dx(definedon=fluid_dom)

# Clamp plate rim (where plate touches tube wall)
fes_u = VectorH1(mesh, order=order_u, complex=True,
                 dirichlet="plate_rim",
                 definedon=solid_dom)

fes_p = H1(mesh, order=order_p, complex=True,
           definedon=fluid_dom)

fes = FESpace([fes_u, fes_p])
(u, p) = fes.TrialFunction()
(v, q) = fes.TestFunction()

n = specialcf.normal(mesh.dim)

# Boundaries
B_open = mesh.Boundaries("inlet|outlet")
B_in   = mesh.Boundaries("inlet")
B_fsi  = mesh.Boundaries("fsi_in|fsi_out")

# ============================================================
# 4) Forms: solid + fluid + coupling + non-reflecting ends + incoming wave
# ============================================================
a = BilinearForm(fes, symmetric=False)

# Solid (steel+rubber):  Ku - ω²Mu
a += InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v))) * dx_solid
a += (-omega**2 * rho_s) * InnerProduct(u, v) * dx_solid

# Fluid Helmholtz:  ∫ ∇p·∇q - k² p q
a += (grad(p)*grad(q) - k**2 * p * q) * dx_fluid

# Open ends: Sommerfeld/impedance  ∂p/∂n - i k p = g
a += (-1j * k) * p * q * ds(definedon=B_open)

# Incoming plane wave at inlet: inject via g_in = -2 i k p0
g_in = -2j * k * p0
f = LinearForm(fes)
f += g_in * q * ds(definedon=B_in)

# Two-way FSI coupling on both faces of the plate
a += (-p) * (v*n) * ds(definedon=B_fsi)                  # pressure traction on solid
a += (rho0 * omega**2) * (u*n) * q * ds(definedon=B_fsi) # normal accel drives fluid Neumann

# ============================================================
# 5) Solve (no UMFPACK)
# ============================================================
gfu = GridFunction(fes)

with TaskManager():
    a.Assemble()
    f.Assemble()
    try:
        inv = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="pardiso")
    except Exception:
        print("Pardiso not available (or failed). Falling back to default inverse.")
        inv = a.mat.Inverse(freedofs=fes.FreeDofs())
    gfu.vec.data = inv * f.vec

gfu_u = gfu.components[0]
gfu_p = gfu.components[1]

# ============================================================
# 6) Export (ParaView-friendly)
# ============================================================
u_re = gfu_u.real
u_im = gfu_u.imag
u_mag = sqrt(InnerProduct(u_re, u_re) + InnerProduct(u_im, u_im))

p_re = gfu_p.real
p_im = gfu_p.imag
p_abs = sqrt(p_re*p_re + p_im*p_im)

vtk = VTKOutput(mesh,
                coefs=[u_mag, p_re, p_im, p_abs],
                names=["u_mag", "p_re", "p_im", "p_abs"],
                filename="tube_plate_rubber_steel_sealed_hole",
                subdivision=1)
vtk.Do()

print("Wrote: tube_plate_rubber_steel_sealed_hole.vtu")
