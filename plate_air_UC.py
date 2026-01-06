import math, cmath
import netgen.occ as occ
from netgen.meshing import IdentificationType
from ngsolve import *

# -----------------------
# 0) User parameters
# -----------------------
# Duct size (x = propagation direction)
Lx = 0.40
Hy = 0.13
Hz = 0.13

# Plate position and thicknesses
xp = 0.20          # steel plate starts at x=xp
ts = 0.002         # steel thickness
tr = 0.003         # rubber layer thickness (downstream of steel)
r_hole = 0.05     # hole radius

maxh = 0.01
curve_order = 2

# Frequency / fluid
f0 = 500.0
omega = 2*math.pi*f0
rho0 = 1.2
c0   = 343.0
k    = omega / c0

# Bloch (transverse) wavevector components (set these!)
ky = 0.0   # [rad/m]
kz = 0.0   # [rad/m]

# Incoming wave amplitude
p0 = 1.0   # [Pa]

# Solid materials
# Steel-like
E_steel  = 210e9
nu_steel = 0.30
rho_steel = 7800.0

# Rubber-like (nearly incompressible: nu ~ 0.49 is typical)
E_rub  = 5e6
nu_rub = 0.49
rho_rub = 1100.0

def lame(E, nu):
    mu  = E/(2*(1+nu))
    lam = E*nu/((1+nu)*(1-2*nu))
    return lam, mu

lam_steel, mu_steel = lame(E_steel, nu_steel)
lam_rub,   mu_rub   = lame(E_rub,   nu_rub)

# -----------------------
# 1) Geometry (duct + steel plate w/ hole + rubber layer + rubber plug sealing hole)
# -----------------------
yc, zc = Hy/2, Hz/2

duct = occ.Box((0,0,0), (Lx,Hy,Hz))
duct.faces.Min(occ.X).name = "inlet"
duct.faces.Max(occ.X).name = "outlet"

# Steel plate box (full cross-section)
steel_box = occ.Box((xp, 0, 0), (xp+ts, Hy, Hz))
# Hole through steel thickness (axis along +x)
hole_steel = occ.Cylinder(occ.Pnt(xp, yc, zc), occ.gp_Vec(1,0,0), r_hole, ts)
steel = steel_box - hole_steel
steel.mat("steel")

# Rubber layer downstream, annulus (hole continues) ...
rub_box = occ.Box((xp+ts, 0, 0), (xp+ts+tr, Hy, Hz))
hole_rub = occ.Cylinder(occ.Pnt(xp+ts, yc, zc), occ.gp_Vec(1,0,0), r_hole, tr)
rub_layer = rub_box - hole_rub

# ... plus a rubber plug sealing the hole (fills hole through steel+rubber)
rub_plug = occ.Cylinder(occ.Pnt(xp, yc, zc), occ.gp_Vec(1,0,0), r_hole, ts+tr)

rubber = rub_layer + rub_plug
rubber.mat("rubber")

# Name only the true fluid-solid coupling faces (upstream and downstream faces)
steel.faces.Min(occ.X).name = "fsi_in"
rub_plug.faces.Min(occ.X).name = "fsi_in"
rubber.faces.Max(occ.X).name = "fsi_out"

# Fluid = duct minus solid union
solid_union = steel + rubber
fluid = duct - solid_union
fluid.mat("fluid")

# Glue multi-materials
geo = occ.Glue([fluid, steel, rubber])

# -----------------------
# 2) Periodic identifications in y and z (Bloch-ready)
#    Identify ALL faces near y=0 with ALL faces near y=Hy, similarly for z.
# -----------------------
eps = 1e-8
trfY = occ.gp_Trsf.Translation(Hy * occ.Y)
trfZ = occ.gp_Trsf.Translation(Hz * occ.Z)

geo.faces[occ.Y < (0+eps)].Identify(geo.faces[occ.Y > (Hy-eps)],
                                   "perY", IdentificationType.PERIODIC, trfY)
geo.faces[occ.Z < (0+eps)].Identify(geo.faces[occ.Z > (Hz-eps)],
                                   "perZ", IdentificationType.PERIODIC, trfZ)

mesh = Mesh(occ.OCCGeometry(geo, dim=3).GenerateMesh(maxh=maxh))

# Avoid elements with dofs on both master+slave (recommended in periodic tutorial)
mesh.ngmesh.Refine()
mesh.Curve(curve_order)

print("Materials:", mesh.GetMaterials())
print("Boundaries:", mesh.GetBoundaries())
print("Identifications:", len(mesh.ngmesh.GetIdentifications()))

# -----------------------
# 3) Bloch phases and axial wavenumber
# -----------------------
phaseY = cmath.exp(1j*ky*Hy)
phaseZ = cmath.exp(1j*kz*Hz)

kx = cmath.sqrt((omega/c0)**2 - ky**2 - kz**2)  # may be complex (evanescent)
print(kx)

# -----------------------
# 4) FE spaces (monolithic FSI), then wrap with Periodic(...)
# -----------------------
order_u = 2
order_p = 2

solid_region = mesh.Materials("steel|rubber")
fluid_region = mesh.Materials("fluid")

fes_u0 = VectorH1(mesh, order=order_u, complex=True, definedon=solid_region)
fes_p0 = H1(mesh,       order=order_p, complex=True, definedon=fluid_region)

# Quasi-periodic spaces: order of phases must match order identifications were added
fes_u = Periodic(fes_u0, phase=[phaseY, phaseZ])
fes_p = Periodic(fes_p0, phase=[phaseY, phaseZ])

fes = FESpace([fes_u, fes_p])
(u, p) = fes.TrialFunction()
(v, q) = fes.TestFunction()

# Region-wise material CFs
lam = mesh.MaterialCF({"steel": lam_steel, "rubber": lam_rub}, default=0)
mu  = mesh.MaterialCF({"steel": mu_steel,  "rubber": mu_rub},  default=0)
rhos = mesh.MaterialCF({"steel": rho_steel, "rubber": rho_rub}, default=0)

def Stress(eps_u):
    return 2*mu*eps_u + lam*Trace(eps_u)*Id(3)

dx_solid = dx(definedon=solid_region)
dx_fluid = dx(definedon=fluid_region)

bnd_in  = mesh.Boundaries("inlet")
bnd_out = mesh.Boundaries("outlet")
bnd_fsi = mesh.Boundaries("fsi_in|fsi_out")

n = specialcf.normal(mesh.dim)

# -----------------------
# 5) Variational forms
# -----------------------
a = BilinearForm(fes, symmetric=False)
f = LinearForm(fes)

# Solid: ∫ σ(u):ε(v) - ω² ρ u·v
a += InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v))) * dx_solid
a += (-omega**2) * rhos * InnerProduct(u, v) * dx_solid

# Fluid Helmholtz: ∫ ∇p·∇q - k² p q
a += (grad(p)*grad(q) - k**2*p*q) * dx_fluid

# Open ends with Floquet-impedance using kx:
#   ∂p/∂n - i*kx*p = g
a += (-1j*kx) * p*q * ds(definedon=bnd_in)
a += (-1j*kx) * p*q * ds(definedon=bnd_out)

# Incoming Floquet plane wave at inlet (consistent with Bloch):
# p_inc(x=0,y,z) = p0*exp(i*(ky*y + kz*z))
pinc = p0 * exp(1j*(ky*y + kz*z))
g_in = (-2j*kx) * pinc
f += g_in * q * ds(definedon=bnd_in)

# FSI coupling on internal interfaces:
#  (1) pressure traction on solid
a += (-p) * (v*n) * ds(definedon=bnd_fsi)
#  (2) normal acceleration drives fluid Neumann (frequency domain)
a += (rho0 * omega**2) * (u*n) * q * ds(definedon=bnd_fsi)

# -----------------------
# 6) Solve (no UMFPACK on Windows)
# -----------------------
gfu = GridFunction(fes)

with TaskManager():
    a.Assemble()
    f.Assemble()
    gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

gfu_u = gfu.components[0]
gfu_p = gfu.components[1]

# -----------------------
# 7) Export (real/imag/magnitude)
# -----------------------
u_re, u_im = gfu_u.real, gfu_u.imag
u_mag = sqrt(InnerProduct(u_re,u_re) + InnerProduct(u_im,u_im))

p_re, p_im = gfu_p.real, gfu_p.imag
p_abs = sqrt(p_re*p_re + p_im*p_im)

vtk = VTKOutput(mesh,
                coefs=[u_re, u_im, u_mag, p_re, p_im, p_abs],
                names=["u_re","u_im","u_mag","p_re","p_im","p_abs"],
                filename="va_duct_plate_bloch",
                subdivision=1)
vtk.Do()
print("Wrote: va_duct_plate_bloch.vtu")
