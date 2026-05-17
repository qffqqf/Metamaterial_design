import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Material & fluid properties (Example: 1 mm aluminium plate in air)
# =============================================================================
rho_plate = 2700.0      # Plate density (kg/m³)
E         = 70.0e9       # Young's modulus (Pa)
nu        = 0.33         # Poisson's ratio
h         = 0.04        # Plate thickness (m)
eta       = 1e-2         # Structural loss factor (damping)

rho0      = 1.21         # Air density (kg/m³)
c0        = 343.0        # Speed of sound in air (m/s)

# =============================================================================
# 2. Derived quantities
# =============================================================================
m_s       = rho_plate * h                       # Surface mass density (kg/m²)
D         = E * h**3 / (12 * (1 - nu**2))       # Flexural rigidity (N·m)

# Critical frequency (lowest coincidence frequency)
f_c       = c0**2 / (2 * np.pi) * np.sqrt(m_s / D)
print(f"Critical frequency: {f_c:.1f} Hz")

# Frequency vector (log-spaced from 10 Hz to 10 kHz)
f         = np.logspace(2, 3, 1000)             # 10 Hz to 10 kHz
omega     = 2 * np.pi * f
k0        = omega / c0                          # Acoustic wavenumber in fluid

# =============================================================================
# 3. Transmission coefficient τ and TL for selected incidence angles
# =============================================================================
angles_deg = [0, 45, 60, 75, 89]            # degrees from normal
angles_rad = np.deg2rad(angles_deg)

TL = {}                                          # store TL curves
for idx, phi in enumerate(angles_rad):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Term (k0*sin(phi))^4 scaled by D/omega
    term_k4 = (k0 * sin_phi)**4
    A = D / omega * term_k4                     # stiffness term (N/m)
    B = D / omega * term_k4 * eta               # damping term (N/m)

    # Transmission coefficient (Eq. from thin plate theory)
    Z0_cos = 2 * rho0 * c0 / cos_phi            # factor 2*ρ0c0 / cos(φ)
    tau = Z0_cos**2 / ( (Z0_cos + B)**2 + (omega * m_s - A)**2 )

    # Avoid log(0) by clipping tau to a small positive value
    tau = np.clip(tau, 1e-12, None)
    TL[angles_deg[idx]] = 10 * np.log10(1.0 / tau)

# =============================================================================
# 4. Diffuse‑field TL (integration over incident angles)
# =============================================================================
# Integration limits: 0 to 78° (common approximation for diffuse field)
theta_max = np.deg2rad(78)
n_theta   = 300                                 # number of integration points
theta_int = np.linspace(0, theta_max, n_theta)

# Pre‑allocate array for τ(θ, f)  (n_freq x n_theta)
tau_2d = np.zeros((len(f), n_theta))

for j, theta in enumerate(theta_int):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    k0_sin4 = (k0 * sin_t)**4
    A_t = D / omega * k0_sin4
    B_t = A_t * eta
    Z0_cos_t = 2 * rho0 * c0 / cos_t
    tau_t = Z0_cos_t**2 / ((Z0_cos_t + B_t)**2 + (omega * m_s - A_t)**2)
    tau_2d[:, j] = np.clip(tau_t, 1e-12, None)

# Integration: ∫ τ(θ) cos(θ) sin(θ) dθ  →  factor 2 inside integral
integrand = 2 * tau_2d * np.cos(theta_int) * np.sin(theta_int)
integral  = np.trapezoid(integrand, x=theta_int, axis=1)   # FIXED: np.trapezoid for NumPy >= 2.0

TL_diffuse = -10 * np.log10(np.clip(integral, 1e-12, None))

# =============================================================================
# 5. Plotting
# =============================================================================
plt.figure()

# Plot results for selected angles
for angle in angles_deg:
    plt.semilogx(f, TL[angle], label=f'{angle}°')

# Diffuse field curve
plt.semilogx(f, TL_diffuse, 'k--', linewidth=2, label='Diffuse field (0°–78°)')

# Critical frequency marker
plt.axvline(f_c, color='gray', linestyle=':', linewidth=1.5,
            label=f'Critical freq. ({f_c:.0f} Hz)')

# Optional: normal‑incidence mass law reference line (uncomment if desired)
# TL_mass = 20 * np.log10(f * m_s) - 47   # Normal incidence mass law [dB]
# plt.semilogx(f, TL_mass, 'r:', label='Mass law (normal)')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Transmission Loss (dB)')
plt.title(f'Sound Transmission Loss of an Infinite Thin Plate (Al {h*1000:.0f} mm)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.xlim(100, 1000)
plt.tight_layout()
plt.show()