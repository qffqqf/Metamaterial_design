import numpy as np
import matplotlib.pyplot as plt

# Parameters (modify these as you like)
r0 = 51.0    # base radius
a = 20     # amplitude
b = 4       # frequency (integer for classic rose-like patterns)

# Create theta values (0 to 2π is enough for integer k; try 0 to 4π for non‑integer k)
theta = np.linspace(0, 2 * np.pi, 1000)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True)

# Compute r in polar coordinates
r = r0 + 20 * (np.cos(4 * theta)-1)
x = r * np.cos(theta)
y = r * np.sin(theta)
ax1.plot(x, y, linewidth=3, label = "a = 20, b = 4")
r = r0 + 15 * (np.cos(8 * theta)-1)
x = r * np.cos(theta)
y = r * np.sin(theta)
ax2.plot(x, y, linewidth=3, label = "a = 15, b = 8")
r = r0 + 10 * (np.cos(12 * theta)-1)
x = r * np.cos(theta)
y = r * np.sin(theta)
ax3.plot(x, y, linewidth=3, label = "a = 10, b = 12")
r = r0 + 15 * (np.cos(3 * theta)-1)
x = r * np.cos(theta)
y = r * np.sin(theta)
ax4.plot(x, y, linewidth=3, label = "a = 15, b = 3")
for ax in [ax1, ax2, ax3, ax4]:
    ax.axis('equal')              # preserve aspect ratio
    ax.set_xlabel('x [mm]', fontsize=12)
    ax.legend()
    ax.grid(True)
ax1.set_ylabel('y [mm]', fontsize=12)
plt.show()