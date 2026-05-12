import numpy as np

def rational_fourier_interpolation(x, y, N, L):
    """
    Use QR decomposition to solve the non‑square system directly,
    mimicking MATLAB’s A\b for overdetermined least squares.
    """
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)
    s = len(x)
    if s <= 4*N:
        raise ValueError(f"s = {s} must be > 4*N = {4*N}.")

    n_unk = 4*N + 1
    A = np.zeros((s, n_unk), dtype=complex)
    for i, xi in enumerate(x):
        A[i, 0] = 1.0
    for n in range(1, N+1):
        for i, xi in enumerate(x):
            theta = 2* n * np.pi * xi / L
            exp_ps = np.exp(1j * theta)
            exp_ng = np.exp(-1j * theta)
            col_a = n
            col_b = N + n
            col_c = 2*N + n
            col_d = 3*N + n
            A[i, col_a] = exp_ps
            A[i, col_b] = exp_ng
            A[i, col_c] = -y[i] * exp_ps
            A[i, col_d] = -y[i] * exp_ng

    # QR decomposition of the non‑square A
    Q, R = np.linalg.qr(A)            # Q: (s,s) or (s,k), R: (k,m) with k = min(s,n_unk)
    # For overdetermined (s > n_unk), R is upper triangular of shape (n_unk, n_unk)
    # Q^T y, then take first n_unk rows
    coeffs = np.linalg.solve(R[:n_unk, :], (Q.T @ y)[:n_unk])

    a = coeffs[0:N+1]
    b = coeffs[N+1:2*N+1]
    c = coeffs[2*N+1:3*N+1]
    d = coeffs[3*N+1:4*N+1]
    return a, b, c, d

def eval_rational_fourier(x, a, b, c, d, L):
    """Evaluate the rational Fourier interpolant at points x."""
    x = np.asarray(x, dtype=complex)
    N = len(b)                     # = len(c) = len(d)
    num = np.full_like(x, a[0])
    den = np.ones_like(x)
    for n in range(1, N+1):
        theta = 2 * n * np.pi * x / L
        exp_ps = np.exp(1j * theta)
        exp_ng = np.exp(-1j * theta)
        num += a[n] * exp_ps + b[n-1] * exp_ng
        den += c[n-1] * exp_ps + d[n-1] * exp_ng
    return num / den