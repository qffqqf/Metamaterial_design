import numpy as np

def rational_poly_interpolation(x, y, N, L):
    """
    Use QR decomposition to solve the non‑square system directly,
    mimicking MATLAB’s A\b for overdetermined least squares.
    """
    x = np.asarray(x, dtype=complex)
    y = np.asarray(y, dtype=complex)
    s = len(x)
    if s <= 2*N:
        raise ValueError(f"s = {s} must be > 2*N = {2*N}.")
    cosi = (2 * x - L) / L
    n_unk = 2*N + 1
    A = np.zeros((s, n_unk), dtype=complex)
    # Evaluate Chebyshev polynomials by recurrence
    # T_0, T_1, ..., T_N for each xi_i
    T = np.zeros((s, N+1), dtype=complex)
    T[:, 0] = 1.0
    if N >= 1:
        T[:, 1] = cosi
        for k in range(2, N+1):
            T[:, k] = 2 * cosi * T[:, k-1] - T[:, k-2]

    # Fill design matrix
    A[:, 0] = 1.0          # coefficient a0
    for n in range(1, N+1):
        col_a = n                # a_n (columns 1..N)
        col_c = N + n            # c_n (columns N+1..2N)
        A[:, col_a] = T[:, n]
        A[:, col_c] = -y * T[:, n]

    # QR decomposition of the non‑square A
    Q, R = np.linalg.qr(A)
    coeffs = np.linalg.solve(R[:n_unk, :], (Q.T @ y)[:n_unk])

    a_coeff = coeffs[:N+1]
    c_coeff = coeffs[N+1:]
    return a_coeff, c_coeff

def eval_rational_poly(x, a_coeff, c_coeff, L):
    """Evaluate the rational polynomial interpolant at points x."""
    x = np.asarray(x, dtype=complex)
    cosi = (2 * x - L) / L
    N = len(c_coeff)
    
    # Recurrence for T_n
    T0 = np.ones_like(cosi)
    T1 = cosi.copy()
    num = a_coeff[0] * T0
    den = np.ones_like(cosi)
    if N > 0:
        num += a_coeff[1] * T1
        den += c_coeff[0] * T1
    T_prev = T1
    T_prev2 = T0
    for n in range(2, N+1):
        T_n = 2 * cosi * T_prev - T_prev2
        num += a_coeff[n] * T_n
        den += c_coeff[n-1] * T_n
        T_prev2, T_prev = T_prev, T_n
    return num / den