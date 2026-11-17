"""Classical 2-D comparator: parallel-beam TV reconstruction with an exactly adjoint
sparse projector (CPU), used for dense TV sweeps across many poses.

Projector: pixel-driven with 2x2 sub-samples per pixel and linear interpolation onto
1-um detector bins (A^T is the exact transpose). Geometry convention matches TIGRE's once
the in-plane axes are swapped (done in `slab_to_cols` / `cols_to_slab`).
Solver: diagonally preconditioned Chambolle-Pock (PDHG) for  min_{x>=0} 1/2 ||A_M x - b_M||^2 + lam * TV_iso(x),
applied to all slices of a slab at once (columns of X).
"""

import numpy as np
import scipy.sparse as sp

import common as c


def build_matrix(angles_deg, n=c.N_DET, n_det=c.N_DET, sub=2, flip_t=False, sign=1.0):
    off = (np.arange(sub) + 0.5) / sub - 0.5
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))  # ii row (y), jj column (x)
    rows, cols, vals = [], [], []
    pix = (ii * n + jj).ravel()
    for a_i, a in enumerate(np.deg2rad(angles_deg)):
        ca, sa = np.cos(sign * a), np.sin(sign * a)
        for dy in off:
            for dx in off:
                x = jj.ravel() + dx - (n - 1) / 2
                y = (n - 1) / 2 - (ii.ravel() + dy)
                t = x * ca + y * sa
                if flip_t:
                    t = -t
                u = t + (n_det - 1) / 2
                u0 = np.floor(u).astype(int)
                w1 = u - u0
                for uu, ww in ((u0, 1 - w1), (u0 + 1, w1)):
                    ok = (uu >= 0) & (uu < n_det) & (ww > 0)
                    rows.append(a_i * n_det + uu[ok])
                    cols.append(pix[ok])
                    vals.append(ww[ok] / sub**2)
    A = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(len(angles_deg) * n_det, n * n))
    A.sum_duplicates()
    return A


def grad(X, n):
    """X: (n*n, k) -> (2, n*n, k) forward differences (Neumann)."""
    I = X.reshape(n, n, -1)
    gy = np.zeros_like(I); gx = np.zeros_like(I)
    gy[:-1] = I[1:] - I[:-1]
    gx[:, :-1] = I[:, 1:] - I[:, :-1]
    return np.stack([gy.reshape(n * n, -1), gx.reshape(n * n, -1)])


def div(P, n):
    py = P[0].reshape(n, n, -1); px = P[1].reshape(n, n, -1)
    d = np.zeros_like(py)
    d[0] += py[0]; d[1:-1] += py[1:-1] - py[:-2]; d[-1] -= py[-2]
    d[:, 0] += px[:, 0]; d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]; d[:, -1] -= px[:, -2]
    return d.reshape(n * n, -1)


def op_norm(A, n, iters=30, seed=0):
    x = np.random.default_rng(seed).standard_normal((n * n, 1)).astype(np.float32)
    for _ in range(iters):
        y = A.T @ (A @ x) - div(grad(x, n), n)
        s = np.linalg.norm(y)
        x = y / s
    return float(np.sqrt(s))


def tv_recon(A, B, lam, n=c.N_DET, iters=300, X0=None):
    """Diagonally preconditioned PDHG (Pock & Chambolle 2011, alpha=1) for
    min_{x>=0} 1/2 ||A x - B||^2 + lam * TV_iso(x), columns of B solved jointly.
    B: (rows of A, k). Returns X (n*n, k)."""
    absA = abs(A)
    sig_d = (1.0 / np.maximum(np.asarray(absA.sum(1)).ravel(), 1e-8)).astype(np.float32)[:, None]
    tau = (1.0 / (np.asarray(absA.sum(0)).ravel() + 4.0)).astype(np.float32)[:, None]
    sig_g = 0.5
    k = B.shape[1]
    X = np.zeros((n * n, k), np.float32) if X0 is None else X0.astype(np.float32).copy()
    Xb = X.copy()
    Y = np.zeros_like(B, dtype=np.float32)
    P = np.zeros((2, n * n, k), np.float32)
    for _ in range(iters):
        Y = (Y + sig_d * (A @ Xb - B)) / (1 + sig_d)
        if lam > 0:
            P = P + sig_g * grad(Xb, n)
            P = P / np.maximum(1.0, np.sqrt((P**2).sum(0)) / lam)
        Xn = np.maximum(X - tau * (A.T @ Y - (div(P, n) if lam > 0 else 0)), 0)
        Xb = 2 * Xn - X
        X = Xn
    return X


def slab_to_cols(vol):
    """(z, y, x) TIGRE-order slab -> columns (x*y, z). The in-plane transpose makes this
    projector follow TIGRE's convention (1.2% difference on the phantom, vs ~60% without)."""
    return np.ascontiguousarray(np.transpose(vol, (0, 2, 1)).reshape(vol.shape[0], -1).T)


def cols_to_slab(X, n=c.N_DET):
    return np.ascontiguousarray(np.transpose(X.T.reshape(-1, n, n), (0, 2, 1)))


def sino_to_cols(sino, views):
    """(z, angle, det) -> rows (view-major, det) x columns z for the selected views."""
    return np.ascontiguousarray(sino[:, views, :].reshape(sino.shape[0], -1).T).astype(np.float32)
