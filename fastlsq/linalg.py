# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Linear-algebra back-ends for FastLSQ's one-shot least-squares solve.

The random-Fourier feature matrix ``A`` is usually severely **rank-deficient**
(many columns are near-duplicates; the effective rank can be a third of ``N``).
The naive routes both fail on it -- ``torch.linalg.lstsq`` amplifies the
near-null directions, and forming the normal equations ``A^T A`` *squares* the
condition number -- leaving several orders of magnitude of accuracy on the floor.

``solve_lstsq`` therefore exposes several back-ends via ``method=``:

* ``"qr"``       -- Householder-QR least squares (ridge via ``[A; sqrt(mu) I]``
                    augmentation).  Backward-stable at ``cond(A)`` -- SVD-grade
                    accuracy with no normal-equations squaring and no required
                    ridge, at ~QR cost (cheaper than SVD).  Assumes (numerically)
                    full column rank; ``"svd"`` is the rank-deficient-safe choice
                    (and ``"auto"``'s ultimate fallback if QR blows up).
* ``"svd"``      -- rank-revealing truncated SVD of ``A`` (LAPACK ``gelsd`` fast
                    path on CPU; explicit SVD elsewhere).  The accuracy reference;
                    use for a genuinely rank-deficient ``A``.
* ``"cholesky"`` -- normal-equations ``(A^T A + mu I)`` Cholesky.  Fast, but only
                    safe when ``A`` is well-conditioned.
* ``"rsvd"``     -- randomized SVD (range-finder + power iterations).  ``O(MNk)``
                    for a target ``rank`` k << N -- the cheap option for strongly
                    low-rank systems.
* ``"auto"`` (default) -- try Cholesky; if the system is ill-conditioned (a
                    cheap pivot-ratio test) use the faster ``"qr"``, and fall back
                    to rank-revealing ``"svd"`` only if QR's solution blows up (the
                    feature matrices can be rank-deficient).  Fast path when
                    well-conditioned, QR speed/accuracy on the rest, SVD as the
                    safety net.

All back-ends are device/dtype-aware.  Apple-MPS lacks a robust ``svd``/``lstsq``,
so the factorization is run on CPU and the result moved back (one-time warning).
"""

import warnings

import torch

_MPS_WARNED = False

# In ``method="auto"``: above this ``||x|| / (1 + ||b||)`` ratio the unpivoted-QR
# solve is treated as a rank-deficiency blow-up and handed to the rank-revealing
# SVD instead.  Real PDE systems measure <= 0.3 here; the degenerate inconsistent
# (random-RHS) rank-deficient case measures ~3e14 -- so the guard is generous and
# a false positive only costs speed, never correctness.
_QR_AUTO_NORM_GUARD = 1e6


def _maybe_cpu(A, b):
    """MPS has no robust svd/lstsq -- factorize on CPU, remember to move back."""
    global _MPS_WARNED
    if A.device.type == "mps":
        if not _MPS_WARNED:
            warnings.warn("FastLSQ: the linear solve runs on CPU because Apple "
                          "MPS lacks a robust SVD/lstsq; the result is moved "
                          "back to MPS. Assembly/feature ops stay on MPS.")
            _MPS_WARNED = True
        return A.cpu(), b.cpu(), A.device
    return A, b, None


def _svd_solve(A, b, mu, rcond):
    # Fast LAPACK rank-revealing driver for the common CPU / mu==0 case.
    if not mu and A.device.type == "cpu":
        try:
            return torch.linalg.lstsq(A, b, rcond=rcond, driver="gelsd").solution
        except (RuntimeError, ValueError):
            pass  # fall through to the explicit SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    filt = torch.where(S > rcond * S[0], S / (S * S + mu), torch.zeros_like(S))
    return Vh.transpose(-2, -1) @ (filt.unsqueeze(-1) * (U.transpose(-2, -1) @ b))


def _cholesky_solve(A, b, mu):
    """Normal-equations Cholesky.  Returns (x, L); raises if not PD."""
    AtA = A.transpose(-2, -1) @ A
    Atb = A.transpose(-2, -1) @ b
    if mu:
        AtA = AtA + mu * torch.eye(AtA.shape[-1], device=A.device, dtype=A.dtype)
    L = torch.linalg.cholesky(AtA)
    return torch.cholesky_solve(Atb, L), L


def _rsvd_solve(A, b, mu, rcond, rank, oversample, n_iter):
    """Randomized truncated-SVD pseudo-solve (Halko-Martinsson-Tropp)."""
    m, n = A.shape[-2], A.shape[-1]
    rank = min(m, n) if rank is None else rank
    k = min(rank + oversample, m, n)
    Omega = torch.randn(n, k, device=A.device, dtype=A.dtype)
    Q, _ = torch.linalg.qr(A @ Omega)
    for _ in range(n_iter):                       # power iterations for accuracy
        Q, _ = torch.linalg.qr(A.transpose(-2, -1) @ Q)
        Q, _ = torch.linalg.qr(A @ Q)
    B = Q.transpose(-2, -1) @ A                    # (k, n)
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub                                     # (m, k)
    filt = torch.where(S > rcond * S[0], S / (S * S + mu), torch.zeros_like(S))
    return Vh.transpose(-2, -1) @ (filt.unsqueeze(-1) * (U.transpose(-2, -1) @ b))


def _qr_solve(A, b, mu):
    """Householder-QR least squares (ridge via [A; sqrt(mu) I] augmentation).
    Backward-stable at cond(A): SVD-grade accuracy with NO normal-equations
    squaring and no required ridge, at ~QR cost (cheaper than SVD).  Assumes
    (numerically) full column rank; use method='svd' for a rank-deficient A."""
    if mu:
        n = A.shape[-1]
        A = torch.cat([A, (mu ** 0.5) * torch.eye(n, dtype=A.dtype, device=A.device)], dim=-2)
        b = torch.cat([b, torch.zeros(n, b.shape[-1], dtype=b.dtype, device=b.device)], dim=-2)
    Q, R = torch.linalg.qr(A, mode="reduced")
    return torch.linalg.solve_triangular(R, Q.transpose(-2, -1) @ b, upper=True)


def _auto_solve(A, b, mu, rcond):
    # Cheap conditioning probe: cond(A) ~ max/min Cholesky pivot.  If well within
    # float64's reach use the fast Cholesky.
    try:
        x, L = _cholesky_solve(A, b, mu)
        d = torch.diagonal(L).abs()
        if torch.isfinite(d).all() and d.min() > (rcond ** 0.25) * d.max():
            return x
    except torch.linalg.LinAlgError:
        pass
    # Ill-conditioned: try the faster, backward-stable QR.  On a genuinely
    # rank-deficient *inconsistent* A unpivoted QR can return a wildly
    # non-minimum-norm solution, so fall back to the rank-revealing SVD when the
    # QR solution blows up (or is non-finite).  See _QR_AUTO_NORM_GUARD.
    x = _qr_solve(A, b, mu)
    nx = torch.linalg.vector_norm(x)
    if torch.isfinite(nx) and nx <= _QR_AUTO_NORM_GUARD * (1.0 + torch.linalg.vector_norm(b)):
        return x
    return _svd_solve(A, b, mu, rcond)


def solve_lstsq(A, b, mu=0.0, rcond=1e-12, method="auto",
                rank=None, oversample=10, n_iter=4):
    """Solve  min ||A x - b||^2 + mu ||x||^2.

    Parameters
    ----------
    A : Tensor, shape (M, N)
    b : Tensor, shape (M, K)
    mu : float
        Tikhonov ridge (applied via spectral filtering / normal-equations, not as
        an unstable add-on).
    rcond : float
        Relative singular-value / pivot threshold for rank determination.
    method : {"auto", "qr", "svd", "cholesky", "rsvd"}
        Solve back-end (see module docstring).  Default "auto".
    rank, oversample, n_iter : int
        Randomized-SVD parameters (``method="rsvd"`` only).  Set ``rank`` << N for
        the speed-up; ``None`` uses the full rank (correct but no acceleration).

    Returns
    -------
    x : Tensor, shape (N, K)
    """
    A2, b2, mps_dev = _maybe_cpu(A, b)
    if method == "auto":
        x = _auto_solve(A2, b2, mu, rcond)
    elif method == "svd":
        x = _svd_solve(A2, b2, mu, rcond)
    elif method == "qr":
        x = _qr_solve(A2, b2, mu)
    elif method == "cholesky":
        x = _cholesky_solve(A2, b2, mu)[0]
    elif method == "rsvd":
        x = _rsvd_solve(A2, b2, mu, rcond, rank, oversample, n_iter)
    else:
        raise ValueError(f"Unknown method {method!r}; "
                         "choose 'auto', 'qr', 'svd', 'cholesky', or 'rsvd'.")
    return x.to(mps_dev) if mps_dev is not None else x
