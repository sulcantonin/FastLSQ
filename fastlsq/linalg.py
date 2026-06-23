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

import time
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
    """Returns ``(x, S)`` where ``S`` is the singular values (descending) when an
    explicit SVD is formed, else ``None`` (the fast LAPACK gelsd path)."""
    # Fast LAPACK rank-revealing driver for the common CPU / mu==0 case.
    if not mu and A.device.type == "cpu":
        try:
            return torch.linalg.lstsq(A, b, rcond=rcond, driver="gelsd").solution, None
        except (RuntimeError, ValueError):
            pass  # fall through to the explicit SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    filt = torch.where(S > rcond * S[0], S / (S * S + mu), torch.zeros_like(S))
    x = Vh.transpose(-2, -1) @ (filt.unsqueeze(-1) * (U.transpose(-2, -1) @ b))
    return x, S


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
    x = Vh.transpose(-2, -1) @ (filt.unsqueeze(-1) * (U.transpose(-2, -1) @ b))
    return x, S                                    # S is the truncated (rank-k) spectrum


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
    """Returns ``(x, S)``; ``S`` is the singular values only when the SVD safety
    net is taken (the Cholesky / QR fast paths return ``None``)."""
    # Cheap conditioning probe: cond(A) ~ max/min Cholesky pivot.  If well within
    # float64's reach use the fast Cholesky.
    try:
        x, L = _cholesky_solve(A, b, mu)
        d = torch.diagonal(L).abs()
        if torch.isfinite(d).all() and d.min() > (rcond ** 0.25) * d.max():
            return x, None
    except torch.linalg.LinAlgError:
        pass
    # Ill-conditioned.  On CPU with no ridge the LAPACK gelsd driver is both
    # faster than Householder QR *and* rank-deficient-safe, so go straight to it
    # -- the QR + blow-up-guard detour would only add a full extra factorization.
    if not mu and A.device.type == "cpu":
        return _svd_solve(A, b, mu, rcond)
    # Otherwise (ridge, or non-CPU device) try the backward-stable QR.  On a
    # genuinely rank-deficient *inconsistent* A unpivoted QR can return a wildly
    # non-minimum-norm solution, so fall back to the rank-revealing SVD when the
    # QR solution blows up (or is non-finite).  See _QR_AUTO_NORM_GUARD.
    x = _qr_solve(A, b, mu)
    nx = torch.linalg.vector_norm(x)
    if torch.isfinite(nx) and nx <= _QR_AUTO_NORM_GUARD * (1.0 + torch.linalg.vector_norm(b)):
        return x, None
    return _svd_solve(A, b, mu, rcond)


def _dispatch(A, b, mu, rcond, method, rank, oversample, n_iter):
    """Run the requested back-end; returns ``(x, S)`` where ``S`` is the singular
    values when the back-end already computed an SVD, else ``None``."""
    if method == "auto":
        return _auto_solve(A, b, mu, rcond)
    elif method == "svd":
        return _svd_solve(A, b, mu, rcond)
    elif method == "qr":
        return _qr_solve(A, b, mu), None
    elif method == "cholesky":
        return _cholesky_solve(A, b, mu)[0], None
    elif method == "rsvd":
        return _rsvd_solve(A, b, mu, rcond, rank, oversample, n_iter)
    else:
        raise ValueError(f"Unknown method {method!r}; "
                         "choose 'auto', 'qr', 'svd', 'cholesky', or 'rsvd'.")


def _sync_device(device):
    """Block until queued work on ``device`` finishes (no-op off CUDA/MPS).

    Without this an async CUDA solve's wall-clock measures only kernel-launch
    overhead, not compute -- the same primitive ``fastlsq.benchmark`` uses."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        sync = getattr(getattr(torch, "mps", None), "synchronize", None)
        if sync is not None:
            sync()


def solve_lstsq(A, b, mu=0.0, rcond=1e-12, method="auto",
                rank=None, oversample=10, n_iter=4, *, return_info=False):
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
    return_info : bool, optional
        If True, return ``(x, info)`` with a per-solve diagnostics dict
        ``{"t_solve", "rank_used", "residual", "cond_estimate"}`` instead of just
        ``x``.  ``t_solve`` is the device-synced wall-time of the **solve step
        only** (not assembly or scale search); ``rank_used`` is the rank-revealing
        effective numerical rank (singular values above ``rcond``); ``residual`` is
        the data residual ``||A x - b||``; ``cond_estimate`` is ``s_max / s_min``
        over the retained subspace.  The diagnostic singular values are computed
        *outside* the timed region (one extra ``svdvals`` when the chosen back-end
        did not already form an SVD), so ``t_solve`` stays honest.

    Returns
    -------
    x : Tensor, shape (N, K)
    info : dict, only if ``return_info=True``
    """
    A2, b2, mps_dev = _maybe_cpu(A, b)

    if not return_info:
        x, _ = _dispatch(A2, b2, mu, rcond, method, rank, oversample, n_iter)
        return x.to(mps_dev) if mps_dev is not None else x

    _sync_device(A2.device)
    t0 = time.perf_counter()
    x, S = _dispatch(A2, b2, mu, rcond, method, rank, oversample, n_iter)
    _sync_device(A2.device)
    t_solve = time.perf_counter() - t0

    # Diagnostics (untimed): reuse the back-end's spectrum, else one extra SVD.
    if S is None:
        S = torch.linalg.svdvals(A2)
    smax = S[0]
    keep = S > rcond * smax
    rank_used = int(keep.sum().item())
    if rank_used > 0:
        cond_estimate = float((smax / S[keep][-1]).item())
    else:
        cond_estimate = float("inf")
    residual = float((A2 @ x - b2).norm().item())
    info = {
        "t_solve": t_solve,
        "rank_used": rank_used,
        "residual": residual,
        "cond_estimate": cond_estimate,
    }
    x_out = x.to(mps_dev) if mps_dev is not None else x
    return x_out, info
