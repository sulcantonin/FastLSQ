# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""
Block-structured assembly for vector-valued PDEs.

For a system with k unknown components u = (u_0, ..., u_{k-1}) and k coupled
equations, the operator matrix has block structure

    A = [[A_{00}, A_{01}, ..., A_{0,k-1}],
         [A_{10}, A_{11}, ..., A_{1,k-1}],
         ...                                ]

where A_{ij} (shape M_i x N) is the operator on component j appearing in
equation i. Coefficients are flattened as beta = [beta_0; beta_1; ...; beta_{k-1}]
in R^{Nk x 1}, and `unpack_beta` reshapes the solve output back to (N, k) so
that ``u(x) = basis.evaluate(x) @ beta`` returns (M, k) directly.
"""

import torch


def block_concat(blocks):
    """Assemble a 2-D nested list of tensors into a single block matrix.

    Parameters
    ----------
    blocks : list[list[Tensor | None]]
        Rectangular nested list. Block ``(i, j)`` is either a Tensor of
        shape ``(M_i, N_j)`` or ``None`` (interpreted as a zero block whose
        shape is inferred from non-None siblings in the same row/column).

    Returns
    -------
    Tensor of shape ``(sum_i M_i, sum_j N_j)``.
    """
    if not blocks or not blocks[0]:
        raise ValueError("block_concat requires a non-empty 2-D list of blocks")

    n_rows = len(blocks)
    n_cols = len(blocks[0])
    for row in blocks:
        if len(row) != n_cols:
            raise ValueError("All block rows must have the same number of columns")

    # Infer row heights M_i (from any non-None block in row i)
    row_h = [None] * n_rows
    col_w = [None] * n_cols
    ref = None
    for i in range(n_rows):
        for j in range(n_cols):
            b = blocks[i][j]
            if b is None:
                continue
            if ref is None:
                ref = b
            if row_h[i] is None:
                row_h[i] = b.shape[0]
            elif row_h[i] != b.shape[0]:
                raise ValueError(f"Inconsistent row height at ({i},{j})")
            if col_w[j] is None:
                col_w[j] = b.shape[1]
            elif col_w[j] != b.shape[1]:
                raise ValueError(f"Inconsistent column width at ({i},{j})")
    if ref is None:
        raise ValueError("block_concat requires at least one non-None block")
    for i, h in enumerate(row_h):
        if h is None:
            raise ValueError(f"Row {i} has no non-None block; cannot infer height")
    for j, w in enumerate(col_w):
        if w is None:
            raise ValueError(f"Column {j} has no non-None block; cannot infer width")

    rows = []
    for i in range(n_rows):
        row_blocks = []
        for j in range(n_cols):
            b = blocks[i][j]
            if b is None:
                b = torch.zeros(row_h[i], col_w[j], device=ref.device, dtype=ref.dtype)
            row_blocks.append(b)
        rows.append(torch.cat(row_blocks, dim=1))
    return torch.cat(rows, dim=0)


def pack_beta(beta):
    """Flatten ``beta`` of shape ``(N, k)`` to ``(N*k, 1)`` in component-major order.

    Component-major means ``beta_flat[j*N:(j+1)*N] == beta[:, j:j+1]``, matching
    the column-block layout used by ``block_concat``.
    """
    if beta.dim() == 1:
        beta = beta.unsqueeze(1)
    N, k = beta.shape
    return beta.T.reshape(N * k, 1)


def unpack_beta(beta_flat, n_features, n_outputs):
    """Reshape a flat ``(N*k, 1)`` coefficient vector back to ``(N, k)``.

    Inverse of ``pack_beta``: ``beta_flat[j*N:(j+1)*N]`` becomes column j.
    """
    if n_outputs == 1:
        return beta_flat.reshape(n_features, 1)
    return beta_flat.reshape(n_outputs, n_features).T
