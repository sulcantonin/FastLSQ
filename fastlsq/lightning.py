# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""PyTorch Lightning integration for FastLSQ."""

try:
    import pytorch_lightning as pl
    import torch.nn as nn
except ImportError:
    pl = None
    nn = None

import torch
from typing import Optional, Dict, Any

from fastlsq.solvers import FastLSQSolver
from fastlsq.utils import device


if pl is not None:

    class FastLSQModule(pl.LightningModule):
        """PyTorch Lightning module wrapper for FastLSQ solver.

        This allows FastLSQ to be integrated into PyTorch Lightning training
        loops, used with Lightning callbacks, and logged with TensorBoard.

        Example:
            >>> module = FastLSQModule(input_dim=2, n_blocks=3, hidden_size=500)
            >>> trainer = pl.Trainer(max_epochs=1)
            >>> trainer.fit(module, datamodule)
        """

        def __init__(
            self,
            input_dim: int,
            n_blocks: int = 3,
            hidden_size: int = 500,
            scale: float = 1.0,
            normalize: bool = False,
            learning_rate: float = 1e-3,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.n_blocks = n_blocks
            self.hidden_size = hidden_size
            self.scale = scale
            self.normalize = normalize
            self.learning_rate = learning_rate

            self.solver = FastLSQSolver(input_dim, normalize=normalize)
            for _ in range(n_blocks):
                self.solver.add_block(hidden_size=hidden_size, scale=scale)

            # Initialize beta as a parameter (will be optimized)
            self.register_parameter(
                "beta",
                nn.Parameter(torch.zeros(self.solver.n_features, 1, device=device)),
            )
            self.solver.beta = self.beta

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: predict solution at points x."""
            return self.solver.predict(x)

        def training_step(self, batch, batch_idx):
            """Training step (for compatibility with Lightning)."""
            # In FastLSQ, training is typically done via least-squares solve,
            # not gradient descent. This is a placeholder for integration.
            x, u_target = batch
            u_pred = self.forward(x)
            loss = torch.nn.functional.mse_loss(u_pred, u_target)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            """Configure optimizer (optional, for fine-tuning)."""
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def predict_step(self, batch, batch_idx):
            """Prediction step."""
            x = batch
            return self.forward(x)

else:

    class FastLSQModule:
        """Placeholder when PyTorch Lightning is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch Lightning is not installed. "
                "Install it with: pip install pytorch-lightning"
            )
