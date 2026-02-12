# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""PDE problem definitions for FastLSQ benchmarks."""

from fastlsq.problems.linear import (
    PoissonND, HeatND, Wave1D, Wave2D_MS, Helmholtz2D, Maxwell2D_TM,
)
from fastlsq.problems.nonlinear import (
    NLPoisson2D, Bratu2D, SteadyBurgers1D, NLHelmholtz2D, AllenCahn1D,
)
from fastlsq.problems.regression import (
    Burgers1D_Regression, KdV_Regression, ReactionDiffusion_Regression,
    SineGordon_Regression, KleinGordon_Regression, GrayScott_Pulse,
    NavierStokes2D_Kovasznay, Bratu2D_Regression, NLHelmholtz2D_Regression,
)

__all__ = [
    # Linear (solver mode)
    "PoissonND", "HeatND", "Wave1D", "Wave2D_MS", "Helmholtz2D",
    "Maxwell2D_TM",
    # Nonlinear (Newton solver mode)
    "NLPoisson2D", "Bratu2D", "SteadyBurgers1D", "NLHelmholtz2D",
    "AllenCahn1D",
    # Regression (data fitting)
    "Burgers1D_Regression", "KdV_Regression", "ReactionDiffusion_Regression",
    "SineGordon_Regression", "KleinGordon_Regression", "GrayScott_Pulse",
    "NavierStokes2D_Kovasznay", "Bratu2D_Regression",
    "NLHelmholtz2D_Regression",
]
