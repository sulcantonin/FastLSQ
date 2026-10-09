# Copyright (c) 2026 Antonin Sulc
# Licensed under the MIT License. See LICENSE file for details.

"""Shared pytest fixtures.

Several tests switch ``torch``'s default dtype (a few exercise the float32
path).  That setting is process-global, so without this fixture a test's
outcome depended on which tests ran before it: ``test_solve_linear_phased_metrics``
followed by ``test_problems_integral.py`` failed seven accuracy assertions.  The
fixture restores whatever dtype was active before each test.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)
