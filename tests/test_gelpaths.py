"""test_gelpaths.py: framework to test the gelpaths module."""

import unittest

import torch
import numpy as np

from gel.gelpaths import ridge_paths


class TestGelPaths(unittest.TestCase):

    """Test cases for functions in gelpaths."""

    def test_ridge_paths(self):
        """Test the ridge_paths function."""
        # Setup
        X = torch.rand(20, 10)
        y = torch.rand(10)
        support = torch.tensor(range(20))
        lambdas = [0.1, 0.2, 0.3, 0.4, 0.5]
        summ_fun = lambda support, b: b

        summaries = ridge_paths(X, y, support, lambdas, summ_fun)
        # Compare each b with the naive solution
        I = torch.eye(X.size()[0])
        Q = X @ X.t()  # X@X.T
        r = X @ y  # X@y
        for l, b in summaries.items():
            b_naive = torch.inverse(Q + l * I) @ r
            self.assertTrue(
                np.allclose(b.numpy(), b_naive.numpy(), atol=1e-4, rtol=1e-3)
            )
