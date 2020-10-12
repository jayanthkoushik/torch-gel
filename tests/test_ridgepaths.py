"""test_ridgepaths.py: tests for ridge_paths function."""

import itertools
import unittest

import torch
from scipy.spatial.distance import cosine

from gel.ridgepaths import ridge_paths


class TestRidgePathsEmptySupport(unittest.TestCase):

    """Test ridge_paths with empty support."""

    def test_ridge_paths_empty_support(self):
        ls = range(5)
        summaries = ridge_paths(None, None, None, ls, lambda _, b: b)
        self.assertCountEqual(ls, summaries.keys())
        for l in ls:
            self.assertIsNone(summaries[l])


class TestRidgePathsBase:

    """Base class for ridge_paths tests."""

    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    def __init__(self, device, dtype, m, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.m = m
        self.p = p
        self.support = 1  # this is ignored

    def setUp(self):
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise unittest.SkipTest("cuda unavailable")

        self.X = torch.randn(self.p, self.m, device=self.device, dtype=self.dtype)
        self.y = torch.randn(self.m, device=self.device, dtype=self.dtype)

    def test_against_naive(self):
        """Compare with directly obtained solution."""
        # pylint: disable=no-member
        summaries = ridge_paths(
            self.X, self.y, self.support, self.lambdas, lambda _, b: b
        )
        # Compare each b with the naive solution.
        I = torch.eye(self.X.shape[0], device=self.device, dtype=self.dtype)
        Q = self.X @ self.X.t()
        r = self.X @ self.y
        for l, b in summaries.items():
            b_naive = torch.inverse(Q + l * I) @ r
            self.assertAlmostEqual(
                cosine(b.cpu().numpy(), b_naive.cpu().numpy()), 0, places=2
            )


def create_ridgepaths_test(device_name, dtype, m, p):
    device = torch.device(device_name)

    def __init__(self, *args, **kwargs):
        TestRidgePathsBase.__init__(self, device, dtype, m, p, *args, **kwargs)

    _doc = "Test ridge_paths on {} with {} ({}x{})".format(device_name, dtype, m, p)
    test_name = "TestRidgePaths" + device_name.upper() + str(dtype)[-2:]
    test_name += "_{}x{}".format(m, p)

    globals()[test_name] = type(
        test_name,
        (TestRidgePathsBase, unittest.TestCase),
        {"__init__": __init__, "__doc__": _doc},
    )


for _device_name, _dtype, _m, _p in itertools.product(
    ["cpu", "cuda"], [torch.float32, torch.float64], [1, 10], [1, 5, 10, 20]
):
    create_ridgepaths_test(_device_name, _dtype, _m, _p)
