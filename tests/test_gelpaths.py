"""test_gelpaths.py: framework to test the gelpaths module."""

import itertools
import unittest

import torch
from scipy.spatial.distance import cosine

from gel.gelpaths import _find_support, compute_ls_grid, gel_paths, gel_paths2
from gel.ridgepaths import ridge_paths
from tests.test_gel import TestGelBirthwtBase


class TestGelPathsBirthwtBase:

    """Base class to test gel_paths with the birth weight data."""

    l_1_bases = [0.0, 1.0, 10.0, 100.0]
    l_2_bases = [0.0, 0.5, 5.0, 10.0, 50.0]
    l_rs = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    supp_thresh = 1e-2

    def __init__(self, device, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype

        self.gel_test = TestGelBirthwtBase(device, dtype)
        self.m = self.gel_test.m

        self.l_1s = [l_1_base / (2 * self.m) for l_1_base in self.l_1_bases]
        self.l_2s = [l_2_base / (2 * self.m) for l_2_base in self.l_2_bases]
        self.done_setup = False

    def setUp(self):
        if self.done_setup:
            return

        self.gel_test.setUp()
        self.As = self.gel_test.As
        self.X_p = torch.cat(self.As, dim=1).t()
        self.yt = self.gel_test.yt
        self.ns = self.gel_test.ns
        self.l1l2_grid = itertools.product(self.l_1s, self.l_2s)
        self._compute_summaries_cvxs()
        self.done_setup = True

    def _compute_summaries_cvxs(self):
        self.summaries_cvxs = dict()
        for l_1, l_2 in self.l1l2_grid:
            self.gel_test.l_1 = l_1
            self.gel_test.l_2 = l_2
            self.gel_test.done_setup = False
            self.gel_test.setUp()

            support_cvx = _find_support(self.gel_test.B_cvx, self.ns, self.supp_thresh)
            X_support = (
                None if support_cvx is None else self.X_p[support_cvx.to(self.device)]
            )

            summaries_cvx = ridge_paths(
                X_support,
                self.yt,
                support_cvx,
                self.l_rs,
                lambda support, b: (support, b),
            )
            for l_r, summary in summaries_cvx.items():
                self.summaries_cvxs[(l_1, l_2, l_r)] = summary

    def _compare_to_cvx(self, summaries):
        # pylint: disable=no-member
        for (l_1, l_2, l_r), (support, b) in summaries.items():
            support_cvx, b_cvx = self.summaries_cvxs[(l_1, l_2, l_r)]
            if support_cvx is None or support is None:
                self.assertIsNone(support)
                self.assertIsNone(b)
                self.assertIsNone(support_cvx)
                self.assertIsNone(b_cvx)
            else:
                self.assertEqual(support.tolist(), support_cvx.tolist())
                self.assertAlmostEqual(
                    cosine(b.cpu().numpy(), b_cvx.cpu().numpy()), 0, places=2
                )

    def _test_implementation(self, make_A, gel_solve, **gel_solve_kwargs):
        summaries = gel_paths(
            gel_solve,
            dict(gel_solve_kwargs),
            make_A,
            self.As,
            self.yt,
            self.l_1s,
            self.l_2s,
            self.l_rs,
            lambda support, b: (support, b),
            self.supp_thresh,
            self.device,
            dtype=self.dtype,
        )
        self._compare_to_cvx(summaries)

    def test_fista(self):
        TestGelBirthwtBase.test_fista(self)

    def test_cd_cvx(self):
        TestGelBirthwtBase.test_cd_cvx(self)

    def test_cd_agd(self):
        TestGelBirthwtBase.test_cd_agd(self)

    def test_cd_newton(self):
        TestGelBirthwtBase.test_cd_newton(self)

    test_fista.__doc__ = TestGelBirthwtBase.test_fista.__doc__
    test_cd_cvx.__doc__ = TestGelBirthwtBase.test_cd_cvx.__doc__
    test_cd_agd.__doc__ = TestGelBirthwtBase.test_cd_agd.__doc__
    test_cd_newton.__doc__ = TestGelBirthwtBase.test_cd_newton.__doc__


class TestGelPaths2BirthwtBase(TestGelPathsBirthwtBase):

    """Base class to test gel_paths2 on birth weight data."""

    ks = [0.1, 0.3, 0.6, 0.9]
    n_ls = 5
    l_eps = 1e-4
    l_rs = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    supp_thresh = 1e-2
    use_ls_grid = False

    def setUp(self):
        if self.done_setup:
            return

        self.gel_test.setUp()
        self.As = self.gel_test.As
        self.X_p = torch.cat(self.As, dim=1).t()
        self.yt = self.gel_test.yt
        self.ns = self.gel_test.ns

        self.sns_vec = self.ns.to(self.device, self.dtype)
        self.ls_grid = compute_ls_grid(
            self.As,
            self.yt,
            self.sns_vec,
            self.m,
            self.ks,
            self.n_ls,
            self.l_eps,
            self.dtype,
        )

        self.l1l2_grid = []
        for k, ls in self.ls_grid.items():
            for l in ls:
                l_1, l_2 = k * l, (1.0 - k) * l
                self.l1l2_grid.append((l_1, l_2))

        self._compute_summaries_cvxs()

    def _test_implementation(self, make_A, gel_solve, **gel_solve_kwargs):
        # pylint: disable=no-member
        if self.use_ls_grid:
            ls_grid_arg = self.ls_grid
        else:
            ls_grid_arg = None

        summaries = gel_paths2(
            gel_solve,
            gel_solve_kwargs,
            make_A,
            self.As,
            self.yt,
            self.ks,
            self.n_ls,
            self.l_eps,
            self.l_rs,
            lambda support, b: (support, b),
            self.supp_thresh,
            self.device,
            ls_grid=ls_grid_arg,
            dtype=self.dtype,
        )

        # Make sure the (l_1, l_2, l_r) grid is the same.
        self.assertSetEqual(set(summaries.keys()), set(self.summaries_cvxs.keys()))

        for l_1, l_2, _ in summaries:
            self.assertGreater(l_1, 0)
            self.assertGreater(l_2, 0)

        self._compare_to_cvx(summaries)


def create_gelpaths_birthwt_test(device_name, dtype, two=False, grid=False):
    device = torch.device(device_name)
    if two:
        base_cls = TestGelPaths2BirthwtBase
        T = "2"
    else:
        base_cls = TestGelPathsBirthwtBase
        T = ""
        assert grid is False

    if grid:
        T += "Grid"

    def __init__(self, *args, **kwargs):
        base_cls.__init__(self, device, dtype, *args, **kwargs)
        if grid:
            self.use_ls_grid = True

    _doc = "Test gel_paths{} on {} with {}".format(T, device_name, dtype)
    if grid:
        _doc += " (with precomputed ls_grid)"
    test_name = "TestGelPaths" + T + "Birthwt" + device_name.upper() + str(dtype)[-2:]

    globals()[test_name] = type(
        test_name,
        (base_cls, unittest.TestCase),
        {"__init__": __init__, "__doc__": _doc},
    )


for _device_name, _dtype in itertools.product(
    ["cpu", "cuda"], [torch.float32, torch.float64]
):
    create_gelpaths_birthwt_test(_device_name, _dtype)
    create_gelpaths_birthwt_test(_device_name, _dtype, two=True)
    create_gelpaths_birthwt_test(_device_name, _dtype, two=True, grid=True)
