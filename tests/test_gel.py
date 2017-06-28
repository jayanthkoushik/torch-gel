"""test_gel.py: framework to test gel implementations."""

import unittest
import os

import torch
import cvxpy as cvx
import numpy as np

from gel.gelfista import gel_solve as gel_solve_fista
from gel.gelfista import make_A as make_A_fista
from gel.gelcd import gel_solve as gel_solve_cd
from gel.gelcd import make_A as make_A_cd
from gel.gelcd import block_solve_cvx, block_solve_agd, block_solve_newton


def gel_solve_cvx(As, y, l_1, l_2, ns):
    """Solve a group elastic net problem with cvx.

    Arguments:
        As: list of numpy arrays.
        y: numpy array.
        l_1, l_2: floats.
        ns: numpy array.
    """
    # Create the b variables
    b_0 = cvx.Variable()
    bs = []
    for A_j, n_j in zip(As, ns):
        bs.append(cvx.Variable(n_j))

    # Form g(b)
    Ab = sum(A_j*b_j for A_j, b_j in zip(As, bs))
    m = As[0].shape[0]
    g_b = cvx.square(cvx.norm2(y - b_0 - Ab)) / (2*m)

    # Form h(b)
    h_b = sum(np.sqrt(n_j)*(l_1*cvx.norm2(b_j) + l_2*cvx.square(cvx.norm2(b_j)))
              for n_j, b_j in zip(ns, bs))

    # Build the optimization problem
    obj = cvx.Minimize(g_b + h_b)
    problem = cvx.Problem(obj, constraints=None)

    problem.solve(solver="CVXOPT")

    b_0 = b_0.value
    # Form B as returned by gel_solve
    p = len(As)
    B = torch.zeros(p, int(max(ns)))
    for j in range(p):
        b_j = np.asarray(bs[j].value)
        if ns[j] == 1:
            b_j = b_j.reshape(1,)
        B[j, :ns[j]] = torch.FloatTensor(b_j)

    return b_0, B


def _b2vec(B, groups):
    """Convert B as returned by gel_solve functions to a single numpy vector."""
    d = sum(len(group_j) for group_j in groups) # The total dimension
    b = np.zeros(d)
    for j, group_j in enumerate(groups):
        b[group_j] = B[j, :len(group_j)].numpy()
    return b


class TestGelBirthwt(unittest.TestCase):

    """Test different gel_solve implementations with the birth weight data."""

    def __init__(self, *args, **kwargs):
        """Load data and solve with cvx to get ground truth solution."""
        super().__init__(*args, **kwargs)
        data_dir = os.path.join(os.path.dirname(__file__), "data", "birthwt")
        self.X = np.loadtxt(os.path.join(data_dir, "X.csv"), skiprows=1,
                            delimiter=",")
        self.y = np.loadtxt(os.path.join(data_dir, "y.csv"), skiprows=1)
        self.groups = [[0, 1, 2], [3, 4, 5], [6, 7], [8], [9, 10], [11], [12],
                       [13, 14, 15]]
        self.p = len(self.groups)
        self.m = len(self.y)
        self.l_1 = 4. / (2*self.m)
        self.l_2 = 0.5 / (2*self.m)

        # Convert things to gel format
        self.As = []
        for j in range(self.p):
            A_j = self.X[:, self.groups[j]]
            self.As.append(torch.FloatTensor(A_j))
        self.yt = torch.FloatTensor(self.y)
        self.ns = torch.LongTensor([len(g) for g in self.groups])

        # Solve with cvx
        As_cvx = [A_j.numpy() for A_j in self.As]
        self.b_0_cvx, B = gel_solve_cvx(As_cvx, self.y, self.l_1, self.l_2,
                                        self.ns.numpy())
        self.b_cvx = _b2vec(B, self.groups)
        self.obj_cvx = self._obj(self.b_0_cvx, self.b_cvx)

    def _obj(self, b_0, b):
        """Compute the objective function value for the given b_0, b."""
        r = self.y - b_0 - self.X@b
        g_b = r@r / (2.*self.m)
        b_j_norms = [np.linalg.norm(b[self.groups[j]], ord=2)
                     for j in range(self.p)]
        h_b = self.l_1 * sum(np.sqrt(len(self.groups[j]))*b_j_norms[j]
                             for j in range(self.p))
        h_b += self.l_2 * sum(np.sqrt(len(self.groups[j]))*(b_j_norms[j]**2)
                              for j in range(self.p))
        return g_b + h_b

    def _compare_to_cvx(self, b_0, b, obj):
        """Compare the given solution to the cvx solution."""
        self.assertTrue(np.allclose(obj, self.obj_cvx))
        self.assertTrue(np.allclose(b_0, self.b_0_cvx, atol=1e-4, rtol=1e-3))
        self.assertTrue(np.allclose(b, self.b_cvx, atol=1e-4, rtol=1e-3))

    def _test_implementation(self, make_A, gel_solve, **gel_solve_kwargs):
        """Test the given implementation."""
        A = make_A(self.As, self.ns)
        b_0, B = gel_solve(A, self.yt, self.l_1, self.l_2, self.ns, self.b_init,
                           **gel_solve_kwargs)
        b = _b2vec(B, self.groups)
        obj = self._obj(b_0, b)
        self._compare_to_cvx(b_0, b, obj)

    def setUp(self):
        self.b_init = 0., torch.zeros(self.p, self.ns.max())

    def test_fista(self):
        """Test the FISTA implementation of gel_solve."""
        self._test_implementation(make_A_fista, gel_solve_fista, t_init=0.1,
                                  ls_beta=0.9, max_iters=None, rel_tol=1e-6)

    def test_cd_cvx(self):
        """Test the CD implementation with cvx internal solver."""
        self._test_implementation(make_A_cd, gel_solve_cd,
                                  block_solve_fun=block_solve_cvx,
                                  block_solve_kwargs={}, max_cd_iters=None,
                                  rel_tol=1e-6)

    def test_cd_agd(self):
        """Test the CD implementation with AGD internal solver."""
        self._test_implementation(make_A_cd, gel_solve_cd,
                                  block_solve_fun=block_solve_agd,
                                  block_solve_kwargs={
                                      "t_init": 0.1,
                                      "ls_beta": 0.9,
                                      "max_iters": None,
                                      "rel_tol": 1e-6
                                  }, max_cd_iters=None, rel_tol=1e-6)

    def test_cd_newton(self):
        """Test the CD implementation with Newton internal solver."""
        # Compute the C_js and I_js
        Cs = [(A_j.t()@A_j)/self.m for A_j in self.As]
        Is = [torch.eye(n_j) for n_j in self.ns]
        self._test_implementation(make_A_cd, gel_solve_cd,
                                  block_solve_fun=block_solve_newton,
                                  block_solve_kwargs={
                                      "ls_alpha": 0.01,
                                      "ls_beta": 0.9,
                                      "max_iters": 4,
                                      "tol": 1e-10
                                  }, max_cd_iters=None, rel_tol=1e-6, Cs=Cs,
                                  Is=Is)
