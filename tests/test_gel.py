"""test_gel.py: framework to test gel implementations."""

import itertools
import os
import unittest

import cvxpy as cvx
import numpy as np
import torch
from scipy.spatial.distance import cosine

from gel.gelcd import (
    block_solve_agd,
    block_solve_newton,
    gel_solve as gel_solve_cd,
    make_A as make_A_cd,
)
from gel.gelfista import gel_solve as gel_solve_fista, make_A as make_A_fista


def gel_solve_cvx(As, y, l_1, l_2, ns):
    """Solve a group elastic net problem with cvx.

    Arguments:
        As: list of tensors.
        y: tensor.
        l_1, l_2: floats.
        ns: iterable.
    """
    # Convert everything to numpy
    dtype = As[0].dtype
    As = [A_j.cpu().numpy() for A_j in As]
    y = y.cpu().numpy()
    ns = np.array([int(n) for n in ns])

    # Create the b variables.
    b_0 = cvx.Variable()
    bs = []
    for _, n_j in zip(As, ns):
        bs.append(cvx.Variable(n_j))

    # Form g(b).
    Ab = sum(A_j * b_j for A_j, b_j in zip(As, bs))
    m = As[0].shape[0]
    g_b = cvx.square(cvx.norm(y - b_0 - Ab)) / (2 * m)

    # Form h(b).
    h_b = sum(
        np.sqrt(n_j) * (l_1 * cvx.norm(b_j) + l_2 * cvx.square(cvx.norm(b_j)))
        for n_j, b_j in zip(ns, bs)
    )

    # Build the optimization problem.
    obj = cvx.Minimize(g_b + h_b)
    problem = cvx.Problem(obj, constraints=None)

    problem.solve(solver="CVXOPT")

    b_0 = b_0.value.item()
    # Form B as returned by gel_solve.
    p = len(As)
    B = torch.zeros(p, int(max(ns)), dtype=dtype)
    for j in range(p):
        b_j = np.asarray(bs[j].value)
        B[j, : ns[j]] = torch.from_numpy(b_j)

    return b_0, B


def block_solve_cvx(r_j, A_j, a_1_j, a_2_j, m, b_j_init, verbose=False):
    # pylint: disable=unused-argument
    """Solve the gelcd optimization problem for a single block with cvx.

    b_j_init and verbose are ignored. b_j_init because cvx doesn't support it.
    verbose because it doesn't go together with tqdm.
    """
    # Convert everything to numpy.
    device = A_j.device
    dtype = A_j.dtype
    r_j = r_j.cpu().numpy()
    A_j = A_j.cpu().numpy()

    # Create the b_j variable.
    b_j = cvx.Variable(A_j.shape[1])

    # Form the objective.
    q_j = r_j - A_j * b_j
    obj_fun = cvx.square(cvx.norm(q_j)) / (2.0 * m)
    obj_fun += a_1_j * cvx.norm(b_j) + (a_2_j / 2.0) * cvx.square(cvx.norm(b_j))

    # Build the optimization problem.
    obj = cvx.Minimize(obj_fun)
    problem = cvx.Problem(obj, constraints=None)

    problem.solve(solver="CVXOPT", verbose=False)
    b_j = np.asarray(b_j.value)
    return torch.from_numpy(b_j).to(device, dtype)


def _b2vec(B, groups):
    """Convert B as returned by gel_solve functions to a single numpy vector."""
    d = sum(len(group_j) for group_j in groups)  # the total dimension
    b = np.zeros(d, dtype=B[0, 0].cpu().numpy().dtype)
    for j, group_j in enumerate(groups):
        b[group_j] = B[j, : len(group_j)].cpu().numpy()
    return b


class TestGelBirthwtBase:

    """Base class to test different gel_solve implementations with the birth
    weight data."""

    l_1_base = 4.0
    l_2_base = 0.5

    def __init__(self, device, dtype, *args, **kwargs):
        """Load data and solve with cvx to get ground truth solution."""
        super().__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        dtype_np = torch.rand(0, dtype=dtype).numpy().dtype

        data_dir = os.path.join(os.path.dirname(__file__), "data", "birthwt")
        self.X = np.loadtxt(
            os.path.join(data_dir, "X.csv"), skiprows=1, delimiter=",", dtype=dtype_np
        )
        self.y = np.loadtxt(os.path.join(data_dir, "y.csv"), skiprows=1, dtype=dtype_np)
        self.m = len(self.y)
        self.l_1 = self.l_1_base / (2 * self.m)
        self.l_2 = self.l_2_base / (2 * self.m)
        self.groups = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7],
            [8],
            [9, 10],
            [11],
            [12],
            [13, 14, 15],
        ]
        self.done_setup = False

    def setUp(self):
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise unittest.SkipTest("cuda unavailable")

        if self.done_setup:
            return

        self.ns = torch.tensor([len(g) for g in self.groups])
        self.p = len(self.groups)

        # Convert things to gel format.
        self.As = []
        for j in range(self.p):
            A_j = self.X[:, self.groups[j]]
            self.As.append(torch.from_numpy(A_j).to(self.device, self.dtype))
        self.yt = torch.from_numpy(self.y).to(self.device, self.dtype)

        # Solve with cvx.
        self.b_0_cvx, self.B_cvx = gel_solve_cvx(
            self.As, self.yt, self.l_1, self.l_2, self.ns
        )
        self.b_cvx = _b2vec(self.B_cvx, self.groups)
        self.obj_cvx = self._obj(self.b_0_cvx, self.b_cvx)

        self.done_setup = True

    def _obj(self, b_0, b):
        """Compute the objective function value for the given b_0, b."""
        r = self.y - b_0 - self.X @ b
        g_b = r @ r / (2.0 * self.m)
        b_j_norms = [np.linalg.norm(b[self.groups[j]], ord=2) for j in range(self.p)]
        h_b = self.l_1 * sum(
            np.sqrt(len(self.groups[j])) * b_j_norms[j] for j in range(self.p)
        )
        h_b += self.l_2 * sum(
            np.sqrt(len(self.groups[j])) * (b_j_norms[j] ** 2) for j in range(self.p)
        )
        return g_b + h_b

    def _compare_to_cvx(self, b_0, b, obj):
        """Compare the given solution to the cvx solution."""
        # pylint: disable=no-member
        self.assertAlmostEqual(obj, self.obj_cvx, places=2)
        self.assertAlmostEqual(b_0, self.b_0_cvx, places=2)
        if np.allclose(b, 0) or np.allclose(self.b_cvx, 0):
            for b_i, b_cvx_i in zip(b, self.b_cvx):
                self.assertAlmostEqual(b_i, b_cvx_i, places=2)
        else:
            self.assertAlmostEqual(cosine(b, self.b_cvx), 0, places=2)

    def _test_implementation(self, make_A, gel_solve, **gel_solve_kwargs):
        """Test the given implementation."""
        A = make_A(self.As, self.ns, self.device, self.dtype)
        b_0, B = gel_solve(A, self.yt, self.l_1, self.l_2, self.ns, **gel_solve_kwargs)
        b = _b2vec(B, self.groups)
        obj = self._obj(b_0, b)
        self._compare_to_cvx(b_0, b, obj)

    def test_fista(self):
        """Test the FISTA implementation of gel_solve."""
        self._test_implementation(
            make_A_fista,
            gel_solve_fista,
            t_init=0.1,
            ls_beta=0.9,
            max_iters=1000,
            rel_tol=1e-6,
        )

    def test_cd_cvx(self):
        """Test the CD implementation with cvx internal solver."""
        self._test_implementation(
            make_A_cd,
            gel_solve_cd,
            block_solve_fun=block_solve_cvx,
            block_solve_kwargs={},
            max_cd_iters=100,
            rel_tol=1e-6,
        )

    def test_cd_agd(self):
        """Test the CD implementation with AGD internal solver."""
        self._test_implementation(
            make_A_cd,
            gel_solve_cd,
            block_solve_fun=block_solve_agd,
            block_solve_kwargs={
                "t_init": 1,
                "ls_beta": 0.5,
                "max_iters": 100,
                "rel_tol": 1e-5,
            },
            max_cd_iters=100,
            rel_tol=1e-6,
        )

    def test_cd_newton(self):
        """Test the CD implementation with Newton internal solver."""
        # Compute the C_js and I_js.
        Cs = [(A_j.t() @ A_j) / self.m for A_j in self.As]
        Is = [torch.eye(n_j, device=self.device, dtype=self.dtype) for n_j in self.ns]
        self._test_implementation(
            make_A_cd,
            gel_solve_cd,
            block_solve_fun=block_solve_newton,
            block_solve_kwargs={
                "ls_alpha": 0.1,
                "ls_beta": 0.5,
                "max_iters": 10,
                "tol": 1e-10,
            },
            max_cd_iters=100,
            rel_tol=1e-6,
            Cs=Cs,
            Is=Is,
        )


def create_gel_birthwt_test(device_name, dtype, *mods):
    # I'm so sorry.
    device = torch.device(device_name)

    def __init__(self, *args, **kwargs):
        TestGelBirthwtBase.__init__(self, device, dtype, *args, **kwargs)
        for mod in mods:
            if mod == "l10":
                self.l_1 = 0
            elif mod == "l20":
                self.l_2 = 0
            elif mod == "nj1":
                self.groups = [[i] for i in range(self.X.shape[1])]
            else:
                raise RuntimeError("unrecognized mod: " + mod)

    _doc = "Test gel implementations on {} with {}".format(device_name, dtype)
    if mods:
        _doc += " (mods: " + ", ".join(mods) + ")"
    test_name = "TestGelBirthwt" + device_name.upper() + str(dtype)[-2:]
    if mods:
        test_name += "_" + "".join(str(m) for m in mods)

    globals()[test_name] = type(
        test_name,
        (TestGelBirthwtBase, unittest.TestCase),
        {"__init__": __init__, "__doc__": _doc},
    )


_mods = ["l10", "l20", "nj1"]
_mod_subsets = set(
    frozenset(s) for s in itertools.combinations_with_replacement(_mods, len(_mods))
)
_mod_subsets.add(frozenset())

for _device_name, _dtype, _mod_subset in itertools.product(
    ["cpu", "cuda"], [torch.float32, torch.float64], _mod_subsets
):
    create_gel_birthwt_test(_device_name, _dtype, *list(sorted(_mod_subset)))
