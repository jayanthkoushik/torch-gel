"""test_gel.py: framework to test gel implementations."""

import unittest

import torch
import cvxpy as cvx
import numpy as np

from gel.gelfista import gel_solve as gel_solve_fista
from gel.gelfista import make_A as make_A_fista
from gel.gelcd import gel_solve as gel_solve_cd
from gel.gelcd import make_A as make_A_cd
from gel.gelcd import block_solve_agd


def make_A_cvx(As, ns):
    """Convert As to a list of numpy arrays as needed by cvx."""
    return [As_j.numpy() for As_j in As]


def gel_solve_cvx(A, y, l_1, l_2, ns, b_init):
    """Solve a group elastic net problem with cvx."""
    # Convert y and ns to numpy array
    y = y.numpy()
    ns = ns.numpy()

    # Create the b variables
    b_0 = cvx.Variable()
    bs = []
    for A_j, n_j in zip(A, ns):
        bs.append(cvx.Variable(n_j))

    # Form g(b)
    Ab = sum(A_j*b_j for A_j, b_j in zip(A, bs))
    m = A[0].shape[0]
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
    p = len(A)
    B = torch.zeros(p, int(max(ns)))
    for j in range(p):
        b_j = np.asarray(bs[j].value)
        if ns[j] == 1:
            b_j = b_j.reshape(1,)
        B[j, :ns[j]] = torch.FloatTensor(b_j)

    return b_0, B


def block_solve_cvx(r_j, A_j, a_1_j, a_2_j, m, b_j_init):
    """Solve the single block optimization problem from gelcd."""
    # Convert everything to numpy
    r_j = r_j.numpy()
    A_j = A_j.numpy()

    # Create the b_j variable
    b_j = cvx.Variable(A_j.shape[1])

    # Form the objective
    q_j = r_j - A_j*b_j
    obj_fun = cvx.square(cvx.norm2(q_j)) / (2.*m)
    obj_fun += a_1_j*cvx.norm2(b_j) + (a_2_j/2.)*cvx.square(cvx.norm2(b_j))

    # Build the optimization problem
    obj = cvx.Minimize(obj_fun)
    problem = cvx.Problem(obj, constraints=None)

    problem.solve(solver="CVXOPT")
    b_j = np.asarray(b_j.value)
    if A_j.shape[1] == 1:
        b_j = b_j.reshape(1,)
    return torch.FloatTensor(b_j)


class TestGel(unittest.TestCase):

    """Test functions for different gel_solve implementations."""

    def test_gel_birthwt(self):
        """Test with the birth weight data."""
        # Load data
        X = np.loadtxt("data/birthwt/X.csv", skiprows=1, delimiter=",")
        y = np.loadtxt("data/birthwt/y.csv", skiprows=1)
        groups = [[0, 1, 2], [3, 4, 5], [6, 7], [8], [9, 10], [11], [12],
                  [13, 14, 15]]

        # Declare parameters
        p = len(groups)
        m = 189
        l_1 = 4. / (2*m)
        l_2 = 0.5 / (2*m)

        # Covert to gel formats
        As = []
        for j in range(p):
            A_j = X[:, groups[j]]
            As.append(torch.FloatTensor(A_j))
        yt = torch.FloatTensor(y)
        ns = torch.LongTensor([len(g) for g in groups])
        B_zeros = torch.zeros(p, ns.max())
        b_init = 0., B_zeros

        # Declare parameters for the different methods
        method_data = {
            "fista": {
                "make_A": make_A_fista,
                "gel_solve": gel_solve_fista,
                "gel_solve_kwargs": {
                    "t_init": 0.1,
                    "ls_beta": 0.9,
                    "max_iters": 1000,
                    "rel_tol": 0
                }
            },
            "cd_cvx": {
                "make_A": make_A_cd,
                "gel_solve": gel_solve_cd,
                "gel_solve_kwargs": {
                    "block_solve_fun": block_solve_cvx,
                    "block_solve_kwargs": {},
                    "rel_tol": 1e-6
                }
            },
            "cd_agd": {
                "make_A": make_A_cd,
                "gel_solve": gel_solve_cd,
                "gel_solve_kwargs": {
                    "block_solve_fun": block_solve_agd,
                    "block_solve_kwargs": {
                        "t_init": 0.1,
                        "ls_beta": 0.9,
                        "max_iters": None,
                        "tol": 1e-4
                    },
                    "rel_tol": 1e-6
                }
            },
            "cvx": {
                "make_A": make_A_cvx,
                "gel_solve": gel_solve_cvx,
                "gel_solve_kwargs": {}
            }
        }

        # Solve with all methods
        for method in method_data:
            make_A = method_data[method]["make_A"]
            gel_solve = method_data[method]["gel_solve"]
            gel_solve_kwargs = method_data[method]["gel_solve_kwargs"]

            A = make_A(As, ns)
            b_0, B = gel_solve(A, yt, l_1, l_2, ns, b_init, **gel_solve_kwargs)

            # Convert B to a vector
            b = np.zeros(X.shape[1])
            for j in range(p):
                b[groups[j]] = B[j, :len(groups[j])].numpy()

            # Compute objective
            r = y - b_0 - X@b
            g_b = r@r / (2.*m)
            b_j_norms = [np.linalg.norm(b[groups[j]], ord=2) for j in range(p)]
            h_b = l_1 * sum(np.sqrt(len(groups[j]))*b_j_norms[j]
                            for j in range(p))
            h_b += l_2 * sum(np.sqrt(len(groups[j]))*(b_j_norms[j]**2)
                             for j in range(p))

            method_data[method]["obj"] = g_b + h_b
            method_data[method]["b"] = b

        # Compare implemented methods to cvx
        for method in method_data:
            if method == "cvx":
                continue
            self.assertTrue(np.allclose(method_data[method]["obj"],
                                        method_data["cvx"]["obj"]))
            self.assertTrue(np.allclose(method_data[method]["b"],
                                        method_data["cvx"]["b"],
                                        atol=1e-4, rtol=1e-3))
