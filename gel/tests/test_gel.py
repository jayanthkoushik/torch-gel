"""test_gel.py: framework to test gel implementations."""

import unittest

import torch
import cvxpy as cvx
import numpy as np

from gel.gelfista import gel_solve as gel_solve_fista
from gel.gelfista import make_A as make_A_fista


def gel_solve_cvx(As, y, l_1, l_2):
    """Solve a group elastic net problem with cvx."""
    # Convert torch tensors to numpy arrays
    As = [As_j.numpy() for As_j in As]
    y = y.numpy()

    # Create the b variables
    b_0 = cvx.Variable()
    bs = []
    ns = []
    for A_j in As:
        n_j = A_j.shape[1]
        ns.append(n_j)
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
    B = torch.zeros(p, max(ns))
    for j in range(p):
        b_j = np.asarray(bs[j].value)
        if ns[j] == 1:
            b_j = b_j.reshape(1,)
        B[j, :ns[j]] = torch.FloatTensor(b_j)

    return b_0, B


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

        # Covert to gel format
        As = []
        for j in range(p):
            A_j = X[:, groups[j]]
            As.append(torch.FloatTensor(A_j))
        yt = torch.FloatTensor(y)
        ns = torch.LongTensor([len(g) for g in groups])
        B_zeros = torch.zeros(p, ns.max())
        b_init = 0., B_zeros
        sns = ns.float().sqrt().unsqueeze(1).expand_as(B_zeros)
        A = make_A_fista(As, ns)

        # Solve with both methods
        for method in ["self", "cvx"]:
            if method == "self":
                b_0, B = gel_solve_fista(A, yt, l_1, l_2, sns, b_init,
                                         t_init=0.1, ls_beta=0.9,
                                         max_iters=1000, rel_tol=0)
            else:
                b_0, B = gel_solve_cvx(As, yt, l_1, l_2)

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
            if method == "self":
                self_obj = g_b + h_b
                self_b = b
            else:
                cvx_obj = g_b + h_b
                cvx_b = b

        # Compare self to ground truth
        self.assertTrue(np.allclose(self_obj, cvx_obj))
        self.assertTrue(np.allclose(self_b, cvx_b, atol=1e-5, rtol=1e-3))
