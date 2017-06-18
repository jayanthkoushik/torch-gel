"""gelpaths.py: solution paths for group elastic net.

This module also implements a 2-stage process where group elastic net
is used to find the support, and ridge regression is performed on it.
"""

import torch


def ridge_paths(X, y, support, lambdas, summ_fun):
    """Solve ridge ridgression for a sequence of regularization values,
    and return the score for each.

    The multiple solutions are obtained efficiently using the Woodbury
    identity. The ridge solution is given by

        b = (X.T@X + l*I)^{-1}@X.T@y

    This can be reduced to

        (1/l)*(X.T@y - X.T@V(e + l*I)^{-1}@(X.T@V).T@X.T@y)

    where V@e@V.T is the eigendecomposition of X@X.T. Since (e + l*I) is
    a diagonal matrix, its inverse can be performed efficiently simply by
    taking the reciprocal of the diagonal elements. Then, (X.T@V).T@X.T@y
    is a vector; so it can be multiplied by (e + l*I)^{-1} just by scalar
    multiplication.

    Arguments:
        X: mxp matrix of features.
        y: m-vector of outcomes.
        support: list of features from X to use.
        lambdas: list of regularization values for which to solve the problem.
        summ_fun: a function that takes (support, b) and returns an arbitrary
            summary.

    The funciton returns a dictionary mapping lambda values to their summaries.
    """
    # Setup
    X = X.transpose(0, 1)[torch.LongTensor(support)].transpose(0, 1)
    e, V = torch.symeig(X@X.transpose(0, 1), eigenvectors=True)
    p = X.transpose(0, 1)@y # X.T@y
    Q = X.transpose(0, 1)@V # X.T@V
    r = Q.transpose(0, 1)@p # (X.T@V).T@X.T@y

    # Main loop
    summaries = {}
    for l in lambdas:
        b = (1./l)*(p - Q@(r / (e + l)))
        summaries[l] = summ_fun(support, b)

    return summaries
