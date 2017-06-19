"""gelpaths.py: solution paths for group elastic net.

This module also implements a 2-stage process where group elastic net
is used to find the support, and ridge regression is performed on it.
"""

import torch

from gel.gel import gel_solve, make_A


def ridge_paths(X, y, support, lambdas, summ_fun):
    """Solve ridge ridgression for a sequence of regularization values,
    and return the summary for each.

    The multiple solutions are obtained efficiently using the Woodbury
    identity. The ridge solution is given by

        b = (X@X.T + l*I)^{-1}@X@y

    This can be reduced to

        (1/l)*(X@y - X@V(e + l*I)^{-1}@(X@V).T@X@y)

    where V@e@V.T is the eigendecomposition of X.T@X. Since (e + l*I) is
    a diagonal matrix, its inverse can be performed efficiently simply by
    taking the reciprocal of the diagonal elements. Then, (X@V).T@X@y
    is a vector; so it can be multiplied by (e + l*I)^{-1} just by scalar
    multiplication.

    Arguments:
        X: pxm matrix of features (where m is the number of samples)
        y: m-vector of outcomes.
        support: LongTensor of features from X to use.
        lambdas: list of regularization values for which to solve the problem.
        summ_fun: a function that takes (support, b) and returns an arbitrary
            summary.

    The funciton returns a dictionary mapping lambda values to their summaries.
    """
    # Setup
    X = X[support]
    e, V = torch.symeig(X.transpose(0, 1)@X, eigenvectors=True)
    p = X@y # X@y
    Q = X@V # X@V
    r = Q.transpose(0, 1)@p # (X@V).T@X@y

    # Main loop
    summaries = {}
    for l in lambdas:
        b = (1./l)*(p - Q@(r / (e + l)))
        summaries[l] = summ_fun(support, b)

    return summaries


def gel_paths(As, y, l_1s, l_2s, l_rs, summ_fun, supp_thresh=1e-6,
              use_gpu=False, t_init=None, ls_beta=None, max_iters=None,
              rel_tol=1e-6):
    """Solve group elastic net to find support and perform ridge on it.

    The problem is solved for multiple values of l_1, l_2, and l_r (the ridge
    regularization), and the result for each is summarized using the given
    summary function.

    Arguments:
        As: list of feature matrices (same as in make_A).
        l_1s, l_2s, l_rs: list of values for l_1, l_2, and l_r respectively.
        summ_fun: function to summarize results (same as in ridge_paths).
        supp_thresh: for computing support, 2-norms below this value are
            considered 0.
        use_gpu: whether or not to use GPU.
        All other arguments are the same as in gel_solve.

    The function returns a dictionary mapping (l_1, l_2, l_r) values to their
    summaries.
    """
    # We have to loop over l_1s, l_2s, and l_rs. Since multiple ridge
    # regressions can be performed efficiently with ridge_paths, l_rs should
    # be the inner most loop. Further, with l_1^1 > l_1^2 > ..., group elastic
    # net for (l_1^j, l_2) can be solved efficiently by warm starting at the
    # solution for (l_1^{j-1}, l_2). So, l_2s should be the outer most loop.

    # First compute various required variables
    l_1s = sorted(l_1s, reverse=True)
    zerot = torch.LongTensor([0])
    p = len(As)
    m = As[0].size()[0]
    ns = torch.LongTensor([A_j.size()[1] for A_j in As])
    B_zeros = torch.zeros(p, ns.max())
    sns = ns.float().sqrt().unsqueeze(1).expand_as(B_zeros)

    # Form the A matrix as needed by gel_solve
    A = make_A(As, ns)

    # Form X which combines all the As and a column of 1s for the bias
    X = torch.cat([torch.ones(m, 1)] + As, dim=1)
    X = X.transpose(0, 1) # ridge_paths expects a pxm matrix

    if use_gpu:
        # Move tensors to GPU
        zerot = zerot.cuda()
        B_zeros = B_zeros.cuda()
        ns = ns.cuda()
        sns = sns.cuda()
        A = A.cuda()
        X = X.cuda()

    summaries = {}
    for l_2 in l_2s:
        b_init = 0., B_zeros # Reset the initial value for each l_2
        for l_1 in l_1s:
            # Solve group elastic net initializing at the previous solution
            b_0, B = gel_solve(A, y, l_1, l_2, m, p, sns, b_init, t_init,
                               ls_beta, max_iters, rel_tol)
            b_init = b_0, B

            # Find support
            try:
                support = (B.norm(p=2, dim=1) >= supp_thresh).expand_as(B)
                support = torch.cat([s_j[:n_j] for s_j, n_j in
                                     zip(support, ns)])
                support = torch.nonzero(support)[:, 0]
                # The numbers above have to be shifted by 1, and the bias index
                # needs to be added to the list
                support = support + 1
                support = torch.cat([zerot, support])
            except IndexError:
                # Empty support; only include bias
                support = zerot

            # Solve ridge on support and store summaries
            ridge_summaries = ridge_paths(X, y, support, l_rs, summ_fun)
            for l_r, summary in ridge_summaries.values():
                summaries[(l_1, l_2, l_r)] = summary

    return summaries
