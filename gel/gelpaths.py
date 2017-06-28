"""gelpaths.py: solution paths for group elastic net.

This module implements a 2-stage process where group elastic net
is used to find the support, and ridge regression is performed on it.
"""

import sys

import torch
import tqdm


def ridge_paths(X, y, support, lambdas, summ_fun, verbose=False):
    """Solve ridge ridgression for a sequence of regularization values,
    and return the summary for each.

    The multiple solutions are obtained efficiently using the Woodbury
    identity. With X (p x m) representing the feature matrix, and y (m x 1)
    the outcomes, the ridge solution is given by

        b = (X@X.T + l*I)^{-1}@X@y

    where l is the regularization coefficient. This can be reduced to

        (1/l)*(X@y - X@V(e + l*I)^{-1}@(X@V).T@X@y)

    where V@e@V.T is the eigendecomposition of X.T@X. Since (e + l*I) is
    a diagonal matrix, its inverse can be performed efficiently simply by
    taking the reciprocal of the diagonal elements. Then, (X@V).T@X@y
    is a vector; so it can be multiplied by (e + l*I)^{-1} just by scalar
    multiplication.

    Arguments:
        X: pxm FloatTensor of features (where m is the number of samples)
        y: FloatTensor (length m vector) of outcomes.
        support: LongTensor vector of features from X to use. This vector
            should contain indices. These dimensions of X will be used for
            the regression.
        lambdas: list of regularization values for which to solve the problem.
        summ_fun: a function that takes (support, b) and returns an arbitrary
            summary.
        verbose: enable/disable the progress bar.

    The function returns a dictionary mapping lambda values to their summaries.
    """
    # Setup
    X = X[support]
    e, V = torch.symeig(X.t()@X, eigenvectors=True)
    p = X@y # X@y
    Q = X@V # X@V
    r = Q.t()@p # (X@V).T@X@y

    # Main loop
    summaries = {}
    for l in tqdm.tqdm(lambdas, desc="Solving ridge regressions", ncols=80,
                       disable=not verbose):
        b = (1./l)*(p - Q@(r / (e + l)))
        summaries[l] = summ_fun(support, b)

    return summaries


def gel_paths(gel_solve, gel_solve_kwargs, make_A, As, y, l_1s, l_2s, l_rs,
              summ_fun, supp_thresh=1e-6, use_gpu=False, verbose=False):
    """Solve group elastic net to find support and perform ridge on it.

    The problem is solved for multiple values of l_1, l_2, and l_r (the ridge
    regularization), and the result for each is summarized using the given
    summary function.

    Arguments:
        gel_solve, make_A: functions from a gel implementation to be used
            internally.
        gel_solve_kwargs: dictionary of keyword arguments to be passed to
            gel_solve.
        As: list of feature matrices (same as in make_A).
        l_1s, l_2s, l_rs: list of values for l_1, l_2, and l_r respectively.
        summ_fun: function to summarize results (same as in ridge_paths).
        supp_thresh: for computing support, 2-norms below this value are
            considered 0.
        use_gpu: whether or not to use GPU.
        verbose: enable/disable verbosity.

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
    X = X.t() # ridge_paths expects a pxm matrix

    if use_gpu:
        # Move tensors to GPU
        zerot = zerot.cuda()
        B_zeros = B_zeros.cuda()
        ns = ns.cuda()
        sns = sns.cuda()
        A = A.cuda()
        X = X.cuda()
        y = y.cuda()

    summaries = {}
    for l_2 in l_2s:
        b_init = 0., B_zeros # Reset the initial value for each l_2
        for l_1 in l_1s:
            # Solve group elastic net initializing at the previous solution
            b_0, B = gel_solve(A, y, l_1, l_2, ns, b_init, verbose=verbose,
                               **gel_solve_kwargs)
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
            if verbose:
                print("Support size: {}".format(len(support) - 1),
                      file=sys.stderr)

            # Solve ridge on support and store summaries
            ridge_summaries = ridge_paths(X, y, support, l_rs, summ_fun,
                                          verbose)
            for l_r, summary in ridge_summaries.items():
                summaries[(l_1, l_2, l_r)] = summary
            if verbose:
                print("", file=sys.stderr)

    return summaries
