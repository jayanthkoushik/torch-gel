"""gelpaths.py: solution paths for group elastic net.

This module implements a 2-stage process where group elastic net
is used to find the support, and ridge regression is performed on it.
"""

import sys
import math

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
        X: pxm FloatTensor of features (where m is the number of samples);
            X should be centered (each row should have mean 0).
        y: FloatTensor (length m vector) of outcomes.
        support: LongTensor vector of features from X to use. This vector
            should contain indices. These dimensions of X will be used for
            the regression. It can also be None to indicate empty support.
        lambdas: list of regularization values for which to solve the problem.
        summ_fun: a function that takes (support, b) and returns an
            arbitrary summary.
        verbose: enable/disable the progress bar.

    The function returns a dictionary mapping lambda values to their summaries.
    """
    summaries = {}

    if support is None:
        # Nothing to do
        for l in tqdm.tqdm(lambdas, desc="Solving ridge regressions", ncols=80,
                           disable=not verbose):
            summaries[l] = summ_fun(None, None)
        return summaries

    else:
        # Setup
        X = X[support]
        e, V = torch.symeig(X.t()@X, eigenvectors=True)
        p = X@y # X@y
        Q = X@V # X@V
        r = Q.t()@p # (X@V).T@X@y

        # Main loop
        for l in tqdm.tqdm(lambdas, desc="Solving ridge regressions", ncols=80,
                           disable=not verbose):
            b = (1./l)*(p - Q@(r / (e + l)))
            summaries[l] = summ_fun(support, b)

    return summaries


def _find_support(B, ns, supp_thresh):
    """Find features with non-zero coefficients."""
    try:
        support = (B.norm(p=2, dim=1) >= supp_thresh).expand_as(B)
        support = torch.cat([s_j[:n_j] for s_j, n_j in
                             zip(support, ns)])
        return torch.nonzero(support)[:, 0]
    except IndexError:
        return None


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
        As: list of feature matrices (same as in make_A). All features should
            be centered.
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
    p = len(As)
    m = As[0].size()[0]
    ns = torch.LongTensor([A_j.size()[1] for A_j in As])
    B_zeros = torch.zeros(p, ns.max())
    sns = ns.float().sqrt().unsqueeze(1).expand_as(B_zeros)

    # Form the A matrix as needed by gel_solve
    A = make_A(As, ns, use_gpu)

    # Form X which combines all the As
    X = torch.cat(As, dim=1)
    X = X.t() # ridge_paths expects a pxm matrix

    if use_gpu:
        # Move tensors to GPU
        B_zeros = B_zeros.cuda()
        ns = ns.cuda()
        sns = sns.cuda()
        X = X.cuda()
        y = y.cuda()
        if "Cs" in gel_solve_kwargs:
            gel_solve_kwargs["Cs"] = [C_j.cuda() for C_j in
                                      gel_solve_kwargs["Cs"]]
        if "Is" in gel_solve_kwargs:
            gel_solve_kwargs["Is"] = [I_j.cuda() for I_j in
                                      gel_solve_kwargs["Is"]]

    summaries = {}
    for l_2 in l_2s:
        b_init = 0., B_zeros # Reset the initial value for each l_2
        for l_1 in l_1s:
            # Solve group elastic net initializing at the previous solution
            b_0, B = gel_solve(A, y, l_1, l_2, ns, b_init, verbose=verbose,
                               **gel_solve_kwargs)
            b_init = b_0, B

            # Find support
            support = _find_support(B, ns, supp_thresh)
            if verbose:
                support_size = 0 if support is None else len(support)
                print("Support size: {}".format(support_size), file=sys.stderr)

            # Solve ridge on support and store summaries
            ridge_summaries = ridge_paths(X, y, support, l_rs, summ_fun,
                                          verbose)
            for l_r, summary in ridge_summaries.items():
                summaries[(l_1, l_2, l_r)] = summary
            if verbose:
                print("", file=sys.stderr)

    return summaries


def gel_paths2(gel_solve, gel_solve_kwargs, make_A, As, y, ks, n_ls, l_eps,
               l_rs, summ_fun, supp_thresh=1e-6, use_gpu=False, verbose=False):
    """Solve for paths with a reparametrized group elastic net.

    The regularization terms can be rewritten as

        l*(k*||b_j|| + (1 - k)*||b_j||^2)

    where k controls the tradeoff between the two norms,
    and l controls the overall strength. This function takes a list of
    tradeoff values through ks. For each k, an upper bound can be found on l
    (which will lead to an empty support). Using this upper bound,
    n_ls l values are computed on a log scale such that
    l_min / l_max = l_eps. Other arguments are same as in gel_paths.
    """
    # Setup is mostly identical to gel_paths
    p = len(As)
    m = As[0].size()[0]
    ns = torch.LongTensor([A_j.size()[1] for A_j in As])
    B_zeros = torch.zeros(p, ns.max())
    sns = ns.float().sqrt().unsqueeze(1).expand_as(B_zeros)
    A = make_A(As, ns, use_gpu)
    X = torch.cat(As, dim=1)
    X = X.t()

    y_cpu = y
    if use_gpu:
        # Move tensors to GPU
        B_zeros = B_zeros.cuda()
        ns = ns.cuda()
        sns = sns.cuda()
        X = X.cuda()
        y = y.cuda()
        if "Cs" in gel_solve_kwargs:
            gel_solve_kwargs["Cs"] = [C_j.cuda() for C_j in
                                      gel_solve_kwargs["Cs"]]
        if "Is" in gel_solve_kwargs:
            gel_solve_kwargs["Is"] = [I_j.cuda() for I_j in
                                      gel_solve_kwargs["Is"]]

    # The bound is given by max{||A_j.T@(y - b_0)||/(m*sqrt{n_j}*k)}
    # where b_0 = 1.T@y/m.
    # So most things can be precomputed
    l_max_b_0 = y.mean()
    l_max_unscaled = max((A_j.t()@(y_cpu - l_max_b_0)).norm(p=2)/(m*sns_j)
                         for A_j, sns_j in zip(As, sns[:, 0]))

    summaries = {}
    for k in ks:
        b_init = 0., B_zeros # Reset the initial value for each k

        # Get l values
        l_max = l_max_unscaled / k
        l_min = l_max * l_eps
        ls = torch.logspace(math.log10(l_min), math.log10(l_max), steps=n_ls)
        # Put the ls in descending order
        # That way, the support will go from 0 to full,
        # and we can stop when that happens.
        ls = sorted(ls, reverse=True)

        full_support = False
        for l in ls:
            # Convert k, l into l_1, l_2
            l_1, l_2 = k*l, (1.-k)*l

            # Rest is similar to gel_paths
            b_0, B = gel_solve(A, y, l_1, l_2, ns, b_init, verbose=verbose,
                               **gel_solve_kwargs)
            b_init = b_0, B

            # Find support
            support = _find_support(B, ns, supp_thresh)
            support_size = 0 if support is None else len(support)
            if support_size == X.size()[0]:
                full_support = True
            if verbose:
                print("Support size: {}".format(support_size), file=sys.stderr)

            # Solve ridge on support and store summaries
            ridge_summaries = ridge_paths(X, y, support, l_rs, summ_fun,
                                          verbose)
            for l_r, summary in ridge_summaries.items():
                summaries[(k, l, l_r)] = summary
            if verbose:
                print("", file=sys.stderr)

            if full_support:
                break

    return summaries
