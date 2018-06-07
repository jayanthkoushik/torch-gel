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
        support: LongTensor vector of features to use. This vector
            should contain indices. It can also be None to indicate empty
            support. X should already be indexed using support.
            This argument is simply passed to the summary function.
        lambdas: list of regularization values for which to solve the problem.
        summ_fun: a function that takes (support, b) and returns an
            arbitrary summary.
        verbose: enable/disable the progress bar.

    The function returns a dictionary mapping lambda values to their summaries.
    """
    summaries = {}

    if support is None:
        # Nothing to do
        for l in tqdm.tqdm(
            lambdas,
            desc="Solving ridge regressions",
            ncols=80,
            disable=not verbose,
        ):
            summaries[l] = summ_fun(None, None)
        return summaries

    else:
        # Setup
        e, V = torch.symeig(X.t() @ X, eigenvectors=True)
        p = X @ y  # X@y
        Q = X @ V  # X@V
        r = Q.t() @ p  # (X@V).T@X@y

        # Main loop
        for l in tqdm.tqdm(
            lambdas,
            desc="Solving ridge regressions",
            ncols=80,
            disable=not verbose,
        ):
            b = (1. / l) * (p - Q @ (r / (e + l)))
            summaries[l] = summ_fun(support, b)

    return summaries


def _find_support(B, ns, supp_thresh):
    """Find features with non-zero coefficients."""
    try:
        support = (B.norm(p=2, dim=1, keepdim=True) >= supp_thresh).expand_as(B)
        support = torch.cat([s_j[:n_j] for s_j, n_j in zip(support, ns)])
        return torch.nonzero(support)[:, 0]
    except IndexError:
        return None


def gel_paths(
    gel_solve,
    gel_solve_kwargs,
    make_A,
    As,
    y,
    l_1s,
    l_2s,
    l_rs,
    summ_fun,
    supp_thresh=1e-6,
    device=None,
    verbose=False,
    aux_rel_tol=1e-3,
):
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
        device: torch device (default cpu)
        verbose: enable/disable verbosity.
        aux_rel_tol: relative tolerance for solving auxiliary problems.

    The function returns a dictionary mapping (l_1, l_2, l_r) values to their
    summaries.
    """
    # We have to loop over l_1s, l_2s, and l_rs. Since multiple ridge
    # regressions can be performed efficiently with ridge_paths, l_rs should
    # be the inner most loop. Further, with l_1^1 > l_1^2 > ..., group elastic
    # net for (l_1^j, l_2) can be solved efficiently by warm starting at the
    # solution for (l_1^{j-1}, l_2). So, l_2s should be the outer most loop.

    # First compute various required variables
    if device is None:
        device = torch.device("cpu")
    l_1s = sorted(l_1s, reverse=True)
    p = len(As)
    m = As[0].size()[0]
    ns = torch.tensor([A_j.size()[1] for A_j in As], device=device)
    B_zeros = torch.zeros(p, ns.max(), device=device)
    sns = ns.float().sqrt().unsqueeze(1).expand_as(B_zeros)

    # Form the A matrix as needed by gel_solve
    A = make_A(As, ns, device)

    # Form X which combines all the As
    X = torch.cat(As, dim=1)
    X = X.t()  # ridge_paths expects a pxm matrix

    y_cp = y.to(X.device)  # Required for compute_ls_grid
    y = y.to(device)
    if "Cs" in gel_solve_kwargs:
        gel_solve_kwargs["Cs"] = [
            C_j.to(device) for C_j in gel_solve_kwargs["Cs"]
        ]
    if "Is" in gel_solve_kwargs:
        gel_solve_kwargs["Is"] = [
            I_j.to(device) for I_j in gel_solve_kwargs["Is"]
        ]

    gel_solve_kwargs_aux = gel_solve_kwargs.copy()
    gel_solve_kwargs_aux["rel_tol"] = aux_rel_tol
    summaries = {}

    for l_2 in l_2s:
        # Find a good initialization to solve for the first (l_1, l_2) pair
        if l_1s[0] != 0:
            # Find k corresponding to the l_1, l_2
            # l_1 = k*l, and l_2 = (1 - k)*l
            k = 1. / (1. + l_2 / l_1s[0])

            # Compute l_max corresponding to k
            l_aux = compute_ls_grid(As, y_cp, sns[:, 0], m, [k], 1, None)[k][0]

            # Solve with k, l_aux
            if verbose:
                print(
                    "Solving auxiliary problem to get good initialization",
                    file=sys.stderr,
                )
            b_init = gel_solve(
                A,
                y,
                k * l_aux,
                (1. - k) * l_aux,
                ns,
                (0., B_zeros),
                verbose=verbose,
                **gel_solve_kwargs
            )
            if verbose:
                print("Done solving auxiliary problem", file=sys.stderr)

        for l_1 in l_1s:
            # Solve group elastic net initializing at the previous solution
            b_0, B = gel_solve(
                A, y, l_1, l_2, ns, b_init, verbose=verbose, **gel_solve_kwargs
            )
            b_init = b_0, B

            # Find support
            support = _find_support(B, ns, supp_thresh)
            if verbose:
                support_size = 0 if support is None else len(support)
                print("Support size: {}".format(support_size), file=sys.stderr)

            # Solve ridge on support and store summaries
            if support is not None:
                X_supp = X[support.to(X.device)].to(device)
            else:
                X_supp = None
            ridge_summaries = ridge_paths(
                X_supp, y, support, l_rs, summ_fun, verbose
            )
            del X_supp
            for l_r, summary in ridge_summaries.items():
                summaries[(l_1, l_2, l_r)] = summary
            if verbose:
                print("", file=sys.stderr)

    return summaries


def gel_paths2(
    gel_solve,
    gel_solve_kwargs,
    make_A,
    As,
    y,
    ks,
    n_ls,
    l_eps,
    l_rs,
    summ_fun,
    supp_thresh=1e-6,
    device=None,
    verbose=False,
    ls_grid=None,
    aux_rel_tol=1e-3,
):
    """Solve for paths with a reparametrized group elastic net.

    The regularization terms can be rewritten as

        l*(k*||b_j|| + (1 - k)*||b_j||^2)

    where k controls the tradeoff between the two norms,
    and l controls the overall strength. This function takes a list of
    tradeoff values through ks. For each k, an upper bound can be found on l
    (which will lead to an empty support). Using this upper bound,
    n_ls l values are computed on a log scale such that
    l_min / l_max = l_eps. Other arguments are same as in gel_paths.

    ls_grid is a pre-computed dictionary mapping each k in ks to a list of
    l values (in decreasing order). If this argument is not None, n_ls and
    l_eps are ignored.

    aux_rel_tol is the relative tolerance for solving auxiliary problems.
    """
    # Setup is mostly identical to gel_paths
    if device is None:
        device = torch.device("cpu")
    p = len(As)
    m = As[0].size()[0]
    ns = torch.LongTensor([A_j.size()[1] for A_j in As], device=device)
    B_zeros = torch.zeros(p, ns.max(), device=device)
    sns = ns.float().sqrt().unsqueeze(1).expand_as(B_zeros)
    A = make_A(As, ns, device)
    X = torch.cat(As, dim=1)
    X = X.t()

    y_cp = y.to(X.device)
    y = y.to(device)
    if "Cs" in gel_solve_kwargs:
        gel_solve_kwargs["Cs"] = [
            C_j.to(device) for C_j in gel_solve_kwargs["Cs"]
        ]
    if "Is" in gel_solve_kwargs:
        gel_solve_kwargs["Is"] = [
            I_j.to(device) for I_j in gel_solve_kwargs["Is"]
        ]

    if ls_grid is None:
        ls_grid = compute_ls_grid(As, y_cp, sns[:, 0], m, ks, n_ls, l_eps)
        ls_grid_self = None
    else:
        # Compute l values with self data to get good initializations
        ls_grid_self = compute_ls_grid(As, y_cp, sns[:, 0], m, ks, n_ls, l_eps)
        gel_solve_kwargs_aux = gel_solve_kwargs.copy()
        gel_solve_kwargs_aux["rel_tol"] = aux_rel_tol

    summaries = {}
    for k in ks:
        b_init = 0., B_zeros  # Reset the initial value for each k
        ls = ls_grid[k]
        full_support = False

        if ls_grid_self is not None:
            # Solve with self l values to get a better initialization
            # This _greatly_ speeds up the optimization
            if verbose:
                print(
                    "Solving auxiliary problems to get good initialization",
                    file=sys.stderr,
                )
            ls_self = ls_grid_self[k]
            for l_self in ls_self:
                if l_self < ls[0]:
                    break
                b_init = gel_solve(
                    A,
                    y,
                    k * l_self,
                    (1. - k) * l_self,
                    ns,
                    b_init,
                    verbose=verbose,
                    **gel_solve_kwargs_aux
                )
            if verbose:
                print("Done solving auxiliary problems", file=sys.stderr)

        ridge_summaries = None
        for l in ls:
            if full_support:
                # Just copy the previous summaries
                for l_r, summary in ridge_summaries.items():
                    summaries[(k, l, l_r)] = summary
                continue

            # Convert k, l into l_1, l_2
            l_1, l_2 = k * l, (1. - k) * l

            # Rest is similar to gel_paths
            b_0, B = gel_solve(
                A, y, l_1, l_2, ns, b_init, verbose=verbose, **gel_solve_kwargs
            )
            b_init = b_0, B

            # Find support
            support = _find_support(B, ns, supp_thresh)
            support_size = 0 if support is None else len(support)
            if support_size == X.size()[0]:
                full_support = True
            if verbose:
                print("Support size: {}".format(support_size), file=sys.stderr)

            # Solve ridge on support and store summaries
            if support is not None:
                X_supp = X[support.to(X.device)].to(device)
            else:
                X_supp = None
            ridge_summaries = ridge_paths(
                X_supp, y, support, l_rs, summ_fun, verbose
            )
            del X_supp
            for l_r, summary in ridge_summaries.items():
                summaries[(k, l, l_r)] = summary
            if verbose:
                print("", file=sys.stderr)

    return summaries


def compute_ls_grid(As, y, sns_vec, m, ks, n_ls, l_eps):
    """Compute l values for each given k and return a dictionary mapping
    k to a list (in decreasing order) of lambda values.

    Arguments have the same meaning as in gel_paths2. sns_vec is a vector of
    sns_j values as opposed to the matrix computed in gel_paths2.
    """
    ls_grid = {}
    # The bound is given by max{||A_j.T@(y - b_0)||/(m*sqrt{n_j}*k)}
    # where b_0 = 1.T@y/m.
    # So most things can be precomputed
    l_max_b_0 = y.mean()
    l_max_unscaled = max(
        (A_j.t() @ (y - l_max_b_0)).norm(p=2) / (m * sns_j)
        for A_j, sns_j in zip(As, sns_vec)
    )
    for k in ks:
        l_max = l_max_unscaled / k
        if n_ls == 1:
            ls_grid[k] = [l_max]
        else:
            l_min = l_max * l_eps
            ls = torch.logspace(
                math.log10(l_min), math.log10(l_max), steps=n_ls
            )
            ls = sorted(ls, reverse=True)
            ls_grid[k] = ls
    return ls_grid
