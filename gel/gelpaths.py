"""gelpaths.py: solution paths for group elastic net.

This module implements a 2-stage process where group elastic net is used to find
the support, and ridge regression is performed on it.
"""

import math
import sys

import torch

from gel.ridgepaths import ridge_paths


def _find_support(B, ns, supp_thresh):
    """Find features with non-zero coefficients."""
    ns_cast = ns.unsqueeze(1).to(B.device, B.dtype)
    norms = (B ** 2).sum(dim=1, keepdim=True) / ns_cast
    support = (norms >= supp_thresh).expand_as(B)
    support = torch.cat([s_j[:n_j] for s_j, n_j in zip(support, ns)])
    support = torch.nonzero(support)[:, 0]
    if not support.numel():
        support = None
    return support


def compute_ls_grid(As, y, sns_vec, m, ks, n_ls, l_eps, dtype):
    """Compute l values for each given k and return a dictionary mapping
    k to a list (in decreasing order) of lambda values.

    Arguments have the same meaning as in gel_paths2. sns_vec is a vector of
    sns_j values as opposed to the matrix computed in gel_paths2.
    """
    ls_grid = {}
    # The bound is given by max{||A_j'@(y - b_0)||/(m*sqrt{n_j}*k)} where b_0 =
    # 1'@y/m. So most things can be precomputed.
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
                math.log10(l_min), math.log10(l_max), steps=n_ls, dtype=dtype
            )
            ls = sorted([l.item() for l in ls], reverse=True)
            ls_grid[k] = ls
    return ls_grid


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
    device=torch.device("cpu"),
    verbose=False,
    aux_rel_tol=1e-3,
    dtype=torch.float32,
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
        As: list of feature matrices (same as in make_A). All features should be
            centered.
        y: vector of predictions.
        l_1s, l_2s, l_rs: list of values for l_1, l_2, and l_r respectively.
        summ_fun: function to summarize results (same as in ridge_paths).
        supp_thresh: for computing support, 2-norms below this value are
            considered 0.
        device: torch device.
        verbose: enable/disable verbosity.
        aux_rel_tol: relative tolerance for solving auxiliary problems.
        dtype: torch dtype.

    The function returns a dictionary mapping (l_1, l_2, l_r) values to their
    summaries.
    """
    # We have to loop over l_1s, l_2s, and l_rs. Since multiple ridge
    # regressions can be performed efficiently with ridge_paths, l_rs should be
    # the inner most loop. Further, with l_1^1 > l_1^2 > ..., group elastic net
    # for (l_1^j, l_2) can be solved efficiently by warm starting at the
    # solution for (l_1^{j-1}, l_2). So, l_2s should be the outer most loop.

    # First compute various required variables.
    l_1s = sorted(l_1s, reverse=True)
    p = len(As)
    m = As[0].shape[0]
    As = [A_j.to(device, dtype) for A_j in As]
    ns = torch.tensor([A_j.shape[1] for A_j in As])
    B_zeros = torch.zeros(p, ns.max().item(), device=device, dtype=dtype)
    sns = ns.to(device, dtype).sqrt().unsqueeze(1).expand_as(B_zeros)

    # Form the A matrix as needed by gel_solve.
    A = make_A(As, ns, device, dtype)

    # Form X which combines all the As (for ridge_paths).
    X = torch.cat(As, dim=1)
    X = X.t()  # ridge_paths expects a pxm matrix

    y = y.to(device, dtype)
    if "Cs" in gel_solve_kwargs:
        gel_solve_kwargs["Cs"] = [
            C_j.to(device, dtype) for C_j in gel_solve_kwargs["Cs"]
        ]
    if "Is" in gel_solve_kwargs:
        gel_solve_kwargs["Is"] = [
            I_j.to(device, dtype) for I_j in gel_solve_kwargs["Is"]
        ]

    gel_solve_kwargs_aux = gel_solve_kwargs.copy()
    gel_solve_kwargs_aux["rel_tol"] = aux_rel_tol
    summaries = {}

    for l_2 in l_2s:
        b_init = 0.0, B_zeros  # reset b_init for each l_2

        # Find a good initialization to solve for the first (l_1, l_2) pair.
        if l_2 != 0 and l_1s[0] != 0:
            # Find k corresponding to the l_1, l_2.
            # l_1 = k*l, and l_2 = (1 - k)*l.
            k = 1.0 / (1.0 + l_2 / l_1s[0])

            # Compute l_max corresponding to k.
            l_aux = compute_ls_grid(As, y, sns[:, 0], m, [k], 1, None, dtype)[k]
            l_aux = l_aux[0]

            # Solve with k, l_aux.
            if verbose:
                print(
                    "Solving auxiliary problem to get good initialization",
                    file=sys.stderr,
                )
            b_init = gel_solve(
                A,
                y,
                k * l_aux,
                (1.0 - k) * l_aux,
                ns,
                b_init,
                verbose=verbose,
                **gel_solve_kwargs,
            )
            if verbose:
                print("Done solving auxiliary problem", file=sys.stderr)

        for l_1 in l_1s:
            # Solve group elastic net initializing at the previous solution.
            b_0, B = gel_solve(
                A, y, l_1, l_2, ns, b_init, verbose=verbose, **gel_solve_kwargs
            )
            b_init = b_0, B

            # Find support.
            support = _find_support(B, ns, supp_thresh)
            support_size = 0 if support is None else len(support)
            if verbose:
                print("Support size: {}".format(support_size), file=sys.stderr)

            # Solve ridge on support and store summaries.
            if support is not None:
                X_supp = X if support_size == X.shape[0] else X[support]
            else:
                X_supp = None
            ridge_summaries = ridge_paths(X_supp, y, support, l_rs, summ_fun, verbose)
            del X_supp
            for l_r, summary in ridge_summaries.items():
                summaries[(l_1, l_2, l_r)] = summary
            if verbose:
                print(file=sys.stderr)

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
    device=torch.device("cpu"),
    verbose=False,
    ls_grid=None,
    aux_rel_tol=1e-3,
    dtype=torch.float32,
):
    """Solve for paths with a reparametrized group elastic net.

    Arguments:
        gel_solve, make_A: functions from a gel implementation to be used
            internally.
        gel_solve_kwargs: dictionary of keyword arguments to be passed to
            gel_solve.
        As: list of feature matrices (same as in make_A). All features should be
            centered.
        y: vector of predictions.
        ks: list of values for trading-off l1 and l2 regularizations.
        n_ls: number of lambda values.
        l_eps: ratio of minimum to maximum lambda value.
        l_rs: list of ridge regularization values.
        summ_fun: function to summarize results (same as in ridge_paths).
        supp_thresh: for computing support, 2-norms below this value are
            considered 0.
        device: torch device.
        verbose: enable/disable verbosity.
        ls_grid: pre-computed dictionary mapping each k in ks to a list of l
            values (in decreasing order). If this argument is not None, n_ls and
            l_eps are ignored.
        aux_rel_tol: relative tolerance for solving auxiliary problems.
        dtype: torch dtype.

    The regularization terms can be rewritten as

        l*(k*||b_j|| + (1 - k)*||b_j||^2)

    where k controls the tradeoff between the two norms, and l controls the
    overall strength. This function takes a list of tradeoff values through ks.
    For each k, an upper bound can be found on l (which will lead to an empty
    support). Using this upper bound, n_ls l values are computed on a log scale
    such that l_min / l_max = l_eps. Other arguments are same as in gel_paths.

    The function returns a dictionary mapping (l_1, l_2, l_r) values to their
    summaries.
    """
    # Setup is mostly identical to gel_paths.
    p = len(As)
    m = As[0].shape[0]
    As = [A_j.to(device, dtype) for A_j in As]
    ns = torch.tensor([A_j.shape[1] for A_j in As])
    B_zeros = torch.zeros(p, ns.max(), device=device, dtype=dtype)
    sns = ns.to(device, dtype).sqrt().unsqueeze(1).expand_as(B_zeros)
    A = make_A(As, ns, device, dtype)
    X = torch.cat(As, dim=1)
    X = X.t()

    y = y.to(device, dtype)
    if "Cs" in gel_solve_kwargs:
        gel_solve_kwargs["Cs"] = [C_j.to(device) for C_j in gel_solve_kwargs["Cs"]]
    if "Is" in gel_solve_kwargs:
        gel_solve_kwargs["Is"] = [I_j.to(device) for I_j in gel_solve_kwargs["Is"]]

    if ls_grid is None:
        ls_grid = compute_ls_grid(As, y, sns[:, 0], m, ks, n_ls, l_eps, dtype)
        ls_grid_self = None
    else:
        # Compute l values with self data to get good initializations.
        ls_grid_self = compute_ls_grid(As, y, sns[:, 0], m, ks, n_ls, l_eps, dtype)
        gel_solve_kwargs_aux = gel_solve_kwargs.copy()
        gel_solve_kwargs_aux["rel_tol"] = aux_rel_tol

    summaries = {}
    for k in ks:
        b_init = 0.0, B_zeros  # reset the initial value for each k
        ls = ls_grid[k]
        full_support = False

        if ls_grid_self is not None:
            # Solve with self l values to get a better initialization.
            # This _greatly_ speeds up the optimization.
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
                    (1.0 - k) * l_self,
                    ns,
                    b_init,
                    verbose=verbose,
                    **gel_solve_kwargs_aux,
                )
            if verbose:
                print("Done solving auxiliary problems", file=sys.stderr)

        ridge_summaries = None
        for l in ls:
            # Convert k, l into l_1, l_2.
            l_1, l_2 = k * l, (1.0 - k) * l

            if full_support:
                # Just copy the previous summaries.
                for l_r, summary in ridge_summaries.items():
                    summaries[(l_1, l_2, l_r)] = summary
                continue

            # Rest is similar to gel_paths.
            b_0, B = gel_solve(
                A, y, l_1, l_2, ns, b_init, verbose=verbose, **gel_solve_kwargs
            )
            b_init = b_0, B

            # Find support.
            support = _find_support(B, ns, supp_thresh)
            support_size = 0 if support is None else len(support)
            full_support = support_size == X.shape[0]
            if verbose:
                print("Support size: {}".format(support_size), file=sys.stderr)

            # Solve ridge on support and store summaries.
            if support is not None:
                X_supp = X if full_support else X[support]
            else:
                X_supp = None
            ridge_summaries = ridge_paths(X_supp, y, support, l_rs, summ_fun, verbose)
            del X_supp
            for l_r, summary in ridge_summaries.items():
                summaries[(l_1, l_2, l_r)] = summary
            if verbose:
                print(file=sys.stderr)

    return summaries
