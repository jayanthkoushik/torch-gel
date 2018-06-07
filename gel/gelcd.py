"""gelcd.py: coordinate descent for group elastic net.

This implementation uses block-wise coordinate descent to solve
the optimization problem:

    min_b (1/2m)||y - b_0 - sum_j A_j@b_j||^2 +
          sum_j {sqrt{n_j}(l_1*||b_j|| + l_2*||b_j||^2)}

The algorithm repeatedly minimizes with respect to b_0 and the b_js.
For b_0, the minimization has a closed form solution. For each b_j,
the minimization objective is convex, twice differentiable; and can
be solved using gradient descent, Newton's method etc.
This implementation allows the internal minimizer to be chosen.
These are the block_solve_* functions. Each solves the above optimization
problem with respect to one of the b_js, while keeping the others fixed.
The gradient with respect to b_j is

    (-1/m)A_j.T@(y - b_0 - sum_j A_j@b_j) + sqrt{n_j}(l_1*b_j/||b_j|| +
                                                      2*l_2*b_j)

and the Hessian is

    (1/m)A_j.T@A_j + sqrt{n_j}(l_1*(I/||b_j|| - b_j@b_j.T/||b_j||^3) + 2*l_2*I)

The coefficients are represented using a scalar bias b_0 and a matrix
B where each row corresponds to a single group (with appropriate 0 padding).
The root of the group sizes are stored in a vector sns.
The features are stored in a list of FloatTensor matrices each of
size m x n_j where m is the number of samples, and n_j is the number of features
in group j. For any j, r_j = y - b_0 - sum_{k =/= j} A_k@b_k,
q_j = r_j - A_j@b_j, C_j = A_j.T@A_j / m, and I_j is an identity matrix of size
n_j. Finally, a_1 = l_1*sns and a_2 = 2*l_2*sns.
"""

import torch
import tqdm


def _f_j(q_j, b_j_norm, a_1_j, a_2_j, m):
    """Compute the objective with respect to one of the coefficients i.e.

        (1/2m)||q_j||^2 + a_1||b_j|| + (a_2/2)||b_j||^2
    """
    return ((q_j @ q_j) / (2. * m)) + (a_1_j * b_j_norm) + (
        (a_2_j / 2.) * (b_j_norm ** 2)
    )


def _grad_j(q_j, A_j, b_j, b_j_norm, a_1_j, a_2_j, m):
    """Compute the gradient with respect to one of the coefficients."""
    return (A_j.t() @ q_j / (-m)) + (b_j * (a_1_j / b_j_norm + a_2_j))


def _hess_j(C_j, I_j, b_j, b_j_norm, a_1_j, a_2_j):
    """Compute the Hessian with respect to one of the coefficients."""
    D_j = torch.ger(b_j, b_j)
    return C_j + (a_1_j / b_j_norm) * (
        I_j - D_j / (b_j_norm ** 2)
    ) + a_2_j * I_j


def block_solve_agd(
    r_j,
    A_j,
    a_1_j,
    a_2_j,
    m,
    b_j_init,
    t_init=None,
    ls_beta=None,
    max_iters=None,
    rel_tol=1e-6,
    verbose=False,
):
    """Solve the optimization problem for a single block with accelerated
    gradient descent."""
    b_j = b_j_init
    b_j_prev = b_j
    k = 1  # Iteration number
    t = 1  # Initial step length (used if t_init is None)
    pbar_stats = {}  # Stats for the progress bar
    pbar = tqdm.tqdm(
        desc="Solving block with AGD", disable=not verbose, leave=False
    )

    while True:
        # Compute the v terms
        mom = (k - 2) / (k + 1.)
        v_j = b_j + mom * (b_j - b_j_prev)
        q_v_j = r_j - A_j @ v_j
        v_j_norm = v_j.norm(p=2)
        f_v_j = _f_j(q_v_j, v_j_norm, a_1_j, a_2_j, m)
        grad_v_j = _grad_j(q_v_j, A_j, v_j, v_j_norm, a_1_j, a_2_j, m)

        b_j_prev = b_j

        # Adjust the step size with backtracking line search
        if t_init is not None:
            t = t_init
        while True:
            b_j = v_j - t * grad_v_j  # Gradient descent update

            if ls_beta is None:
                # Don't perform line search
                break

            # Line search: exit when
            #   f_j(b_j) <= f_j(v_j) + grad_v_j.T@(b_j - v_j) +
            #       (1/2t)||b_j - v_j||^2
            q_b_j = r_j - A_j @ b_j
            b_j_norm = b_j.norm(p=2)
            f_b_j = _f_j(q_b_j, b_j_norm, a_1_j, a_2_j, m)
            b_v_diff = b_j - v_j
            c2 = grad_v_j @ b_v_diff
            c3 = b_v_diff @ b_v_diff / (2. * t)

            if f_b_j <= f_v_j + c2 + c3:
                break
            else:
                t *= ls_beta

        # Make b_j non-zero if it is 0
        if len((b_j.abs() < 1e-6).nonzero()) == len(b_j):
            b_j.fill_(1e-3)
            b_j_norm = b_j.norm(p=2)
        b_diff_norm = (b_j - b_j_prev).norm(p=2)

        pbar_stats["t"] = "{:.2g}".format(t)
        pbar_stats["rel change"] = "{:.2g}".format(b_diff_norm / b_j_norm)
        pbar.set_postfix(pbar_stats)
        pbar.update()

        # Check max iterations exit criterion
        if max_iters is not None and k == max_iters:
            break
        k += 1

        # Check tolerance exit criterion
        # Exit when the relative change is less than the tolerance
        if b_diff_norm <= rel_tol * b_j_norm:
            break

    pbar.close()
    return b_j


def block_solve_newton(
    r_j,
    A_j,
    a_1_j,
    a_2_j,
    m,
    b_j_init,
    C_j,
    I_j,
    ls_alpha=0.5,
    ls_beta=0.9,
    max_iters=20,
    tol=1e-8,
    verbose=False,
):
    """Solve the optimization problem for a single block with
    Newton's method."""
    b_j = b_j_init
    k = 1
    pbar_stats = {}  # Stats for the progress bar
    pbar = tqdm.tqdm(
        desc="Solving block with Newton's method",
        disable=not verbose,
        leave=False,
    )

    while True:
        # First, compute the Newton step and decrement
        q_b_j = r_j - A_j @ b_j
        b_j_norm = b_j.norm(p=2)
        grad_b_j = _grad_j(q_b_j, A_j, b_j, b_j_norm, a_1_j, a_2_j, m)
        hess_b_j = _hess_j(C_j, I_j, b_j, b_j_norm, a_1_j, a_2_j)
        hessinv_b_j = torch.inverse(hess_b_j)
        v_j = hessinv_b_j @ grad_b_j
        dec_j = grad_b_j @ (hessinv_b_j @ grad_b_j)

        # Check tolerance stopping criterion
        # Exit if dec_j / 2 is less than the tolerance
        if dec_j / 2 <= tol:
            break

        # Perform backtracking line search
        t = 1
        f_b_j = _f_j(q_b_j, b_j_norm, a_1_j, a_2_j, m)
        k_j = grad_b_j @ v_j
        while True:
            # Compute the update and evaluate function at that point
            bp_j = b_j - t * v_j
            q_bp_j = r_j - A_j @ bp_j
            bp_j_norm = bp_j.norm(p=2)
            f_bp_j = _f_j(q_bp_j, bp_j_norm, a_1_j, a_2_j, m)

            if f_bp_j <= f_b_j - ls_alpha * t * k_j:
                b_j = bp_j
                break
            else:
                t *= ls_beta

        # Make b_j non-zero if it is 0
        if len((b_j.abs() < 1e-6).nonzero()) == len(b_j):
            b_j.fill_(1e-3)

        pbar_stats["t"] = "{:.2g}".format(t)
        pbar_stats["1/2 newton decrement"] = "{:.2g}".format(dec_j / 2)
        pbar.set_postfix(pbar_stats)
        pbar.update()

        # Check max iterations stopping criterion
        if max_iters is not None and k == max_iters:
            break
        k += 1

    pbar.close()
    return b_j


def make_A(As, ns, device=None):
    """Moves the As to the provided device (or cpu), and returns as A."""
    if device is None:
        device = torch.device("cpu")
    As = [A_j.to(device) for A_j in As]
    return As


def gel_solve(
    A,
    y,
    l_1,
    l_2,
    ns,
    b_init=None,
    block_solve_fun=block_solve_agd,
    block_solve_kwargs={},
    max_cd_iters=None,
    rel_tol=1e-6,
    Cs=None,
    Is=None,
    verbose=False,
):
    """Solve a group elastic net problem.

    Arguments:
        A: list of feature matrices, one per group (size m x n_j).
        y: FloatTensor vector of predictions.
        l_1: the 2-norm coefficient.
        l_2: the squared 2-norm coefficient.
        ns: LongTensor of group sizes.
        b_init: tuple (b_0, B) to initialize b.
        block_solve_fun: the function to be used for minimizing individual
            blocks (should be one of the block_solve_* functions).
        block_solve_kwargs: dictionary of arguments to be passed to
            block_solve_fun.
        max_cd_iters: maximum number of outer CD iterations.
        rel_tol: tolerance for exit criterion; after every CD outer iteration,
            b is compared to the previous value, and if the relative difference
            is less than this value, the loop is terminated.
        Cs, Is: lists of C_j and I_j matrices respectively; used if the internal
            solver is Newton's method.
        verbose: boolean to enable/disable verbosity.
    """
    p = len(A)
    m = len(y)

    # Create initial values if not specified
    if b_init is None:
        b_init = 0., torch.zeros(p, ns.max())

    sns = ns.float().sqrt()
    a_1 = l_1 * sns
    ma_1 = m * a_1
    a_2 = 2 * l_2 * sns
    b_0, B = b_init
    b_0_prev, B_prev = b_0, B
    k = 1  # Iteration number
    pbar_stats = {}  # Stats for the outer progress bar
    pbar = tqdm.tqdm(
        desc="Solving gel with CD (l_1 {:.2g}, l_2 {:.2g})".format(l_1, l_2),
        disable=not verbose,
    )

    while True:
        # First minimize with respect to b_0. This has a closed form solution
        # given by
        #   b_0 = 1.T@(y - sum_j A_j@b_j) / m
        b_0 = (y - sum(A[j] @ B[j, :ns[j]] for j in range(p))).sum() / m

        # Now, minimize with respect to each b_j
        for j in tqdm.trange(
            p,
            desc="Solving individual blocks",
            ncols=80,
            disable=not verbose,
            leave=False,
        ):
            r_j = y - b_0 - sum(A[k] @ B[k, :ns[k]] for k in range(p) if k != j)

            # Check if b_j must be set to 0
            # The condition is
            #   ||A_j.T@r_j|| <= m*a_1
            if (A[j].t() @ r_j).norm(p=2) <= ma_1[j]:
                B[j] = 0
            else:
                # Otherwise, minimize
                # First make sure initial value is not 0
                if len((B[j, :ns[j]].abs() < 1e-6).nonzero()) == ns[j]:
                    B[j, :ns[j]] = 1e-3

                # Add C_j and I_j to the arguments if using Newton's method
                if block_solve_fun is block_solve_newton:
                    block_solve_kwargs["C_j"] = Cs[j]
                    block_solve_kwargs["I_j"] = Is[j]

                B[j, :ns[j]] = block_solve_fun(
                    r_j,
                    A[j],
                    float(a_1[j]),
                    float(a_2[j]),
                    m,
                    B[j, :ns[j]],
                    verbose=verbose,
                    **block_solve_kwargs
                )

        # Compute relative change in b
        b_0_diff = b_0 - b_0_prev
        B_diff = B - B_prev
        delta_norm = (b_0_diff ** 2 + (B_diff ** 2).sum()).sqrt()
        b_norm = (b_0 ** 2 + (B ** 2).sum()).sqrt()

        pbar_stats["rel change"] = "{:.2g}".format(
            delta_norm.item() / b_norm.item()
        )
        pbar.set_postfix(pbar_stats)
        pbar.update()

        # Check max iterations exit criterion
        if max_cd_iters is not None and k == max_cd_iters:
            break
        k += 1

        # Check tolerance exit criterion
        if delta_norm.item() <= rel_tol * b_norm.item():
            break
        b_0_prev, B_prev = b_0, B

    pbar.close()
    return float(b_0), B
