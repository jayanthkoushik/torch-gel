"""gelfista.py: FISTA like algorithm for group elastic net.

This implementation uses accelerated proximal gradient descent to
solve the optimization problem:

    min_b (1/2m)||y - b_0 - sum_j A_j@b_j||^2 +
          sum_j {sqrt{n_j}(l_1*||b_j|| + l_2*||b_j||^2)}

Here, and everywhere else, the "@" symbol is used to denote matrix
multiplication. Denoting the first part of the objective by g,
and the second part by h, the algorithm can be written as

    v = b^{k-1} + ((k-2)/(k-1))*(b^{k-1} - b^{k-2})
    b^k = prox_{t_k}(v - t_k*grad(g(v)))

Here t_k is a step length chosen by backtracking line search,
and prox is the proximal operator of h.

The functions in this module use certain variables and conventions.
These will be defined next.

The coefficients are represented using a scalar bias b_0 and a matrix
B where each row corresponds to a single group (with appropriate 0 padding).
The root of the group sizes are stored in a matrix sns broadcasted to the
shape of B. The features are stored in a 3D tensor A of size
#groups x max{n_j} x m. Two helper variables are used: a_1 = l_1*sns,
and a_2 = 2*l_2*sns.
"""

import torch
import tqdm


def _prox(B, a_1, a_2, t):
    """Compute the prox operator of h.

    In this case, this reduces to a simple soft thresholding:

        prox_t(x)_0 = x_0
        prox_t(x)_j =
            if ||x_j|| > t*l_1*sqrt(n_j):
                (x_j/||x_j||)*((||x_j|| - t*l_1*sqrt(n_j)) /
                               (1 + 2*t*l_2*sqrt(n_j)))
            else:
                0

    Here x_j denotes the coefficient corresponding to the j^th group.

    Arguments have usual meanings. b_0 is not
    passed to the function since it is not changed.
    """
    # First compute assuming 'if' part of the condition
    ta_1 = t * a_1
    norms = B.norm(p=2, dim=1, keepdim=True).expand_as(B)
    shrinkage = (norms - ta_1) / (1 + t * a_2)
    prox = B * shrinkage / norms

    # Threshold to zero if ||x_j|| is below threshold
    thresh_cond = norms <= ta_1
    prox[thresh_cond] = 0
    return prox


def _r(A, b_0, B, y):
    """Compute y - b_0 - sum_j A_j@b_j."""
    # bmm will return a tensor with individual A_j@b_j products
    AB = torch.bmm(A.permute(0, 2, 1), B.unsqueeze(2)).sum(dim=0).squeeze()
    return y - b_0 - AB


def _g(A, b_0, B, y, m):
    """Compute g(b)."""
    # m is the number of samples
    r = _r(A, b_0, B, y)
    return r @ r / (2. * m)


def _grad(A, b_0, B, y, p, m):
    """Compute the gradient of g.

    The gradient for b_0 is 1.T@r / -m, and the gradient for b_j is
    A_j.T@r / -m.
    """
    r = _r(A, b_0, B, y)
    grad_b_0 = r.sum() / -m
    # r must be broadcasted for the next operation
    r = r.unsqueeze(0).unsqueeze(2).expand(p, m, 1)
    grad_B = torch.bmm(A, r).squeeze() / -m
    return grad_b_0, grad_B


def make_A(As, ns, device=None):
    """Create the 3D tensor A as needed by gel_solve, given a list of feature
        matrices.

    Arguments:
        As: list of feature matrices, one per group (size mxn_j).
        ns: LongTensor of group sizes.
        device: torch device (default cpu)
    """
    A = torch.zeros(len(ns), ns.max(), As[0].size()[0])
    for j, n_j in enumerate(ns):
        # Fill A[j] with A_j.T
        A_j = As[j]
        A[j, :n_j, :] = A_j.t()
    if device is None:
        device = torch.device("cpu")
    A = A.to(device)
    return A


def gel_solve(
    A,
    y,
    l_1,
    l_2,
    ns,
    b_init=None,
    t_init=None,
    ls_beta=None,
    max_iters=None,
    rel_tol=1e-6,
    verbose=False,
):
    """Solve a group elastic net problem.

    Arguments:
        A: 3D FloatTensor of features as described in the header,
            returned by make_A.
        y: FloatTensor vector of predictions.
        l_1: the 2-norm coefficient.
        l_2: the squared 2-norm coefficient.
        ns: LongTensor of group sizes.
        b_init: tuple (b_0, B) to initialize b.
        t_init: initial step size to start line search; if None, it's set to
            the value from the previous iteration.
        ls_beta: shrinkage factor for line search; if None, no line search is
            performed, and t_init is used as the step size for every iteration.
        max_iters: maximum number of iterations; if None, there is no limit.
        rel_tol: tolerance for exit criterion.
        verbose: boolean to enable/disable verbosity.
    """
    p = len(ns)
    m = len(y)

    # Create initial values if not specified
    if b_init is None:
        b_init = 0., torch.zeros(p, ns.max())

    b_0, B = b_init
    b_0_prev, B_prev = b_0, B
    sns = ns.float().sqrt().unsqueeze(1).expand_as(B)
    a_1 = l_1 * sns
    a_2 = 2 * l_2 * sns
    k = 1  # Iteration number
    t = 1  # Initial step length (used if t_init is None)
    pbar_stats = {}  # Stats for the progress bar
    pbar = tqdm.tqdm(
        desc="Solving gel with FISTA (l_1 {:.2g}, l_2 {:.2g})".format(l_1, l_2),
        disable=not verbose,
    )

    while True:
        # Compute the v terms
        mom = (k - 2) / (k + 1.)
        v_0 = b_0 + mom * (b_0 - b_0_prev)
        V = B + mom * (B - B_prev)
        g_v = _g(A, v_0, V, y, m)
        grad_v_0, grad_V = _grad(A, v_0, V, y, p, m)

        b_0_prev, B_prev = b_0, B

        # Adjust the step size with backtracking line search
        if t_init is not None:
            t = t_init
        while True:
            # Compute the update based on gradient, then apply prox
            b_0 = v_0 - t * grad_v_0
            B = V - t * grad_V
            B = _prox(B, a_1, a_2, t)

            if ls_beta is None:
                # Don't perform line search
                break

            # The line search condition is to exit when
            #   g(b) <= g(v) + grad_v.T@(b - v) + (1/2t)||b - v||^2
            g_b = _g(A, b_0, B, y, m)
            b0_v0_diff = b_0 - v_0
            B_V_diff = B - V
            # grad_v.T@(b - v):
            c_2 = grad_v_0 * b0_v0_diff + (grad_V * B_V_diff).sum()
            # (1/2t)||b - v||^2:
            c_3 = (b0_v0_diff ** 2 + (B_V_diff ** 2).sum()) / (2. * t)

            if g_b <= g_v + c_2 + c_3:
                break
            else:
                t *= ls_beta

        # Compute relative change in b
        b_0_diff = b_0 - b_0_prev
        B_diff = B - B_prev
        delta_norm = (b_0_diff ** 2 + (B_diff ** 2).sum()).sqrt()
        b_norm = (b_0 ** 2 + (B ** 2).sum()).sqrt()

        pbar_stats["t"] = "{:.2g}".format(t)
        pbar_stats["rel change"] = "{:.2g}".format((delta_norm / b_norm).item())
        pbar.set_postfix(pbar_stats)
        pbar.update()

        # Check max iterations exit criterion
        if max_iters is not None and k == max_iters:
            break
        k += 1

        # Check tolerance exit criterion
        # Break if the relative change in 2-norm between b and
        # b_prev is less than tol
        if delta_norm.item() <= rel_tol * b_norm.item():
            break

    pbar.close()
    return float(b_0), B
