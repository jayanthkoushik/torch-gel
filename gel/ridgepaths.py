"""ridgepaths.py: solution paths for ridge regression."""

import torch
import tqdm


def ridge_paths(X, y, support, lambdas, summ_fun, verbose=False):
    """Solve ridge ridgression for a sequence of regularization values,
    and return the summary for each.

    The multiple solutions are obtained efficiently using the Woodbury identity.
    With X (p x m) representing the feature matrix, and y (m x 1) the outcomes,
    the ridge solution is given by

        b = (X@X' + l*I)^{-1}@X@y

    where l is the regularization coefficient. This can be reduced to

        (1/l)*(X@y - X@V(e + l*I)^{-1}@(X@V)'@X@y)

    where V@e@V' is the eigendecomposition of X'@X. Since (e + l*I) is a
    diagonal matrix, its inverse can be performed efficiently simply by taking
    the reciprocal of the diagonal elements. Then, (X@V)'@X@y is a vector; so it
    can be multiplied by (e + l*I)^{-1} just by scalar multiplication.

    Arguments:
        X: pxm tensor of features (where m is the number of samples);
            X should be centered (each row should have mean 0).
        y: tensor (length m vector) of outcomes.
        support: LongTensor vector of features to use. This vector should
            contain indices. It can also be None to indicate empty support.
            X should already be indexed using support. This argument is simply
            passed to the summary function.
        lambdas: list of regularization values for which to solve the problem.
        summ_fun: a function that takes (support, b) and returns an arbitrary
            summary.
        verbose: enable/disable the progress bar.

    The function returns a dictionary mapping lambda values to their summaries.
    """
    summaries = {}

    if support is None:
        # Nothing to do.
        for l in tqdm.tqdm(
            lambdas, desc="Solving ridge regressions", disable=not verbose
        ):
            summaries[l] = summ_fun(None, None)
        return summaries

    # Setup.
    _, S, V = torch.svd(X)
    e = S ** 2
    p = X @ y
    Q = X @ V
    r = Q.t() @ p  # (X@V)'@X@y

    # Main loop.
    for l in tqdm.tqdm(lambdas, desc="Solving ridge regressions", disable=not verbose):
        b = (1.0 / l) * (p - Q @ (r / (e + l)))
        summaries[l] = summ_fun(support, b)
    return summaries
