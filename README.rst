torch-gel
=========

This package provides PyTorch implementations to solve the group elastic net
problem. Let :math:`A_j` be :math:`p` feature matrices of sizes :math:`m \times
n_j` (:math:`m` is the number of samples, and :math:`n_j` is the number of
features in the :math:`j^{th}` group), and let :math:`y` be an :math:`m \times
1` vector of the responses. Group elastic net finds coefficients :math:`\beta_j`
and a bias :math:`\beta_0` that solve the optimization problem

.. math::
    \min_{\beta_0,\dots,\beta_p} \frac{1}{2m}\|y - \beta_0 - \sum_{j=1}^p
        A_j\beta_j\|_2^2 + \sum_{j=1}^p\sqrt{n_j}(\lambda_1\|\beta_j\|_2 +
        \lambda_2\|\beta_j\|_2^2)

Here :math:`\lambda_1` and :math:`\lambda_2` are scalar coefficients that
control the amount of 2-norm and squared 2-norm regularization.
This 2-norm regularization encourages sparsity at the group level; entire
:math:`\beta_j` might become :math:`0`. The squared 2-norm regularization is
in similar spirit to elastic net, and addresses some of the issues of lasso.
Note that group elastic net includes as special cases group lasso
(:math:`\lambda_2 = 0`), ridge regression (:math:`\lambda_1 = 0`), elastic net
(each :math:`n_j = 1`), and lasso (each :math:`n_j = 1` and :math:`\lambda_2` =
0). The optimization problem is convex, and can be solved efficiently. This
package provides two implementations; one based on proximal gradient descent,
and one based on coordinate descent.

Installation
------------
Install with :code:`pip`::

    pip install torchgel

:code:`tqdm` (for progress bars) is pulled in as a dependency. PyTorch is also
needed, and needs to be installed manually. Refer to the `PyTorch website
<http://pytorch.org>`_ for instructions.

Solving Single Instances
------------------------
The modules :code:`gel.gelfista` and :code:`gel.gelcd` provide implementations
based on proximal gradient descent and coordinate descent respectively. Both
have similar interfaces, and expose two main public functions: :code:`make_A`
and :code:`gel_solve`. The feature matrices should be stored in a list
(say :code:`As`) as PyTorch tensor matrices , and the responses should be stored
in a PyTorch vector (say :code:`y`). Additionally, the sizes of the groups
(:math:`n_j`) should be stored in a vector (say :code:`ns`). First use the
:code:`make_A` function to convert the feature matrices into a suitable format::

    A = make_A(As, ns)

Then pass :code:`A`, :code:`y` and other required arguments to
:code:`gel_solve`. The general interface is::

    b_0, B = gel_solve(A, y, l_1, l_2, ns, **kwargs)

:code:`l_1` and :code:`l_2` are floats representing :math:`\lambda_1` and
:math:`\lambda_2` respectively. The method returns a float :code:`b_0`
representing the bias and a PyTorch matrix :code:`B` holding the other
coefficients. :code:`B` has size :math:`p \times \max_j n_j` with suitable zero
padding. The follwing sections cover additional details for the specific
implementations.

Proximal Gradient Descent (FISTA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :code:`gel.gelfista` module contains a proximal gradient descent
implementation. It's usage is just as described in the template above.
Refer to the docstring for :code:`gel.gelfista.gel_solve` for details about
the other arguments.

Coordinate Descent
~~~~~~~~~~~~~~~~~~
The :code:`gel.gelcd` module contains a coordinate descent implementation. Its
usage is a bit more involved than the FISTA implementation. Coordinate descent
iteratively solves single blocks (each corresponding to a single :math:`b_j`).
There are multiple solvers provided to solve the individual blocks. These are
the :code:`gel.gelcd.block_solve_*` functions. Refer to their doc strings for
details about their arguments. :code:`gel.gelcd.gel_solve` requires passing a
block solve functions and its arguments (as a dictionary). Refer to its doc
string for further details.

Solution Paths
--------------
:code:`gel.gelpaths` provides a wrapper function :code:`gel_paths` to solve
the group elastic net problem for multiple values of the regularization
coefficients. It implements a two-stage process. For a given :math:`\lambda_1`
and :math:`\lambda_2`, first the group elastic net problem is solved and the
feature blocks with non-zero coefficients is extracted (the support). Then
ridge regression models are learned for each of several provided regularization
values. The final model is summarized using an arbitrary provided summary
function, and the summary for each combination of the regularization values is
returned as a dictionary. The doc string contains more details. The module
contains another useful function :code:`ridge_paths` which can efficiently
solve ridge regression for multiple regularization values.
