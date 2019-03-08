# torch-gel

[![Travis][travis-badge]][travis]
[![Appveyor][appveyor-badge]][appveyor]
[![PyPI][pypi-badge]][pypi]
[![PyPi - Python Version][pypi-version-badge]][pypi]
[![GitHub license][license-badge]][license]
[![Code style: black][black-badge]][black]

[travis-badge]: https://img.shields.io/travis/jayanthkoushik/torch-gel.svg?style=for-the-badge&logo=travis
[travis]: https://travis-ci.org/jayanthkoushik/torch-gel
[pypi-badge]: https://img.shields.io/pypi/v/torchgel.svg?style=for-the-badge
[pypi-version-badge]: https://img.shields.io/pypi/pyversions/torchgel.svg?style=for-the-badge
[pypi]: https://pypi.org/project/torchgel/
[license-badge]: https://img.shields.io/github/license/jayanthkoushik/torch-gel.svg?style=for-the-badge
[license]: https://github.com/jayanthkoushik/torch-gel/blob/master/LICENSE
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[black]: https://github.com/ambv/black
[appveyor-badge]: https://img.shields.io/appveyor/ci/jayanthkoushik/torch-gel.svg?style=for-the-badge&logo=appveyor
[appveyor]: https://ci.appveyor.com/project/jayanthkoushik/torch-gel

This package provides PyTorch implementations to solve the group elastic net
problem. Let _A<sub>j</sub>_ (_j = 1 … p_) be feature matrices of sizes _m ×
n<sub>j</sub>_ (_m_ is the number of samples, and _n<sub>j</sub>_ is the number
of features in the _j_<sup>th</sup> group), and let _y_ be an _m × 1_ vector of
the responses. Group elastic net finds coefficients _β<sub>j</sub>_, and a bias
_β<sub>0</sub>_ that solve the optimization problem

> min _<sub>β<sub>0</sub>, …, β<sub>p</sub></sub>_
>     _½ ║y - β<sub>0</sub> - ∑ A<sub>j</sub> β<sub>j</sub>║<sup>2</sup>_
>     + _m ∑ √n<sub>j</sub> (λ<sub>1</sub>║β<sub>j</sub>║_
>                           _+ λ<sub>2</sub>║β<sub>j</sub>║<sup>2</sup>)._

Here _λ<sub>1</sub>_ and _λ<sub>2</sub>_ are scalar coefficients that control
the amount of 2-norm and squared 2-norm regularization. This 2-norm
regularization encourages sparsity at the group level; entire _β<sub>j</sub>_
might become 0. The squared 2-norm regularization is in similar spirit to
elastic net, and addresses some of the issues of lasso. Note that group elastic
net includes as special cases group lasso (_λ<sub>2</sub> = 0_), ridge
regression (_λ<sub>1</sub> = 0_), elastic net (each _n<sub>j</sub> = 1_), and
lasso (each _n<sub>j</sub> = 1_ and _λ<sub>2</sub> = 0_). The optimization
problem is convex, and can be solved efficiently. This package provides two
implementations; one based on proximal gradient descent, and one based on
coordinate descent.

## Installation
Install with `pip`

```bash
pip install torchgel
```

`tqdm` (for progress bars) is pulled in as a dependency. PyTorch (`v1.0+`) is
also needed, and needs to be installed manually. Refer to the [PyTorch
website](<http://pytorch.org>) for instructions.

## Solving Single Instances
The modules `gel.gelfista` and `gel.gelcd` provide implementations based on
proximal gradient descent and coordinate descent respectively. Both have similar
interfaces, and expose two main public functions: `make_A` and `gel_solve`. The
feature matrices should be stored in a list (say `As`) as PyTorch tensor
matrices, and the responses should be stored in a PyTorch vector (say `y`).
Additionally, the sizes of the groups (_n<sub>j</sub>_) should be stored in a
vector (say `ns`). First use the `make_A` function to convert the feature
matrices into a suitable format:

```python
A = make_A(As, ns)
```

Then pass `A`, `y` and other required arguments to `gel_solve`. The general
interface is::

```python
b_0, B = gel_solve(A, y, l_1, l_2, ns, **kwargs)
```

`l_1` and `l_2` are floats representing _λ<sub>1</sub>_ and _λ<sub>2</sub>_
respectively. The method returns a float `b_0` representing the bias and a
PyTorch matrix `B` holding the other coefficients. `B` has size _p ×_
max<sub>_j_</sub> _n<sub>j</sub>_ with suitable zero padding. The following
sections cover additional details for the specific implementations.

### Proximal Gradient Descent (FISTA)
The `gel.gelfista` module contains a proximal gradient descent implementation.
It's usage is just as described in the template above. Refer to the docstring
for `gel.gelfista.gel_solve` for details about the other arguments.

### Coordinate Descent
The `gel.gelcd` module contains a coordinate descent implementation. Its usage
is a bit more involved than the FISTA implementation. Coordinate descent
iteratively solves single blocks (each corresponding to a single
_β<sub>j</sub>_). There are multiple solvers provided to solve the individual
blocks. These are the `gel.gelcd.block_solve_*` functions. Refer to their
docstrings for details about their arguments. `gel.gelcd.gel_solve` requires
passing a block solve function and its arguments (as a dictionary). Refer to
its docstring for further details.

## Solution Paths
`gel.gelpaths` provides a wrapper function `gel_paths` to solve the group
elastic net problem for multiple values of the regularization coefficients. It
implements a two-stage process. For a given _λ<sub>1</sub>_ and _λ<sub>2</sub>_,
first the group elastic net problem is solved and the feature blocks with
non-zero coefficients is extracted (the support). Then ridge regression models
are learned for each of several provided regularization values. The final model
is summarized using an arbitrary provided summary function, and the summary for
each combination of the regularization values is returned as a dictionary. The
docstring contains more details. `gel.ridgepaths` contains another useful function,
`ridge_paths` which can efficiently solve ridge regression for multiple
regularization values.

## Citation
If you find this code useful in your research, please cite

```
@misc{koushik2017torchgel,
  author = {Koushik, Jayanth},
  title = {torch-gel},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jayanthkoushik/torch-gel}},
}
```
