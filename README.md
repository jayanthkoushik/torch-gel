torch-gel
=========

This package provides PyTorch implementations to solve the group elastic
net problem. Let <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/58c9277a170088a03229936790d23a98.svg?invert_in_darkmode" align=middle width=18.364500000000003pt height=22.381919999999983pt/> be <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode" align=middle width=8.239720500000002pt height=14.102549999999994pt/> feature matrices of sizes <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/a4ed036335baa13d206a2d6ea4609d93.svg?invert_in_darkmode" align=middle width=50.334405000000004pt height=19.10667000000001pt/> (<img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.379255000000002pt height=14.102549999999994pt/> is the number of samples, and <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/54158e2c605c3ecf783cdc13e7235676.svg?invert_in_darkmode" align=middle width=15.911775pt height=14.102549999999994pt/> is the number of features
in the <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/95291b39ba5d9dba052b40bf07b12cd2.svg?invert_in_darkmode" align=middle width=20.29962pt height=27.852989999999977pt/> group), and let <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode" align=middle width=8.616960000000002pt height=14.102549999999994pt/> be an <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/1386649e9283ede5e47d3f0868b4c4aa.svg?invert_in_darkmode" align=middle width=42.611085pt height=21.10812pt/> vector of the responses. Group elastic net finds coefficients
<img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/d03f98cb70df5aa6597689da142dc0af.svg?invert_in_darkmode" align=middle width=15.345000000000002pt height=22.745910000000016pt/> and a bias <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/3bde0199092dbb636a2853735fb72a69.svg?invert_in_darkmode" align=middle width=15.791325000000004pt height=22.745910000000016pt/> that solve the optimization problem

<p align="center"><img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/6fd12c4315adee0e731317a0a33a604c.svg?invert_in_darkmode" align=middle width=451.5984pt height=47.884155pt/></p>

Here <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/ce9b0d1765717c60b7915f2a48951a92.svg?invert_in_darkmode" align=middle width=16.081395pt height=22.745910000000016pt/> and <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/22d952fd172ae91ac1817c8f2b3be088.svg?invert_in_darkmode" align=middle width=16.081395pt height=22.745910000000016pt/> are scalar coefficients that control
the amount of 2-norm and squared 2-norm regularization. This 2-norm
regularization encourages sparsity at the group level; entire <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/d03f98cb70df5aa6597689da142dc0af.svg?invert_in_darkmode" align=middle width=15.345000000000002pt height=22.745910000000016pt/>
might become <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.188554000000002pt height=21.10812pt/>. The squared 2-norm regularization is in similar spirit
to elastic net, and addresses some of the issues of lasso. Note that
group elastic net includes as special cases group lasso
(<img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/c0f9168b293956eb3a2324b751fa0b96.svg?invert_in_darkmode" align=middle width=46.98606000000001pt height=22.745910000000016pt/>), ridge regression (<img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/9213fd7c65cd9538991c0dae5212dcf6.svg?invert_in_darkmode" align=middle width=46.98606000000001pt height=22.745910000000016pt/>), elastic net (each
<img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/89e4b6df2e5ddf9ab142cf0b8cc4d4a6.svg?invert_in_darkmode" align=middle width=46.814789999999995pt height=21.10812pt/>), and lasso (each <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/89e4b6df2e5ddf9ab142cf0b8cc4d4a6.svg?invert_in_darkmode" align=middle width=46.814789999999995pt height=21.10812pt/> and <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/22d952fd172ae91ac1817c8f2b3be088.svg?invert_in_darkmode" align=middle width=16.081395pt height=22.745910000000016pt/> = 0). The
optimization problem is convex, and can be solved efficiently. This
package provides two implementations; one based on proximal gradient
descent, and one based on coordinate descent.

Installation
------------

Install with `pip`{.sourceCode}:

    pip install torchgel

`torch`{.sourceCode} and `tqdm`{.sourceCode} (for progress bars) are
installed as dependencies. A suitable CUDA package needs to be manually
installed for GPU support. Refer to the PyTorch docs for details.

Solving Single Instances
------------------------

The modules `gel.gelfista`{.sourceCode} and `gel.gelcd`{.sourceCode}
provide implementations based on proximal gradient descent and
coordinate descent respectively. Both have similar interfaces, and
expose two main public functions: `make_A`{.sourceCode} and
`gel_solve`{.sourceCode}. The feature matrices should be stored in a
list (say `As`{.sourceCode}) as PyTorch tensor matrices , and the
responses should be stored in a PyTorch vector (say `y`{.sourceCode}).
Additionally, the sizes of the groups (<img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/54158e2c605c3ecf783cdc13e7235676.svg?invert_in_darkmode" align=middle width=15.911775pt height=14.102549999999994pt/>) should be stored in a
vector (say `ns`{.sourceCode}). First use the `make_A`{.sourceCode}
function to convert the feature matrices into a suitable format:

    A = make_A(As, ns)

Then pass `A`{.sourceCode}, `y`{.sourceCode} and other required
arguments to `gel_solve`{.sourceCode}. The general interface is:

    b_0, B = gel_solve(A, y, l_1, l_2, ns, **kwargs)

`l_1`{.sourceCode} and `l_2`{.sourceCode} are floats representing
<img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/ce9b0d1765717c60b7915f2a48951a92.svg?invert_in_darkmode" align=middle width=16.081395pt height=22.745910000000016pt/> and <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/22d952fd172ae91ac1817c8f2b3be088.svg?invert_in_darkmode" align=middle width=16.081395pt height=22.745910000000016pt/> respectively. The method returns a float
`b_0`{.sourceCode} representing the bias and a PyTorch matrix
`B`{.sourceCode} holding the other coefficients. `B`{.sourceCode} has
size <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/9204ed5765e66e2b42cf3013bc35d786.svg?invert_in_darkmode" align=middle width=84.45459pt height=19.10667000000001pt/> with suitable zero padding. The follwing
sections cover additional details for the specific implementations.

### Proximal Gradient Descent (FISTA)

The `gel.gelfista`{.sourceCode} module contains a proximal gradient
descent implementation. It's usage is just as described in the template
above. Refer to the docstring for `gel.gelfista.gel_solve`{.sourceCode}
for details about the other arguments.

### Coordinate Descent

The `gel.gelcd`{.sourceCode} module contains a coordinate descent
implementation. Its usage is a bit more involved than the FISTA
implementation. Coordinate descent iteratively solves single blocks
(each corresponding to a single <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/2020a79c00e140ee1a054ecab57a289c.svg?invert_in_darkmode" align=middle width=13.110240000000003pt height=22.745910000000016pt/>). There are multiple solvers
provided to solve the individual blocks. These are the
`gel.gelcd.block_solve_*`{.sourceCode} functions. Refer to their doc
strings for details about their arguments.
`gel.gelcd.gel_solve`{.sourceCode} requires passing a block solve
functions and its arguments (as a dictionary). Refer to its doc string
for further details.

Solution Paths
--------------

`gel.gelpaths`{.sourceCode} provides a wrapper function
`gel_paths`{.sourceCode} to solve the group elastic net problem for
multiple values of the regularization coefficients. It implements a
two-stage process. For a given <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/ce9b0d1765717c60b7915f2a48951a92.svg?invert_in_darkmode" align=middle width=16.081395pt height=22.745910000000016pt/> and <img src="https://rawgit.com/in	git@github.com:jayanthkoushik/torch-gel/None/svgs/22d952fd172ae91ac1817c8f2b3be088.svg?invert_in_darkmode" align=middle width=16.081395pt height=22.745910000000016pt/>, first the
group elastic net problem is solved and the feature blocks with non-zero
coefficients is extracted (the support). Then ridge regression models
are learned for each of several provided regularization values. The
final model is summarized using an arbitrary provided summary function,
and the summary for each combination of the regularization values is
returned as a dictionary. The doc string contains more details. The
module contains another useful function `ridge_paths`{.sourceCode} which
can efficiently solve ridge regression for multiple regularization
values.
