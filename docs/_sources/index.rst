.. sqwash documentation master file

SQwash: Distributionally Robust Learning in PyTorch with 1 Additional Line of Code
==================================================================================


This package implements reducers based on the 
`superquantile <https://en.wikipedia.org/wiki/Expected_shortfall>`_
a.k.a. Conditional Value at Risk (CVaR) for distributionally robust learning in PyTorch with GPU support.
The package is licensed under the GPLv3 license.

The superquantile allows for distributional robustness by averaging over the worst 
:math:`\alpha` fraction of the losses in each minibatch, as illustrated in the following figure.

.. image:: ../fig/superquantile3.png
   :scale: 30 %

Table of Contents
------------------
* :ref:`installation`
* :ref:`quick start`
* :ref:`functionality`
* :ref:`mathematical definitions`
* `API Details <api.html>`_
* :ref:`authors`
* :ref:`cite`
* :ref:`acknowledgments`

Installation
------------
Once you have PyTorch >=1.7, you can grab SQwash from pip:

.. code-block:: bash

    $ pip install sqwash

Alternatively, if you would like to edit the package, clone the repository, `cd` into the main directory of the repository and run

.. code-block:: bash

    $ pip install -e .

The only dependency of SQwash is PyTorch, version 1.7 or higher.
See `the PyTorch webpage <https://pytorch.org/>`_ for install instructions.

Quick Start
--------------

As the name suggests, it requires only a one-line modification to the usual PyTorch training loops.
See the notebooks folder for `a full example on CIFAR-10 <https://github.com/krishnap25/sqwash/notebooks/cifar10_example.ipynb>`_.

.. code-block:: python
  :linenos:
  :emphasize-lines: 3,9

    from sqwash import SuperquantileReducer
    criterion = torch.nn.CrossEntropyLoss(reduction='none')  # set `reduction='none'`
    reducer = SuperquantileReducer(superquantile_tail_fraction=0.5)  # define the reducer

    # Training loop
    for x, y in dataloader:
        y_hat = model(x)
        batch_losses = criterion(y_hat, y)  # shape: (batch_size,)
        loss = reducer(batch_losses)  # Additional line to use the superquantile reducer
        loss.backward()  # Proceed as usual from here
        ...

The package also gives a functional version of the reducers, similar to ``torch.nn.functional``:

.. code-block:: python
  :linenos:
  :emphasize-lines: 2,7

    import torch.nn.functional as F
    from sqwash import reduce_superquantile

    for x, y in dataloader:
        y_hat = model(x)
        batch_losses = F.cross_entropy(y_hat, y, reduction='none')  # must set `reduction='none'`
        loss = reduce_superquantile(batch_losses, superquantile_tail_fraction=0.5)  # Additional line
        loss.backward()  # Proceed as usual from here
        ...

The package can also be used for distributionally robust learning over 
pre-specified groups of data. Simply obtain a tensor of losses for each element of the batch and 
use the reducers in this pacakge as follows:

.. code-block:: python
  :linenos:

    loss_per_group = ...  # shape: (num_groups,)
    reducer = reduce_superquantile(loss_per_group, superquantile_tail_fraction=0.6)

Functionality
---------------
This package provides 3 reducers, which take a tensor of losses on a minibatch and reduce them to a single value. 

* ``MeanReducer``: the usual reduction, which is equivalent to specifying ``reduction='mean'`` in your criterion.
    Given a ``torch.Tensor`` denoting a vector :math:`\ell = (\ell_1, \cdots, \ell_n)`, the ``MeanReducer`` 
    simply returns the mean :math:`\sum_{i=1}^n \ell_i / n`. The functional equivalent of this is 
    ``reduce_mean``.

* ``SuperquantileReducer``: computes the superquantile/CVaR of the batch losses.
    Given a ``torch.Tensor`` denoting a vector :math:`\ell = (\ell_1, \cdots, \ell_n)`, the ``SuperquantileReducer`` 
    with a ``superquantile_tail_fraction`` denoted by :math:`\alpha` returns the :math:`(1-\alpha)-` superquantile :math:`\mathrm{SQ}_\alpha` of :math:`\ell`.
    See the :ref:`mathematical definitions` for its precise definition.
    Its functional counterpart is ``reduce_superquantile``.

* ``SuperquantileSmoothReducer``: computes a smooth counterpart of the superquantile/CVaR of the batch losses.
    Given a ``torch.Tensor`` denoting a vector :math:`\ell = (\ell_1, \cdots, \ell_n)`, the ``SuperquantileReducer`` 
    with a ``superquantile_tail_fraction`` denoted by :math:`\alpha` and a smoothing parameter 
    denoted by :math:`\nu`
    returns the :math:`\nu-` smoothed :math:`(1-\alpha)-` superquantile :math:`\mathrm{SQ}_\alpha^\nu` of :math:`\ell`.
    See the :ref:`mathematical definitions` for its precise definition.
    Its functional counterpart is ``reduce_superquantile_smooth``.

See here for `details of the API. <api.html>`_
Each of these reducers work just as well with cuda tensors for efficient 
distributionally robust learning on the GPU.

Mathematical Definitions
------------------------

The :math:`(1-\alpha)-` superquantile of :math:`\ell=(\ell_1, \cdots, \ell_n)`
to an average over the :math:`\alpha` fraction of the largest elements of 
:math:`\ell`, if :math:`n\alpha` is an integer. See the figure at the top of the page.
Formally, it is given by the two equivalent expressions (which are also valid when 
:math:`n\alpha` is not an integer):

.. math:: 

    \mathrm{SQ}_{\alpha}(\ell) = \max\Bigg\{ q^\top \ell \, : \, q \in R^n_+, \, q^\top 1 = 1, \, q_i \le \frac{1}{n\alpha} \Bigg\}
        = \min_{\eta \in R} \Bigg\{ \eta + \frac{1}{n\alpha} \sum_{i=1}^n \max\{\ell_i - \eta, 0\}  \Bigg\}.

The :math:`\nu-` smoothed :math:`(1-\alpha)-` superquantile of :math:`\ell=(\ell_1, \cdots, \ell_n)`
is given by

.. math:: 

    \mathrm{SQ}_{\alpha}^\nu(\ell) = \max\Bigg\{ q^\top \ell - \frac{\nu}{2n}\big\|q - u \big\|^2_2 \, : \, q \in R^n_+, \, q^\top 1 = 1, \, q_i \le \frac{1}{n\alpha} \Bigg\}.

where :math:`u = \mathbf{1}_n / n` denotes the uniform distribution over :math:`n` atoms.


Authors
-------
* `Krishna Pillutla <https://homes.cs.washington.edu/~pillutla/>`_
* `Yassine Laguel <https://yassine-laguel.github.io>`_
* `Jérôme Malick <https://ljk.imag.fr/membres/Jerome.Malick/>`_
* `Zaid Harchaoui <http://faculty.washington.edu/zaid/>`_

For any questions or comments, please raise an issue on github, or 
contact `Krishna Pillutla <https://krishnap25.github.io>`_.

Cite
----
If you found this package useful, please cite the following work.
If you use this code, please cite::

	@article{sfl_mlj_2023,
	title = {Federated Learning with Superquantile Aggregation for Heterogeneous Data},
	author={Pillutla, Krishna and Laguel, Yassine and Malick, J{\'{e}}r{\^{o}}me and Harchaoui, Zaid},
	journal   = {Mach. Learn.},
	year = {2023},
	publisher={Springer}
	}

	@inproceedings{DBLP:conf/ciss/LPMH21,
	author    = {Yassine Laguel and
		Krishna Pillutla and
		J{\'{e}}r{\^{o}}me Malick and
		Zaid Harchaoui},
	title     = {{A Superquantile Approach to Federated Learning with Heterogeneous
		Devices}},
	booktitle = {55th Annual Conference on Information Sciences and Systems, {CISS}
		2021, Baltimore, MD, USA, March 24-26, 2021},
	pages     = {1--6},
	publisher = {{IEEE}},
	year      = {2021},
	}


Acknowledgments
---------------
We acknowledge support from NSF DMS 2023166,
DMS 1839371, CCF 2019844, the CIFAR program "Learning
in Machines and Brains", faculty research awards, and a JP
Morgan PhD fellowship. This work has been partially supported
by MIAI – Grenoble Alpes, (ANR-19-P3IA-0003).
