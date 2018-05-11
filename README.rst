.. -*- mode: rst -*-

soft-DTW
=========

Python implementation of soft-DTW.

What is it?
-----------

The celebrated dynamic time warping (DTW) [1] defines the discrepancy between
two time series, of possibly variable length, as their minimal alignment cost.
Although the number of possible alignments is exponential in the length of the
two time series, [1] showed that DTW can be computed in only quadractic time
using dynamic programming.

Soft-DTW [2] proposes to replace this minimum by a soft minimum. Like the
original DTW, soft-DTW can be computed in quadratic time using dynamic
programming. However, the main advantage of soft-DTW stems from the fact that
it is differentiable everywhere and that its gradient can also be computed in
quadratic time. This enables to use soft-DTW for time series averaging or as a
loss function, between a ground-truth time series and a time series predicted
by a neural network, trained end-to-end using backpropagation.

Supported features
------------------

* soft-DTW (forward pass) and gradient (backward pass) computations,
  implemented in Cython for speed
* barycenters (time series averaging)
* dataset loader for the `UCR archive <http://www.cs.ucr.edu/~eamonn/time_series_data/>`_
* `Chainer <http://chainer.org>`_ function

Planned features
-----------------

* PyTorch function

Example
--------

.. code-block:: python

    from sdtw import SoftDTW
    from sdtw.distance import SquaredEuclidean

    # Time series 1: numpy array, shape = [m, d] where m = length and d = dim
    X = ...
    # Time series 2: numpy array, shape = [n, d] where n = length and d = dim
    Y = ...

    # D can also be an arbitrary distance matrix: numpy array, shape [m, n]
    D = SquaredEuclidean(X, Y)
    sdtw = SoftDTW(D, gamma=1.0)
    # soft-DTW discrepancy, approaches DTW as gamma -> 0
    value = sdtw.compute()
    # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
    E = sdtw.grad()
    # gradient w.r.t. X, shape = [m, d]
    G = D.jacobian_product(E)

Installation
------------

Binary packages are not available.

This project can be installed from its git repository. It is assumed that you
have a working C compiler.

1. Obtain the sources by::

    git clone https://github.com/mblondel/soft-dtw.git

or, if `git` is unavailable, `download as a ZIP from GitHub <https://github.com/mblondel/soft-dtw/archive/master.zip>`_.


2. Install the dependencies::

    # via pip

    pip install numpy scipy scikit-learn cython nose


    # via conda

    conda install numpy scipy scikit-learn cython nose


3. Build and install soft-dtw::

    cd soft-dtw
    python setup.py install


References
----------

.. [1] Hiroaki Sakoe, Seibi Chiba.
       *Dynamic programming algorithm optimization for spoken word recognition.*
       In: IEEE Trans. on Acoustics, Speech, and Sig. Proc, 1978.

.. [2] Marco Cuturi, Mathieu Blondel.
       *Soft-DTW: a Differentiable Loss Function for Time-Series.*
       In: Proc. of ICML 2017.
       [`PDF <https://arxiv.org/abs/1703.01541>`_]

Author
------

- Mathieu Blondel, 2017
