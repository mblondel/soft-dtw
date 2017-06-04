.. -*- mode: rst -*-

soft-DTW
=========

Python implementation of soft-DTW [1].

Supported features
------------------

* soft-DTW (forward pass) and gradient (backward pass) computations, implemented in Cython for speed

Planned features
-----------------

* barycenters (time series averaging)
* Chainer function
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
    # gradient w.r.t. D, shape = [m, n]
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

    pip install numpy scipy cython nose


    # via conda

    conda install numpy scipy cython nose


3. Build and install soft-dtw::

    cd soft-dtw
    python setup.py build
    sudo python setup.py install


References
----------

.. [1] Marco Cuturi, Mathieu Blondel.
       *Soft-DTW: a Differentiable Loss Function for Time-Series.*
       In: Proc. of ICML 2017.
       [`PDF <https://arxiv.org/abs/1703.01541>`_]

Author
------

- Mathieu Blondel, 2017
