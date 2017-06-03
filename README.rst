.. -*- mode: rst -*-

soft-DTW
=========

Python implementation of soft-DTW [1].

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
