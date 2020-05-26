pylada-crystal
==============

.. image:: https://travis-ci.org/pylada/pylada-light.svg?branch=master
    :align: left
    :target: https://travis-ci.org/pylada/pylada-light

A python computational physics framework.

Minimal version of pylada necessary to just run the crystal,VASP,ewald,jobs,and
database modules

Constructed by Peter Graf from Mayeul d'Avezac's pylada

Usage
-----

There are some IPython notebooks in the notebooks subdirectory. And documentation can be found
[online](http://pylada.github.io/pylada-light/), though somewhat out of date. Notably, it does not
describe the PWSCF wrapper.

For more examples, look at the tests in each subfolder, and at the BDD scenarios in the
espresso/tests/bdd/features folder.

Finally, do join us on [slack](pylada.slack.com). Send one of the authors an email if you need
access.

Installation
------------

The simplest approach is to install via
`pip <https://pip.pypa.io/en/latest/>`__:

- global installation

    .. code:: bash

        pip install git+https://github.com/pylada/pylada-light

- local (user) installation

    .. code:: bash

        pip install --user git+https://github.com/pylada/pylada-light

- in a `virtual environment <https://virtualenv.pypa.io/en/latest/>`__

    .. code:: bash

        python -m venv pylada
        source pylada/bin/activate
        pip install git+https://github.com/pylada/pylada-light

    This last approach is recommended since it keeps the pylada environment
    isolated from the rest of the system. Susbsequently, this environment can
    be accessed by running the second line.

Installation for development
----------------------------

- python setup.py develop

    .. code:: bash

        python -m venv pylada
        source pylada/bin/activate
        git clone https://github.com/pylada/pylada-light
        cd pylada-light
        python -m pip install cython setuptools wheel scikit-build cmake ninja numpy
        python -m pip install -e .[dev]
        python setup.py test
        ln -s src/pylada . # because https://github.com/scikit-build/scikit-build/issues/363

    The above creates a virtual environment and installs pylada inside it in
    development mode. This means that the virtual environment will know about
    the pylada flavor in development. It is possible to edit a file, do
    :bash:`make`, launch python and debug. One just needs to active the
    virtual environment once per session.

    When modifying files that are built (`.pyx`, `.cc`, `.h`), it may be
    necessary to run `python setup.py develop` again.
