pylada-crystal
==============

.. image:: https://travis-ci.org/pylada/pylada-light.svg?branch=master
    :align: left
    :target: https://travis-ci.org/pylada/pylada-light

A python computational physics framework.

Minimal version of pylada necessary to just run the crystal,VASP,ewald,jobs,and
database modules

Constructed by Peter Graf from Mayeul d'Avezac's pylada

Installation
------------

There are currently two pre-requisites:

- `CMake <http://www.cmake.org/>`__, a cross-platform build system
- `git <http://git-scm.com/`, a distributed version control system

Both are generally available on Linux and OS/X (via `homebrew <http://brew.sh/>`__ for instance). 

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

        mkvirtualenv --system-site-packages pylada
        source pylada/bin/activate
        pip install --user git+https://github.com/pylada/pylada-light
    
    This last approach is recommended since it keeps the pylada environment
    isolated from the rest of the system. Susbsequently, this environment can
    be accessed by running the second line.

Installation for development
----------------------------

There are two approaches for developping with pylada. One is to use the
bare-bone cmake build system. The other is to two have pip setup the cmake
system for you.

In either case, the source should first be obtained from the github repo.

- bare-bone

    .. code:: bash

        cd /path/to/source
        mkdir build
        cd build
        cmake ..
        make
        make test

    The usual cmake options apply. In order to facilitate debugging, a script
    :bash:`localpython.sh` exists in the build directory to a run a python
    session that will know where to find the pylada package that is being
    developped.  For instance :bash:`./localpython.sh -m IPython` will launch
    an ipython session where the current pylada can be imported.


- pip develop

    .. code:: bash

        mkvirtualenv --system-site-packages pylada
        source pylada/bin/activate
        pip install -e git+https://github.com/pylada/pylada-light#egg=pylada
        cd pylada/src/pylada/build
        make test
    
    The above creates a virtual environment and installs pylada inside it in
    development mode. This means that the virtual environment will know about
    the pylada flavor in development. It is possible to edit a file, do
    :bash:`make`, launch python and debug. One just needs to active the virtual
    environment once per session.
