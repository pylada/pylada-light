Compiling and installing
========================

The source code for pylada can be obtained from github_. 

Requirements
------------

- cmake_: software tool to configure and create the makefiles. Any
  version 2.8 and larger should work. This program is often available on
  supercomputers *via* ``module load cmake``.

- Boost_: A set of high-performance C++ libraries. Most of these
  libraries contain only header files. Hence, they can be installed
  rather easily. Pylada does not require any compiled boost.
  PyLada uses the following header-only libraries: mpl,
  enable if, type traits, preprocessor, bind, ref, static assert,
  exceptions. Most Linux distributions will contain Boost in some rpm of
  deb package. It is often not available on supercomputers and will need
  to be installed there. 

  .. note::

     Pylada also uses ``boost.mpi`` in some of its default configuration
     scripts. It is a python package which interface with the MPI
     library.  It is not strictly required and other MPI package could
     replace it in the relevant config files.
 
- Eigen_: An efficient C++ numerical library. This is a header-only
  library which is fairly easy to install with CMake. Versions 2 and 3
  are supported.

- Python_: programming language with which to interface with Pylada.
  Should be between version 2.6.4 included and 3.0 excluded. Python_ is
  likely available by default on any unix system. 

- Numpy_: a Python_ package for numerical calculations. It is available
  on all Linux distributions. It is often time available by default on
  supercomputers. If not, it can be installed *via* easy_install_ or
  pip_. 

- quantities_: A Python_ package which implements physical dimensions and
  units, such as length, weight, etc. It works well with Numpy_. It is
  unlikely to be installed by default on any system. It can be installed
  easily with easy_install_ or pip_.

- IPython_: an interactive shell for Python_. It forms the basis of
  Pylada's user-interface. It must be version 0.13 or higher. It can be
  installed in its simplest form without any of its optional
  dependencies. This package will not likely to be found installed by
  default. It can be installed easily with easy_install_ or pip_.

- sphinx_: Only required to create the documentation. Can be installed
  *via* easy_install_ or pip_. 

- ctest_: Only required to run the test suite. It is distributed with
  cmake_.

Recommended
-----------

- matplotlib_: A Python_ package to make nice plots. Not strictly
  required, but it will enable users to plot Pylada generated results
  easily. Can be installed *via* easy_install_ or pip_. It is included in
  all Linux distributions.

Procedure
---------

On a unix machine
~~~~~~~~~~~~~~~~~

The following assumes the required packages listed above are installed.
It also assumes that the source code for Pylada has been extracted to some
directory.

The CMakeLists.txt file in the pylada root directory guides the build.  It contains mechanisms to search for several
required tools and packages (e.g. location of your c++ compiler).
Due to frequent failure of automatic library detection mechanisms, and/or your desire to customize the process 
(e.g. install location), it is quite likely that some or all of the 
following variables will need to be manually specified in a separate file, e.g. `myCmake-env.txt`: 

     - CMAKE_INSTALL_PREFIX 
     - CMAKE_CXX_COMPILER 
     - CMAKE_C_COMPILER 
     - CMAKE_Fortran_COMPILER 
     - PYTHON_LIBRARY 
     - EIGEN3_INCLUDE_DIR

The syntax for such specifications is illustrated by the following example:

.. code-block:: bash

    SET(CMAKE_INSTALL_PREFIX "/Users/pgraf/root/mylada" CACHE STRING "prefix")
    SET(CMAKE_CXX_COMPILER "g++-mp-4.5"  CACHE STRING "c++")
    SET(CMAKE_C_COMPILER "gcc-mp-4.5"  CACHE STRING "c")
    SET(CMAKE_Fortran_COMPILER "gfortran-mp-4.5"  CACHE STRING "fortran")
    SET(PYTHON_LIBRARY  "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib"   CACHE STRING "pylib")
    SET(EIGEN3_INCLUDE_DIR  "/Users/pgraf/opt/eigen3.1.2/"   CACHE STRING "eigen3")

With such an auxilliary CMake file prepared (assuming it is `myCmake-env.txt` in the pylada root directory), 
the following steps configure pylada for building:

.. code-block:: bash

   > cd directory/with/pylada/source
   > mkdir build
   > cd build
   > cmake -C ../myCmake-env.txt ../

After this configuration is done, it is time to build and install pylada:

.. code-block:: bash

   > make
   > make install

The last two lines should see you through the installation.

.. warning::

  cmake_ and Python_ don't always interact well. You may want to do the
  following:

  .. code-block:: bash
     
     > ccmake ..
  
  Then hit [t] for advanced options. Hit [/] followed by
  [PYTHON_INCLUDE_DIR] to look.  This will bring you to
  ``PYTHON_INCLUDE_DIR``. You should make sure that it points to the
  correct python version. Also check ``PYTHON_LIBRARY``. cmake_ often
  picks up different version of python on a system.

Checking your installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Be sure to adjust your ``PYTHONPATH`` environment variable to include the location you installed pylada.
Then, in a python shell, try importing pylada:

>>> import pylada

If you have an error, then python does not know how to find the directory
where Pylada was installed. Make sure that `python is is set up
correctly`__ to include this directory. In general, it is defined by
CMAKE_INSTALL_PREFIX and CMAKE_PYINSTALL_PREFIX.

.. __: http://docs.python.org/2/tutorial/modules.html#the-module-search-path

Once the above works: 

.. code-block:: bash

   > cd path/to/pylada/source
   > cd build
   > make test

This will run Pylada's unit-tests.  It may take some time. Hopefully,
most tests will run. Note that some tests for VASP_ and such
will require that Pylada be configured first, e.g., told how to find the
relevant programs.

Beyond this base installation, there are additional steps required to get pylada working with VASP_,
with your batch queueing system, and with IPython.  Please see :ref:`Setting up Pylada <configuration_ug>`

Creating the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

This documentation is generated using sphinx_. Assuming that python can
find the Pylada installation, the documentation can be generated with:
 
.. code-block:: bash

   > cd path/to/pylada/source
   > cd sphinx-doc
   > make html

This will create a directory "build/html" with an "index.html" file. View
it with your favorite browser and voilà!


.. note::
  
   Virtualenv users will want to do ``pip install sphinx`` to make sure
   that sphinx is started using the virtualenv python. 

Compatibility
~~~~~~~~~~~~~

Unices with gnu compiler, intel compiler, but not Portland compiler (eigen
does not compile). Known to work on Mac OS/X.

.. _ctest: http://www.cmake.org/  
.. _easy_install: http://pypi.python.org/pypi/setuptools
.. _pip: http://pypi.python.org/pypi/pip
.. _sphinx: http://sphinx-doc.org
.. _boost: http://www.boost.org
.. _eigen: http://www.eigen.tuxfamily.org
