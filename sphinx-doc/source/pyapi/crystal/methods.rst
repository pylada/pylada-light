Methods
-------
All the methods below are available directly under :py:mod:`pylada.crystal`.

.. currentmodule:: pylada.crystal.cppwrappers

Vector transformations
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: zero_centered(a, cell, invcell=None)->[numpy.array(..),...]
.. autofunction:: into_voronoi(a, cell, invcell=None)->[numpy.array(..),...]
.. autofunction:: into_cell(a, cell, invcell=None)->[numpy.array(..),...]

Structure transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: supercell(lattice, cell)->Structure
.. autofunction:: primitive(supercell, tolerance=1e-8)->Structure
.. autofunction:: transform(structure, operation)->Structure
.. currentmodule:: pylada.crystal
.. autofunction:: vasp_ordered
.. currentmodule:: pylada.crystal.cppwrappers

Structure and Geometry Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: is_primitive(supercell, tolerance=1e-8)->Boolean
.. autofunction:: space_group(structure, tolerance=1e-8)->Structure
.. autofunction:: map_sites(mapper, mappee, cmp=None, tolerance=1e-8)->Boolean
.. autofunction:: neighbors(structure, nmax, center, tolerance=1e-8)->[(Atom, numpy.array, float), ...]
.. autofunction:: coordination_shells(structure, nshells, center, tolerance=1e-8, natoms=0)->[ [(Atom, numpy.array, float), ...], ...]
.. autofunction:: periodic_dnc(structure, overlap, mesh=None, nperbox=None, tolerance=1e-8)->[ [(Atom, numpy.array, bool), ...], ...]
.. autofunction:: splitconfigs(structure, center, nmax, configurations=None, tolerance=1e-8)->[ ( (Atom, numpy.array), ...), float), ...]
.. currentmodule:: pylada.crystal
.. autofunction:: specieset


Iterators
~~~~~~~~~

.. automodule:: pylada.crystal.iterator


Reading and Writing
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylada.crystal.read
.. autofunction:: poscar
.. autofunction:: castep

.. currentmodule:: pylada.crystal.write
.. autofunction:: poscar
.. autofunction:: castep
