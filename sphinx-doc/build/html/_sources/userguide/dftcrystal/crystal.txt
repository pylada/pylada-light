.. _dftcrystal_crystal_ug: 

CRYSTAL's approach to crystal structures
========================================

.. currentmodule:: pylada.dftcrystal

CRYSTAL_ proposes a functional_ approach to crystals, as opposed to the
imperative_ style used by Pylada. In practice, this means that CRYSTAL_ declares
a chain of functions which acts upon an initial data and transform it. The
initial data is generally a space-group with a set of atomic sites. The
functions can be affine transformations on these sites, additions and removals
of sites, transformation upon the space-group, strain relaxation, or others.

In practice, both approaches can "do the same thing", and indeed both CRYSTAL_
and Pylada provide similar functionalities, e.g. creating a supercell from  a
unit-cell. However, there are clear benefits from allowing users to keep
working the CRYSTAL_ way when using with CRYSTAL_. 

Pylada currently provides three classes which allow to work with
:py:class:`molecules <molecule.Molecule>` defined from a spacegroup and
atomic-site occupations, :py:class:`3d-periodic crystalline structures
<crystal.Crystal>` defined from a space-group, lattice parameters, and the
occupation of Wyckoff positions, and finally, :py:class:`structures
<external.External>` defined the Pylada way, where symmetries must explicitely
provided to CRYSTAL_ (as in provided by user or determined by Pylada and then fed
to CRYSTAL_).

Since it is at present the main use cases, only the last two are described
here. 

.. _functional: http://en.wikipedia.org/wiki/Functional_programming
.. _imperative: http://en.wikipedia.org/wiki/Imperative_programming

Defining structures starting from the initial lattice
-----------------------------------------------------


.. currentmodule:: pylada.dftcrystal.crystal

This is the main use case scenario. The structures are defined much as in
CRYSTAL_. The following creates the diamond unit cell.

.. code-block:: python

   from pylada.dftcrystal import Crystal
   crystal = Crystal(227, 3.57)
   crystal.add_atom(0.125, 0.125, 0.125, 'C')

The first argument is the space-group, and the second the only required lattice
parameter. If other parameters were needed they would be listed directly
``Crystal(?, a, b, c, alpha, beta, gamma)``. Only the parameters required for
that particular space-group should be listed.

The second line adds an atomic site to the initial structure. Only
symmetrically inequivalent sites should be given. Diamond contains two sites
linked by a symmetry operations of the space-group 227. Hence, only one site is
actually listed. CRYSTAL_ sets the fractional positions of the symmetry
operations following the international crystallography table. Hence, the
Wyckoff positions are predetermined. To change the origin of the unit-cell, one
can introduce a ``shift`` keyword in the call to :py:class:`Crystal`.
:py:meth:`Crystal.add_atom` returns an the :py:class:`Crystal` instance itself,
so that atomic site declarations can be chained. Putting the last two
statements together,

.. code-block:: python

   crystal = Crystal(227, 5, shift=(0.125, 0.125, 0.125))           \
             .add_atom(0, 0, 0, 'C')                                \
             .add_atom(0.5, 0.5, 0.5, 'C') 

we can define a fairly unlikely half-Heusler compound where the
parameterization of lattice is such that carbon atoms of the original diamond
are now at the origin (and another at :math:`\left(-\frac{1}{4}, -\frac{1}{4},
-\frac{1}{4}\right)`). 

:py:class:`Crystal` instances function as lists of transformations which
are to be applied to the initial structure. The simplest way to add an
operation is to use the traditional CRYSTAL_ format:


>>> crystal = Crystal(227, 5.43)                                        \\
...                  .add_atom(0.125, 0.125, 0.125, "Si")               \\
...                  .append('supercel', '2 0 0 0 1 0 0 0 1')

Here we have created a supercell of diamond with two unit cell in the
(100) direction. The first string in :py:meth:`append` is the keyword and
the second string its input. Much as :py:meth:`add_atom`,
:py:meth:`append` returns the calling instance of :py:class:`Crystal` so
that calls can be chained into a single declarative expression.

A number of operations are implemented in a more pythonic manner. These
can be added to the chain of functions by calling :py:meth:`append` with
an operation instance as a the only argument.

>>> from pylada.dftcrystal import Slabcut
>>> crystal.append( Slabcut(hkl=(1, 0, 0), isup=1, nl=3) )

:py:class:`~pylada.dftcrystal.input.Slabcut` is an operation to create a
thin-film from a 3d bulk material. 

Finally, the whole  "data+functions" object can be evaluated with
:py:meth:`eval`. This will return a
:py:class:`~pylada.crystal.cppwrappers.Structure` instance which can be
used with other Pylada functionalities. Internally, :py:meth:`eval` makes a
call to CRYSTAL_ and greps the output to construct the output structure.
