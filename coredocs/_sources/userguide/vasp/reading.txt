Reading from pre-existing VASP files
====================================

There is a fairly good equivalence between the INCAR_ file and the python
functional. It is possible to re-create the python from an INCAR_:

>>> from pylada.vasp import read_incar
>>> functional = read_incar()

By default, :py:func:`pylada.vasp.read_incar` will read from 'INCAR' in the
current directory. However, it is also possible to pass as argument the path
to a file situated somewhere else. 

Similarly, one can read a POSCAR_ file and create a
:py:class:`pylada.crystal.Structure` instance from it:

>>> from pylada.crystal import read
>>> structure = read.poscar(types=['Al', 'Mg', 'O'])

:py:func:`pylada.crystal.read.poscar` will read both VASP_ 4 and 5 POSCAR_
formats. However, in the former case, it is important to include the species in
the POSCAR_, since those are actually missing from the file. If the POSCAR_
file contains dynamic attributes, that information is stored in each atom
within an explicitly created ``freeze`` attribute.

.. warning::

   The INCAR_ file does not contain all the information necessary to recreate a
   run. For instance, :py:attr:`~pylada.vasp.functional.Vasp.kpoints` is not set
   correctly at this point. Nor can it re-create a functional with "enhanced"
   parameters such as :py:attr:`~pylada.vasp.functional.Vasp.ediff_per_specie`.
   Hence, re-creating a functional from an INCAR of a previous Pylada run will
   not necessarily yield the exact same functional as that which created it in
   the first place. However, it will yield the exact same INCAR_ file.
