Extraction classes
******************
.. module:: pylada.vasp.extract

Instances of the extraction classes are returned by calls to the
:py:class:`vasp <pylada.vasp.functional.Vasp>` and affiliated methods. They
simply grep the OUTCAR for values of interest, e.g. eigenvalues or vasp
parameters. Indeed, these should contain an ``Extract`` attribute which refers
to a class capable of handling the output of the method or vasp object. They
can be instanciated as follows:

>>> a = vasp.Extract('outdir')
>>> a = vasp.epitaxial.Extract('outdir')

Where outdir is the location of the relevant calculation.
The results from the calculations can then be accessed as attributes:

>>> a.total_energy
array(-666.667) * eV

It is possible to extract calculations from a complete folder tree:

>>> from pylada.vasp import MassExtract
>>> a = MassExtract('outdir')
>>> a.total_energy
{
  '/some/path':       array(-666.667) * eV,
  '/some/otherpath':  array(-999.996) * eV,
}

The extraction classes are separated into I/O (:py:class:`IOMixin
<pylada.vasp.extract.mixin.IOMixin>`) and actual methods to grep the OUTCAR for
results (:py:class:`ExtractBase <pylada.vasp.extract.base.ExtractBase>`). This
setup makes it convenient to change the kind of object that can be grepped,
from the standard file on a hard-disk, to a file in a database.

:py:class:`Extract` derives from the following classes.

.. toctree::
   :maxdepth: 1

   pylada.vasp.extract.base <extractbase>
   pylada.vasp.extract.mixin <mixin>


.. autoclass:: Extract
   :show-inheritance:
   :members:
   :inherited-members:
  
.. autoclass:: MassExtract
   :show-inheritance:
   :members:
