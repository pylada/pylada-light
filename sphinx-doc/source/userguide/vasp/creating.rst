.. _vasp_creating_ug:

Creating and Setting up the vasp functional
===========================================

Specifying vasp parameters
--------------------------

As shown in the quick primer above, it is relatively easy to setup a new
:py:class:`~functional.Vasp`. The parameters to the calculation are generally
the VASP_ defaults. They can be changed directly in the constructor:

>>> from pylada.vasp import Vasp
>>> vasp = Vasp(ispin=2)
>>> print vasp.ispin
2

or later, after creation:

>>> vasp.ispin = 1
>>> print vasp.ispin
2

or even right at the moment when executing a calculation:

>>> vasp.encut = 1.2
>>> result = vasp(structure, ispin=2, encut=1)
>>> print vasp.encut
1.2

In the line above, vasp is called with ``ispin=2`` and ``encut=1``. However,
specifying a vasp parameter directly at execution *does* not modify the vasp
object itself, as indicated by the third line. The parameter is only modified
for the duration of the run and not beyond.

At this point, it might have come as a surprise to see ``encut=1``. What units
are we talking about here? A number of parameters have enhanced behaviors with
respect to the original VASP_ code, in order to make it easier to specify
parameters valid for many calculations. :py:attr:`~incar.Incar.encut`
one of those. It can be specified:

  - as floating point smaller than 3, in which case it is a factor of the
    largest `ENMAX` of all the species in the calculation.
  - as floating point larger than 3, in which case it will printed as is in the INCAR.
  - as floating point signed by a quantity:

    >>> from pylada.physics import Ry
    >>> vasp.encut = 10*Ry

    In that case, the result will be converted to electron-Volts in the INCAR.

To see the full list of parameters  defined by Pylada, do ``vasp.[TAB]`` on the
ipython_ command line, or go :py:class:`~pylada.vasp.functional.Vasp`.


Adding parameters Pylada does not know about
--------------------------------------------

A large number of parameters control VASP_. Pylada only knows about the most
common. However, it is fairly easy to add parameters which are not there yet.

>>> vasp.add_keyword('encut', 300)

This will add a parameter to VASP. It will appear in the incar as "ENCUT =
300", with the tag name in uppercase. The parameter can be later accessed and
modified just as if it where a pre-existing parameter.

>>> print vasp.encut
300

The attribute ``vasp.encut`` is (in this case) an integer which can be
manipulated as any other python integer. Any python object can be given. It
will appear in the INCAR as it appears when transformed to a string (using
python's str_ function).

If you do *not* want a parameter printed to the INCAR, i.e. you want to use the
VASP_ default for that parameter, then simply set that parameter to ``None``:

>>> vasp.encut = None

.. note::

   The examples above specifies a parameter which already exists,
   :py:attr:`~functional.Vasp.encut`.  In this case, when calling
   :py:meth:`~pylada.functional.Vasp.add_keyword`,  the parameter is replaced
   with a simple integer (or float, or whatnot).  Its special behavior
   described above is simply gone.  If you do not like a special behavior, you
   can always do away with it.

.. _str: http://docs.python.org/2/library/functions.html#str

Specifying kpoints
------------------

K-points can be specified in  a variety of ways.

Simply as a KPOINTS_ file in string:

>>> vasp.kpoints = "Automatic generation\n0\nMonkhorst\n2 2 2\n0 0 0" 

As a 4xn matrix where the first three components of each row define a single
kpoint in *cartesian* coordinates, and the last component its multiplicity. In
that case, VASP_ does not reduce the kpoints by symmetry, as explained `here
<KPOINTS>`_.

>>> vasp.kpoints = [ [0, 0, 0, 1], [0.1, 0.1, 0.1, 3] ]


As a callable function which takes the vasp functional and the structure as
input. It must return either a string defining a semantically correct KPOINTS_
file, or an array of kpoints as above.

>>> def some_function(functional, structure):
>>>   .... do something
>>>   return kpoints
>>>
>>> vasp.kpoints = some_function

This function is called before each execution. Hence the kpoint grid can be
tailored to each call.


Specifying the pseudos and Hubbard U parameters
-----------------------------------------------

Pseudo-potentials must be specified explicitely:

>>> vasp.add_specie = 'Tc', '/path/to/pseudo/directory/Tc'

The first item is the name of the atomic specie. It corresponds to the type of
the atom in the structure to compute. The second item is a path to a directory
where the appropriate *unzipped* POTCAR resides. It will not affect calculations.
For convenience, the path may be given with the usual unix short-cuts and/or with
a shell environment variable.

>>> vasp.add_specie = 'Tc', '~/pseudos/$PAW/Tc'

To specify a Hubbard U parameter, do:

>>> from pylada.vasp.specie import U
>>> vasp.add_specie = 'Tc', '~/pseudos/$PAW/Tc', U('dudarev', l=2, U=1.5)

The species can be latter accessed through a dictionary in the vasp object:

>>> vasp.add_specie = 'Tc', '~/pseudos/$PAW/Tc'
>>> print vasp['Tc']
Specie('Tc', '~/pseudos/$PAW/Tc')
>>> vasp['Tc'].moment = 5

At which point, other elemental properties could be added for latter use in a
script.

.. note::
 
   It is possible to specify more species than exists in a given structure.
   Pylada will figure out at runtime.
