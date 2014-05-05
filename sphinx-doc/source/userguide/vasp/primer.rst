.. _vasp_primer_ug: 
    
A fast primer
=============

:py:class:`~functional.Vasp` is a python wrapper which hides from the user all
the dreary file manipulation and grepping that working with scientific codes
generally imply. It is initialized as follows:

>>> from pylada.vasp import Vasp
>>> vasp = Vasp()
>>> vasp.add_specie = 'Si', '/path/to/directory/Si' 
>>> vasp.add_specie = 'Ge', '/path/to/directory/Ge' 

We will get to the details below, but suppose that we already have some
silicon-germanium structure. Then launching the calculation and retrieving the
gap is as complicated as: 

>>> result = vasp(structure, ispin=2)
>>> print "Success?", result.success
True
>>> print result.eigenvalues
array([-5, -6., -7]) * eV

This will launch vasp in the current directory and wait until the calculation
is finished. We then check whether the calculations completed and print the
eigenvalues. The latter are a numpy array_, signed with the appropriate `units
<qantities>`_.  Since they are a python object, it is easy to create more
complex post-processing. 

.. _array: http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
