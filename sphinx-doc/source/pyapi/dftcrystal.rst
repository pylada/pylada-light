======================
CRYSTAL wrapper module
======================

.. module:: pylada.dftcrystal 
   :synopsis: Wrapper for the CRYSTAL code 

This module creates wrappers around the CRYSTAL_ code. It is main focus are
three kinds of objects: 
  
  - :py:class:`~crystal.Crystal`, :py:class:`~molecule.Molecule`, which define
    the structure to be optimized in a functional manner (as opposed to the
    declarative approach of :py:class:`~pylada.crystal.cppwrappers.Structure`)
  - :py:class:`~functional.Functional`, which handles writing the input and
    calling CRYSTAL_ itself
  - :py:class:`~extract.Extract`, which handles grepping values from the output

It strives to reproduce the input style of CRYSTAL_, while still providing a
pythonic interface. As such, the structure CRYSTAL_'s input is mirrored by the
structure of :py:class:`~functional.Functional`'s attributes. 

Content:

.. toctree::

   Functional class and attributes <dftcrystal/functional>

   Crystal structure à la CRYSTAL <dftcrystal/crystal>

.. currentmodule:: pylada.dftcrystal


******
Others
******

.. py:data::  registered

   Map of geometry keywords to their Pylada implementation.


.. autofunction:: read(path) -> structure, functional

.. autofunction:: read_gaussian_basisset(path) -> structure, functional

.. _open: http://docs.python.org/library/functions.html#open
.. _CRYSTAL: http://www.crystal.unito.it/

