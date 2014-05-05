===========
Vasp Module
===========

Pylada provides an interface wrapping the VASP_ density functional theory code.
This interface manages the input to VASP_, launching the code itself (as an
external program), and retrieving the results as python object. It makes it
much easier to use VASP_ as one step of a more complex computational scheme.

Contents:

.. toctree::
   :maxdepth: 1

   Vasp class <vasp/functional>
   Relaxation methods <vasp/relax>
   Extraction classes <vasp/extract>

.. currentmodule:: pylada.vasp
.. autofunction:: read_input
.. autofunction:: read_incar

.. _VASP: http://www.vasp.at/
