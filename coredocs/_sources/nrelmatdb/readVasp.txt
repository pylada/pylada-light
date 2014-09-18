
readVasp.py
=============


Our initial implementation is for VASP output,
so let's focus on VASP.

VASP requires the following input files, although
it can use many more:


  * INCAR: Main list of parameters.
  * KPOINTS: Specify the k-point grid.
  * POSCAR: Specify atom positions.
  * POTCAR: Specify atomic pseudopotentials.

VASP produces many output files, but the only files we retain
are the input ones, listed above, and the two output files:

  * OUTCAR: General log and results in human-readable format.
  * vasprun.xml: All results in XML format.

During a study a researcher may produce terabytes of VASP-related
information: for each of many structures, multiple VASP runs
perform different relaxations.

Typically a group of related directories will be uploaded
at once.  They are identified by a wrapId.

-------------------------------------------------------

.. automodule:: nrelmat.readVasp

.. currentmodule:: nrelmat.readVasp
.. autoclass:: ResClass
.. autofunction:: main
.. autofunction:: parseDir
.. autofunction:: setRunType
.. autofunction:: calcMisc
.. autofunction:: calcBandgaps
