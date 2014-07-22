.. _further_ug:

Further example and documentation
*********************************
In addition to the examples discussed here, there are many papers written using LaDa.  There are many directories of the 
current distribution even, that are included as advanced examples, documentation to come...


Simple example from:
mkdir testlada
cd testlada
cp /nopt/nrel/ecom/cid/pylada/dist/pylada.5.0.006/test/highthroughput/{inputCif.py,test.py} .

cp cifs/icsd_060845.cif  structs  #  Cu2 Al2 O4 (mid 68523)

ipython
import test
test.nonmagnetic_wave('pickle', inputpath='inputCif.py')
launch scattered --ppn 24 --account x --queue batch --walltime=1:00:00


.. currentmodule:: pylada.vasp

.. toctree:: 
   :maxdepth: 1

