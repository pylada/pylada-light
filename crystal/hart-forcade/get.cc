/******************************
   This file is part of PyLaDa.

   Copyright (C) 2013 National Renewable Energy Lab
  
   PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
   large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
   crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
   is able to organise and launch computational jobs on PBS and SLURM.
  
   PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
   Public License as published by the Free Software Foundation, either version 3 of the License, or (at
   your option) any later version.
  
   PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.
  
   You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
   <http://www.gnu.org/licenses/>.
******************************/

//! Returns transformation matrix as numpy array.
PyObject* hftransform_gettransform(PyHFTObject *_self, void *closure)
   { return python::numpy::wrap_to_numpy(_self->transform, (PyObject*)_self); }
//! Returns periodicity quotient as numpy array.
PyObject* hftransform_getquotient(PyHFTObject *_self, void *closure)
   { return python::numpy::wrap_to_numpy(_self->quotient, (PyObject*)_self); }
//! Number of unit-cells in the supercell.
PyObject* hftransform_getsize(PyHFTObject *_self, void *closure)
   { return PyLong_FromLong(   _self->quotient[0] 
                             * _self->quotient[1]
                             * _self->quotient[2] ); }
