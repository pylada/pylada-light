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

//! Returns cell as a numpy array. 
PyObject* structure_getcell(PyStructureObject *_self, void *closure)
   { return python::numpy::wrap_to_numpy(_self->cell, (PyObject*)_self); }
//! Sets cell from a sequence of 3x3 numbers.
int structure_setcell(PyStructureObject *_self, PyObject *_value, void *_closure);
// Returns the scale.
PyObject* structure_getscale(PyStructureObject *_self, void *closure);
//! Sets the scale from a number.
int structure_setscale(PyStructureObject *_self, PyObject *_value, void *_closure);
//! Gets the volume of the structure
PyObject* structure_getvolume(PyStructureObject *_self, void *_closure)
{
  types::t_real const scale = python::get_quantity(_self->scale);
  types::t_real const result = std::abs(_self->cell.determinant() * std::pow(scale, 3));
  return python::fromC_quantity(result, _self->scale);
}


// Sets cell from a sequence of three numbers.
int structure_setcell(PyStructureObject *_self, PyObject *_value, void *_closure)
{
  if(_value == NULL)
  {
    PYLADA_PYERROR(TypeError, "Cannot delete cell attribute.");
    return -1;
  }
  return python::numpy::convert_to_matrix(_value, _self->cell) ? 0: -1;
}

// Returns the scale of the structure.
PyObject* structure_getscale(PyStructureObject *_self, void *closure)
{
  Py_INCREF(_self->scale); 
  return _self->scale;
}
// Sets the scale of the structure from a number.
int structure_setscale(PyStructureObject *_self, PyObject *_value, void *_closure)
{
  if(_value == NULL) 
  {
    PYLADA_PYERROR(TypeError, "Cannot delete scale attribute.");
    return -1;
  }
  PyObject *result = python::fromPy_quantity(_value, _self->scale);
  if(not result) return -1;
  PyObject *dummy = _self->scale;
  _self->scale = result;
  Py_DECREF(dummy);
  return 0;
}
