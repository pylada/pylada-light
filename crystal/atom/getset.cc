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

//! Returns position as a numpy array. 
PyObject* pylada_atom_getpos(PyAtomObject *_self, void *closure);
//! Sets position from a sequence of three numbers.
int pylada_atom_setpos(PyAtomObject *_self, PyObject *_value, void *_closure);
//! Returns type python object.
PyObject* pylada_atom_gettype(PyAtomObject *_self, void *closure);
//! Sets type python object.
int pylada_atom_settype(PyAtomObject *_self, PyObject *_value, void *_closure);


// Returns position as a numpy array. 
// Numpy does not implement python's cyclic garbage, hence new wrapper need be
// created each call.
PyObject* pylada_atom_getpos(PyAtomObject *_self, void *closure)
{
  npy_intp dims[1] = {3};
  int const value = python::numpy::type<math::rVector3d::Scalar>::value;
  PyArrayObject* result = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, value, _self->pos.data());
  if(result == NULL) return NULL;
  Py_INCREF(_self);
  PyArray_SetBaseObject(result, (PyObject*)_self);
  return (PyObject*)result;
}
// Sets position from a sequence of three numbers.
int pylada_atom_setpos(PyAtomObject *_self, PyObject *_value, void *_closure)
{
  if(_value == NULL)
  {
    PYLADA_PYERROR(TypeError, "Cannot delete pos attribute.");
    return -1;
  }
  return python::numpy::convert_to_vector(_value, _self->pos) ? 0: -1;
}
// Returns type python object.
PyObject* pylada_atom_gettype(PyAtomObject *_self, void *closure)
  { Py_INCREF(_self->type); return _self->type; }
  
// Sets type python object.
int pylada_atom_settype(PyAtomObject *_self, PyObject *_value, void *_closure)
{
  if(_value == NULL)
  {
    PYLADA_PYERROR(TypeError, "Cannot delete type. You can set it to None though.");
    return -1;
  }
  PyObject* dummy = _self->type;
  _self->type = _value;
  Py_INCREF(_value);
  Py_DECREF(dummy);
  return 0;
}
