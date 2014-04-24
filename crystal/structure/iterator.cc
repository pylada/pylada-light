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

struct StructureIterator
{
  PyObject_HEAD;
  PyStructureObject* parent; 
  std::vector<Atom>::const_iterator i_first;
  bool is_first;
};
//! Returns Self.
PyObject* structureiterator_iter(PyObject* _self) { Py_INCREF(_self); return _self; }
//! Returns next object.
PyObject* structureiterator_next(StructureIterator* _self);
//! Function to deallocate a string atom.
void structureiterator_dealloc(StructureIterator *_self);
//! Creates iterator.
PyObject* structureiterator_create(PyStructureObject* _self);


// Returns next object.
PyObject* structureiterator_next(StructureIterator* _self)
{
  if(_self->i_first != _self->parent->atoms.end()) 
  {
    if(_self->is_first) _self->is_first = false; else ++_self->i_first;
    if(_self->i_first != _self->parent->atoms.end()) return _self->i_first->new_ref();
  }
  PyErr_SetNone(PyExc_StopIteration);
  return NULL;
}
//! Function to deallocate a string atom.
void structureiterator_dealloc(StructureIterator *_self)
{
  PyObject* dummy = (PyObject*)_self->parent;
  _self->parent = NULL;
  Py_XDECREF(dummy);
}
// Creates iterator.
PyObject* structureiterator_create(PyStructureObject* _in)
{
  StructureIterator *result = PyObject_New(StructureIterator, structureiterator_type());
  if(result == NULL) return NULL;
  result->parent = _in;
  Py_INCREF(result->parent);
  new(&result->i_first) std::vector<Atom>::iterator(_in->atoms.begin());
  result->is_first = true;
  return (PyObject*) result;
}
