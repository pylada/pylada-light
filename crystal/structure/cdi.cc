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

//! Function to deallocate a string atom.
void structure_dealloc(PyStructureObject *_self);
//! Function to initialize a string atom.
int structure_init(PyStructureObject* _self, PyObject* _args, PyObject *_kwargs);
//! Traverses to back-reference.
int structure_traverse(PyStructureObject *_self, visitproc _visit, void *_arg);
//! Clears back reference.
int structure_gcclear(PyStructureObject *_self);

// Function to deallocate a string atom.
void structure_dealloc(PyStructureObject *_self)
{
  if(_self->weakreflist != NULL)
    PyObject_ClearWeakRefs((PyObject *) _self);
 
  structure_gcclear(_self);

  if(_self->scale)
  {
    PyObject *dummy = _self->scale;
    _self->scale = NULL;
    Py_DECREF(dummy);
  }

  // calls destructor explicitely.
  PyTypeObject* ob_type = _self->ob_type;
  _self->~PyStructureObject();

  ob_type->tp_free((PyObject*)_self);
}

// Function to initialize an atom.
int structure_init(PyStructureObject* _self, PyObject* _args, PyObject *_kwargs)
{
  Py_ssize_t const N = PyTuple_Size(_args);
  
  if(_self->scale)
  {
    PyObject *dummy = _self->scale;
    _self->scale = NULL;
    Py_DECREF(dummy);
  }
  _self->scale = python::fromC_quantity(1,  "angstrom");

  if(N != 0 and N != 1 and N != 9 and N != 3)
  {
    PYLADA_PYERROR(TypeError, "Unexpected argument: arguments should represent the cell "
                            "and be given as a matrix, or as a series of 9 numbers." );
    return -1;
  }
  if(N != 0 and _kwargs != NULL and PyDict_GetItemString(_kwargs, "cell") != NULL)
  {
    PYLADA_PYERROR(TypeError, "Cell given as both argument and keyword.");
    return -1;
  }
  if(N == 1)
  {
    PyObject *item = PyTuple_GetItem(_args, 0);
    if(item == NULL) return -1;
    if(structure_setcell(_self, item, NULL) < 0) return -1;
  }
  else if(N == 9 and structure_setcell(_self, _args, NULL) < 0) return -1;
  else if(N == 3 and structure_setcell(_self, _args, NULL) < 0) return -1;

  if(_kwargs == NULL) return 0;
  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(_kwargs, &pos, &key, &value)) 
    if(PyObject_SetAttr((PyObject*)_self, key, value) < 0) return -1;
  return 0;
}

int structure_traverse(PyStructureObject *self, visitproc visit, void *arg)
{
  Py_VISIT(self->pydict);
  std::vector<Atom>::const_iterator i_first = self->atoms.begin();
  std::vector<Atom>::const_iterator const i_end = self->atoms.end();
  for(; i_first != i_end; ++i_first) Py_VISIT(i_first->borrowed());
  return 0;
}

int structure_gcclear(PyStructureObject *self)
{ 
  Py_CLEAR(self->pydict);
  self->atoms.clear();
  return 0;
}
