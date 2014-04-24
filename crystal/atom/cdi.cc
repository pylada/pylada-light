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
void pylada_atom_dealloc(PyAtomObject *_self);
//! Function to initialize a string atom.
int pylada_atom_init(PyAtomObject* _self, PyObject* _args, PyObject *_kwargs);
//! Traverses to back-reference.
int pylada_atom_traverse(PyAtomObject *_self, visitproc visit, void *arg)
  { Py_VISIT(_self->pydict); Py_VISIT(_self->type); return 0; }
//! Clears back reference.
int pylada_atom_gcclear(PyAtomObject *self) { Py_CLEAR(self->pydict); Py_CLEAR(self->type); return 0; }


// Function to deallocate a string atom.
void pylada_atom_dealloc(PyAtomObject *_self)
{
  if(_self->weakreflist != NULL)
    PyObject_ClearWeakRefs((PyObject *) _self);
 
  pylada_atom_gcclear(_self);

  // Calls c++ destructor explicitely.
  PyTypeObject *ob_type = _self->ob_type;
  _self->~PyAtomObject();

  ob_type->tp_free((PyObject*)_self);
}

// Function to initialize a string atom.
int pylada_atom_init(PyAtomObject* _self, PyObject* _args, PyObject *_kwargs)
{
  Py_ssize_t const N = PyTuple_Size(_args);

  if(N > 0) 
  {
    if(N < 3)
    {
      PYLADA_PYERROR(TypeError, "Atom(...): Expect at least three arguments.");
      return -1;
    }
    for(Py_ssize_t i(0); i < 3; ++i)
    {
      PyObject *item = PyTuple_GET_ITEM(_args, i);
      if(PyInt_Check(item) == 1)  _self->pos[i] = PyInt_AS_LONG(item);
      else if(PyFloat_Check(item) == 1)  _self->pos[i] = PyFloat_AS_DOUBLE(item);
      else
      {
        PYLADA_PYERROR( TypeError,
                      "Atom(...): Expects the first three arguments to be the position. "
                      "Or, give everything as keywords" );
        return -1;
      }
    }
    if(N == 4) 
    {
      // deletes current type.
      PyObject* dummy = _self->type;
      _self->type = PyTuple_GET_ITEM(_args, 3);
      Py_INCREF(_self->type);
      Py_DECREF(dummy);
    }
    else if(N > 4)
    {
      // deletes current type.
      PyObject* dummy = _self->type;
      PyObject *slice = PyTuple_GetSlice(_args, 3, N);
      if(slice == NULL) return -1;
      _self->type = PyList_New(N-3);
      if(_self->type == NULL)
      {
        _self->type = dummy;
        return -1;
      }
      Py_DECREF(dummy);
      if(PyList_SetSlice(_self->type, 0, N-3, slice) < -1) return -1;
    }
  }

  // check for keywords.
  if(_kwargs == NULL) return 0;

  // Sanity check first.
  if( N >= 3 and PyDict_GetItemString(_kwargs, "pos") != NULL)
  {
    PYLADA_PYERROR(TypeError, "Atom(...): Cannot set position both from keyword and argument.");
    return -1;
  }
  // Sanity check first.
  if( N >= 4 and PyDict_GetItemString(_kwargs, "type") != NULL)
  {
    PYLADA_PYERROR(TypeError, "Atom(...): Cannot set type both from keyword and argument.");
    return -1;
  }

  PyObject *key, *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(_kwargs, &pos, &key, &value)) 
    if(PyObject_SetAttr((PyObject*)_self, key, value) < 0) return -1;
  return 0;
}
