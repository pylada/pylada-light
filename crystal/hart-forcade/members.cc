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

//! Returns a deepcopy of the atom.
PyObject* hftransform_copy(PyHFTObject* _self)
  { return (PyObject*) copy_hftransform(_self, NULL); }
//! Implements shallow copy.
PyObject* hftransform_shallowcopy(PyHFTObject* _self)
  { Py_INCREF(_self); return (PyObject*)_self; }
//! Implements getstate for pickling.
PyObject* hftransform_getstate(PyHFTObject* _self);
//! Implements setstate for pickling.
PyObject* hftransform_setstate(PyHFTObject* _self, PyObject *_dict);
//! Implements reduce for pickling.
PyObject* hftransform_reduce(PyHFTObject* _self);
//! Computes Z-group indices of position \a _pos.
PyObject* hftransform_indices(PyHFTObject* _self, PyObject* _args);
// Computes flat hf index from non-flat hf index.
PyObject* hftransform_flatten_indices( PyHFTObject* _self,
                                                 PyObject* _args, PyObject *_kwargs );
//! Computes flat index into Z-group from atomic position.
PyObject* hftransform_flat_index( PyHFTObject* _self,
                                            PyObject* _args, PyObject *_kwargs );


// Implements __reduce__ for pickling.
PyObject* hftransform_reduce(PyHFTObject* _self)
{
  // Creates return tuple of three elements.
  python::Object type = PyObject_Type((PyObject*)_self);
  if(not type) return NULL;
  // Second element is a null tuple, argument to the callable type above.
  python::Object tuple = PyTuple_New(0);
  if(not tuple) return NULL;
  // Third element is the state of this object.
  char getstate[] = "__getstate__";
  python::Object state = PyObject_CallMethod((PyObject*)_self, getstate, NULL, NULL);
  if(not state) return NULL;

  return PyTuple_Pack(3, type.borrowed(), tuple.borrowed(), state.borrowed());
}

// Implements getstate for pickling.
PyObject* hftransform_getstate(PyHFTObject* _self)
{
  // get cell attribute.
  python::Object cell = hftransform_gettransform(_self, NULL);
  if(not cell) return NULL;
  // get scale attribute.
  python::Object scale = hftransform_getquotient(_self, NULL);
  if(not scale) return NULL;
  return PyTuple_Pack(2, cell.borrowed(), scale.borrowed());
}

// Implements setstate for pickling.
PyObject* hftransform_setstate(PyHFTObject* _self, PyObject *_tuple)
{
  if(not PyTuple_Check(_tuple))
  {
    PYLADA_PYERROR(TypeError, "Expected state to be a tuple.");
    return NULL;
  }
  if(PyTuple_Size(_tuple) != 2)
  {
    PYLADA_PYERROR(TypeError, "Expected state to be a 2-tuple.");
    return NULL;
  }
  // first cell and scale.
  if(not python::numpy::convert_to_matrix(PyTuple_GET_ITEM(_tuple, 0), _self->transform)) return NULL;
  if(not python::numpy::convert_to_vector(PyTuple_GET_ITEM(_tuple, 1), _self->quotient)) return NULL;
  Py_RETURN_NONE;
}

// defines macros also used in hf.
#include "macro.hpp"
// Computes flat hf index from non-flat hf index.
PyObject* hftransform_flatten_indices( PyHFTObject* _self,
                                              PyObject* _args, PyObject *_kwargs )
{
  int site = -1, i = 0, j = 0, k = 0;
  static char *kwlist[] = { const_cast<char*>("i"), const_cast<char*>("j"),
                            const_cast<char*>("k"),
                            const_cast<char*>("site"), NULL };
  if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "iii|i:pos_index",
                                      kwlist, &i, &j, &k, &site) )
    return NULL;
  return PyInt_FromLong( 
            site == -1 ? 
              k  + _self->quotient(2) * (j + _self->quotient(1) * i):
              k  + _self->quotient(2) * (j + _self->quotient(1) 
                                           * (i + site * _self->quotient(0)) )
          );
}
//! Computes flat hf index from non-flat hf index, including sites.
PyObject* hftransform_flat_index( PyHFTObject* _self,
                                            PyObject* _args, PyObject *_kwargs )
{
  PyObject *posatom = NULL;
  int site = -1;
  static char *kwlist[] = { const_cast<char*>("indices"), const_cast<char*>("site"), NULL };
  if(not PyArg_ParseTupleAndKeywords(_args, _kwargs, "O|i:flat_index", kwlist, &posatom, &site) )
    return NULL;
  math::rVector3d pos;
  if(not python::numpy::convert_to_vector(posatom, pos)) return NULL;
  PYLADA_HFTRANSFORM_SHARED1(_self->quotient, _self->transform, pos, PYLADA_PYERROR, return NULL);
  PYLADA_HFTRANSFORM_SHARED0(_self->quotient, vector_result, site);
  return PyInt_FromLong(flat_result);
}
// Computes hf indices of position \a _pos.
PyObject* hftransform_indices(PyHFTObject* _self, PyObject* _args)
{
  math::rVector3d pos;
  if(not python::numpy::convert_to_vector(_args, pos)) return NULL;
  PYLADA_HFTRANSFORM_SHARED1(_self->quotient, _self->transform, pos, PYLADA_PYERROR, return NULL);
  return python::numpy::wrap_to_numpy(vector_result);
}
// undefs macros also used in hf.
#include "macro.hpp"
