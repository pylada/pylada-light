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

#include "PyladaConfig.h"

#include <Python.h>
#include <python/include_numpy.h>
#include "../crystal.h"

using namespace Pylada::crystal;
PyObject* get_static_object(PyObject* _module, PyObject*)
{ 
  Pylada::python::Object module = PyImport_ImportModule("_atom_self");
  if(not module) return NULL;
  PyObject *result = PyObject_GetAttrString(module.borrowed(), "_atom");
  return result;
}
PyObject* set_static_object(PyObject* _module, PyObject *_object)
{
  if(not check_atom(_object))
  {
    PYLADA_PYERROR(TypeError, "Wrong type.");
    return NULL;
  }
  Pylada::python::Object module = PyImport_ImportModule("_atom_self");
  if(not module) return NULL;
  if( PyObject_SetAttrString(module.borrowed(), "_atom", _object) < 0) return NULL;
  Py_RETURN_NONE;
}

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
# define PyMODINIT_FUNC void
#endif

#ifdef PYLADA_DECLARE
#  error PYLADA_DECLARE already defined.
#endif
#define PYLADA_DECLARE(name, args) {#name, (PyCFunction)name, METH_ ## args, ""} 

static PyMethodDef methods[] = { 
  PYLADA_DECLARE(get_static_object, NOARGS),
  PYLADA_DECLARE(set_static_object, O),
  {NULL},
};

#undef PYLADA_DECLARE

PyMODINIT_FUNC init_atom_self(void) 
{
  PyObject* module = Py_InitModule("_atom_self", methods);
  if(not module) return; 
  if(not Pylada::python::import()) return;
  if(not Pylada::crystal::import()) return;
  Atom satom; 
  PyModule_AddObject(module, "_atom", (PyObject *)satom.new_ref());
  satom.release();
}
