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
#include "../include_numpy.h"

#include "../python.h"


using namespace Pylada::python;
PyObject* represent(PyObject *_module, PyObject *_in)
{ 
  Object o = Object::acquire(_in);
  std::ostringstream sstr;
  try { sstr << o; }
  catch(std::exception &_e)
  {
    PYLADA_PYERROR(internal, "caught error");
    return NULL;
  }
  return PyString_FromString(sstr.str().c_str());
}

PyObject* add_attribute(PyObject *_module, PyObject* _args)
{
  Object o = Object::acquire(PyTuple_GET_ITEM(_args, 0));
  PyObject_SetAttr(o.borrowed(), PyTuple_GET_ITEM(_args, 1), PyTuple_GET_ITEM(_args, 2));
  Py_RETURN_NONE;
}
PyObject* callme(PyObject *_module, PyObject* _in)
{
  Object o = Object::acquire(_in);
  return PyObject_CallObject(o.borrowed(), NULL);
}

PyObject* equality(PyObject* _module, PyObject* _args)
{
  Object a = Object::acquire(PyTuple_GET_ITEM(_args, 0));
  Object b = Object::acquire(PyTuple_GET_ITEM(_args, 1));
  if(a == b) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
# define PyMODINIT_FUNC void
#endif

#ifdef PYLADA_DECLARE
#  error PYLADA_DECLARE already defined.
#endif
#define PYLADA_DECLARE(name, args) {#name, (PyCFunction)name, METH_ ## args, ""} 

static PyMethodDef methods[] = { 
  PYLADA_DECLARE(represent, O),
  PYLADA_DECLARE(add_attribute, VARARGS),
  PYLADA_DECLARE(callme, O),
  PYLADA_DECLARE(equality, VARARGS),
  {NULL},
};

#undef PYLADA_DECLARE

PyMODINIT_FUNC init_pyobject(void) 
{
  PyObject* module = Py_InitModule("_pyobject", methods);
  if(not module) return;
  if(not Pylada::python::import()) return;
}
