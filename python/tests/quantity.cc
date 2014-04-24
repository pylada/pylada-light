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

#include <iostream>

#include "../python.h"


using namespace Pylada::python;
PyObject* is_quantity(PyObject *_module, PyObject *_in)
{ 
  if(not check_quantity(_in)) Py_RETURN_FALSE;
  Py_RETURN_TRUE;
}
PyObject* fromC(PyObject *_module, PyObject *_args)
{
  double pynumber;
  char *pyunits;
  if(not PyArg_ParseTuple(_args, "ds", &pynumber, &pyunits)) return NULL;
  return fromC_quantity(pynumber, std::string(pyunits));
}
PyObject* fromPy(PyObject *_module, PyObject *_args)
{
  PyObject *number;
  PyObject *units;
  if(not PyArg_ParseTuple(_args, "OO", &number, &units)) return NULL;
  return fromPy_quantity(number, units);
}
PyObject* get_angstrom(PyObject *_module, PyObject *_in)
{
  Pylada::types::t_real result(get_quantity(_in, "angstrom"));
  if(std::abs(result) < 1e-8 and PyErr_Occurred())
  {
    PyErr_Clear();
    Py_RETURN_NONE;
  }
  return PyFloat_FromDouble(result);
}

PyObject* as_real(PyObject *_module, PyObject *_in)
{
  Pylada::types::t_real result(get_quantity(_in));
  if(std::abs(result) < 1e-8 and PyErr_Occurred())
  {
    PyErr_Clear();
    Py_RETURN_NONE;
  }
  return PyFloat_FromDouble(result);
}
PyObject* get_as(PyObject *_module, PyObject *_args)
{
  PyObject *number;
  PyObject *units;
  if(not PyArg_ParseTuple(_args, "OO", &number, &units)) return NULL;
  Pylada::types::t_real result(get_quantity(number, units));
  if(std::abs(result) < 1e-8 and PyErr_Occurred())
  {
    PyErr_Clear();
    Py_RETURN_NONE;
  }
  return PyFloat_FromDouble(result);
}
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
# define PyMODINIT_FUNC void
#endif

#ifdef PYLADA_DECLARE
#  error PYLADA_DECLARE already defined.
#endif
#define PYLADA_DECLARE(name, args) {#name, (PyCFunction)name, METH_ ## args, ""} 

static PyMethodDef methods[] = { 
  PYLADA_DECLARE(is_quantity, O),
  PYLADA_DECLARE(fromC, VARARGS),
  PYLADA_DECLARE(fromPy, VARARGS),
  PYLADA_DECLARE(get_angstrom, O),
  PYLADA_DECLARE(as_real, O),
  PYLADA_DECLARE(get_as, VARARGS),
  {NULL},
};

#undef PYLADA_DECLARE

PyMODINIT_FUNC init_quantity(void) 
{
  PyObject* module = Py_InitModule("_quantity", methods);
  if(not module) return;
  import_array();
  if(not Pylada::python::import()) return;
}
