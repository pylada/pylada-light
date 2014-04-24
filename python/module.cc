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
#include <structmember.h>
#include "include_numpy.h"

#include <errors/exceptions.h>
#include <math/math.h>

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
# define PyMODINIT_FUNC void
#endif
#define PYLADA_PYTHON_MODULE 0
#include "python.h"

namespace Pylada
{
  namespace python
  {
    namespace 
    {
#     include "object.cc"
#     include "quantity.cc"
    }
  }
}



PyMODINIT_FUNC initcppwrappers(void) 
{
  using namespace Pylada::python;
  using namespace Pylada;
  static void *api_capsule[PYLADA_SLOT(python)];
  static PyMethodDef methods_table[] = { {NULL, NULL, 0, NULL} };
  PyObject *c_api_object;

  char const doc[] =  "Wrapper around basic C++ helper functions.\n\n"
                      "This module only contains a capsule for cpp functions.\n";
  PyObject* module = Py_InitModule3("cppwrappers", methods_table, doc);
  if(not module) return;

  import_array(); // needed for NumPy 

  /* Initialize the C API pointer array */
# undef PYLADA_PYTHON_PYTHON_H
# include "python.h"

  /* Create a Capsule containing the API pointer array's address */
  # ifdef PYLADA_PYTHONTWOSIX
      c_api_object = PyCObject_FromVoidPtr((void *)api_capsule, NULL);
  # else
      static const char name[] = "pylada.cppwrappers._C_API";
      c_api_object = PyCapsule_New((void *)api_capsule, name, NULL);
  # endif
  if (c_api_object != NULL) PyModule_AddObject(module, "_C_API", c_api_object);
}
