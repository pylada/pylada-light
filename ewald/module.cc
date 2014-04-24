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
#define PY_ARRAY_UNIQUE_SYMBOL pylada_ewald_ARRAY_API
#include <python/include_numpy.h>

#include <algorithm>

#include <errors/exceptions.h>
#include <python/python.h>
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
# define PyMODINIT_FUNC void
#endif

#include "ewald.h"

namespace Pylada
{
  namespace pcm
  {
    //! Methods table for crystal module.
    static PyMethodDef methods_table[] = {
        {"ewald", (PyCFunction)ewald, METH_KEYWORDS,
         "Performs an ewald summation." },
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };
  }
}

PyMODINIT_FUNC initcppwrappers(void) 
{
  import_array(); // needed for NumPy 

  char const doc[] =  "Wrapper around C++/fortan point-ion models methods.";
  PyObject* module = Py_InitModule3("cppwrappers", Pylada::pcm::methods_table, doc);
  if(not module) return;
  import_array();
  if(not Pylada::python::import()) return;
}
