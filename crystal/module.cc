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

//PG
#include <python/include_numpy.h>
//#include <numpy/arrayobject.h>

#include <cmath>
#include <iterator> 
#include <limits>
#include <list>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

//PG
//#include <Eigen/LU> 

#include <errors/exceptions.h>
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
# define PyMODINIT_FUNC void
#endif
#define PYLADA_CRYSTAL_MODULE 0
#include "crystal.h"

namespace Pylada
{
  namespace crystal
  {
    namespace 
    {
#     include "atom/pybase.cc"
#     include "structure/pybase.cc"
#     include "hart-forcade/pybase.cc"
#     include "utilities.cc"
#     include "map_sites.cc"
#     include "equivalent_structures.cc"
#     include "primitive.cc"
#     include "space_group.cc"
#     include "neighbors.cc"
#     include "coordination_shells.cc"
#     include "confsplit.cc"
#     include "periodic_dnc.cc"

#     include "methods.cc"
    }
  }
}



PyMODINIT_FUNC initcppwrappers(void) 
{
  using namespace Pylada::crystal;
  static void *api_capsule[PYLADA_SLOT(crystal)];
  PyObject *c_api_object;

  char const doc[] =  "Wrapper around C++ atom/structure class and affiliates.";
  PyObject* module = Py_InitModule3("cppwrappers", methods_table, doc);
  if(not module) return;
  if(not Pylada::python::import()) return;
  if(not Pylada::math::import()) return;
  import_array();

  /* Initialize the C API pointer array */
# undef PYLADA_CRYSTALMODULE_H
# include "crystal.h"

  /* Create a Capsule containing the API pointer array's address */
# ifdef PYLADA_PYTHONTWOSIX
    c_api_object = PyCObject_FromVoidPtr((void *)api_capsule, NULL);
# else
    static const char name[] = "pylada.crystal.cppwrappers._C_API";
    c_api_object = PyCapsule_New((void *)api_capsule, name, NULL);
# endif
  if (c_api_object != NULL) PyModule_AddObject(module, "_C_API", c_api_object);

  if (PyType_Ready(atom_type()) < 0) return;
  if (PyType_Ready(structure_type()) < 0) return;
  if (PyType_Ready(structureiterator_type()) < 0) return;
  if (PyType_Ready(hftransform_type()) < 0) return;

  Py_INCREF(atom_type());
  Py_INCREF(structure_type());
  Py_INCREF(structureiterator_type());
  Py_INCREF(hftransform_type());

  PyModule_AddObject(module, "Atom", (PyObject *)atom_type());
  PyModule_AddObject(module, "Structure", (PyObject *)structure_type());
  PyModule_AddObject(module, "HFTransform", (PyObject *)hftransform_type());
}
