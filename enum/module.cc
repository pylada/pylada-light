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
#define PY_ARRAY_UNIQUE_SYMBOL pylada_enum_ARRAY_API
#include <python/include_numpy.h>

#include <errors/exceptions.h>
#include <math/math.h>
#include <python/python.h>
#include "ndimiterator.h"
#include "fciterator.h"
#include "manipulations.h"
#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
# define PyMODINIT_FUNC void
#endif

namespace Pylada
{
  namespace enumeration
  {
    //! Checks whether a numpy array is an integer. 
    PyObject* is_integer(PyObject *_module, PyObject *_array)
    {
      if(not PyArray_Check(_array))
      {
        PYLADA_PYERROR(TypeError, "input must be a numpy array.\n");
        return NULL;
      }
      if(PyArray_ISINTEGER((PyArrayObject*)_array)) Py_RETURN_TRUE;
      python::Object iterator(PyArray_IterNew(_array));
      if(not iterator) return NULL;
      PyObject* i_iterator = iterator.borrowed();
      int const type = PyArray_TYPE((PyArrayObject*)_array);
#     ifdef PYLADA_IFTYPE
#       error PYLADA_IFTYPE already defined
#     endif
#     define PYLADA_IFTYPE(TYPENUM, TYPE)                                        \
        if(type == TYPENUM)                                                    \
        {                                                                      \
          while(PyArray_ITER_NOTDONE(i_iterator))                              \
          {                                                                    \
            TYPE const a = *((TYPE*) PyArray_ITER_DATA(i_iterator));           \
            if(not math::is_null(std::floor(a+1e-5) - a)) Py_RETURN_FALSE;     \
            PyArray_ITER_NEXT(i_iterator);                                     \
          }                                                                    \
        }
      PYLADA_IFTYPE(NPY_FLOAT, python::numpy::type<npy_float>::np_type)
      else PYLADA_IFTYPE(NPY_DOUBLE, python::numpy::type<npy_double>::np_type)
      //else PYLADA_IFTYPE(NPY_LONGDOUBLE, python::numpy::type<npy_longdouble>::np_type)
#     undef PYLADA_WITH_DATA_TYPE
      Py_RETURN_TRUE;
    }
    PyObject* lexcompare(PyObject *_module, PyObject *_args)
    {
#     ifdef PYLADA_DEBUG
      if(_args == NULL)
      {
        PYLADA_PYERROR(TypeError, "_lexcompare expects two arguments.");
        return NULL;
      }
      if(PyTuple_Size(_args) != 2)
      {
        PYLADA_PYERROR(TypeError, "_lexcompare expects two arguments.");
        return NULL;
      }
#     endif
      PyArrayObject *first = (PyArrayObject*) PyTuple_GET_ITEM(_args, 0);
      PyArrayObject *second = (PyArrayObject*) PyTuple_GET_ITEM(_args, 1);
#     ifdef PYLADA_DEBUG
      if(not PyArray_Check(first))
      {
        PYLADA_PYERROR(TypeError, "First argument to _lexcompare is not a numpy array.");
        return NULL;
      }
      if(not PyArray_Check(second))
      {
        PYLADA_PYERROR(TypeError, "Second argument to _lexcompare is not a numpy array.");
        return NULL;
      }
      if(PyArray_NDIM(first) != 1 or PyArray_NDIM(second) != 1)
      {
        PYLADA_PYERROR(TypeError, "_lexcompare arguments should be 1d array.");
        return NULL;
      }
      if(PyArray_DIM(first, 0) != PyArray_DIM(second, 0))
      {
        PYLADA_PYERROR(TypeError, "_lexcompare arguments should have the same size.");
        return NULL;
      }
      if( PyArray_TYPE(first) != python::numpy::type<t_ndim>::value
          or PyArray_TYPE(second) != python::numpy::type<t_ndim>::value )
      {
        PYLADA_PYERROR(TypeError, "Wrong kind for _lexcompare arguments.");
        return NULL;
      }
#     endif

      npy_intp const stridea = PyArray_STRIDE(first, 0);
      npy_intp const strideb = PyArray_STRIDE(second, 0);
      npy_intp const n = PyArray_DIM(first, 0);
      char * dataa = (char*) PyArray_DATA(first);
      char * datab = (char*) PyArray_DATA(second);
      for(npy_intp i(0); i < n; ++i, dataa += stridea, datab += strideb)
      {
        t_ndim const a = *((t_ndim*) dataa);
        t_ndim const b = *((t_ndim*) datab);
        if(a > b) { return PyInt_FromLong(1); }
        else if(a < b) { return PyInt_FromLong(-1); }
      }
      return PyInt_FromLong(0); 
    }
    //! Methods table for crystal module.
    static PyMethodDef methods_table[] = {
        {"is_integer",  is_integer, METH_O,
         "True if the numpy array is an integer.\n\n"
         "This method takes a single argument which *must* be a *numpy* array.\n"},
        {"_lexcompare",  lexcompare, METH_VARARGS,
         "Lexicographic compare of two numpy arrays.\n\n"
         "Read the code for this function. If you do not understand it, do not\n"
         "use it.\n\n"
         ":returns:\n\n"
         "   - a > b: 1\n"
         "   - a == b: 0\n"
         "   - a < b: -1\n" },
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };
  }
}

PyMODINIT_FUNC initcppwrappers(void) 
{
  char const doc[] =  "Wrapper around C++ enumeration methods.";
  PyObject* module = Py_InitModule3("cppwrappers", Pylada::enumeration::methods_table, doc);
  if(not module) return;
  import_array(); // needed for NumPy 
  if(not Pylada::python::import()) return;
  if(not Pylada::math::import()) return;

  if (PyType_Ready(Pylada::enumeration::ndimiterator_type()) < 0) return;
  Py_INCREF(Pylada::enumeration::ndimiterator_type());
  if (PyType_Ready(Pylada::enumeration::manipulations_type()) < 0) return;
  Py_INCREF(Pylada::enumeration::manipulations_type());
  if (PyType_Ready(Pylada::enumeration::fciterator_type()) < 0) return;
  Py_INCREF(Pylada::enumeration::fciterator_type());

  PyModule_AddObject(module, "NDimIterator", (PyObject *)Pylada::enumeration::ndimiterator_type());
  PyModule_AddObject(module, "Manipulations", (PyObject *)Pylada::enumeration::manipulations_type());
  PyModule_AddObject(module, "FCIterator", (PyObject *)Pylada::enumeration::fciterator_type());
}
