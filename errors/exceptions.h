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

#ifndef PYLADA_PYTHON_EXCEPTIONS_H
#define PYLADA_PYTHON_EXCEPTIONS_H
#include "PyladaConfig.h"

#include <Python.h>
#include <root_exceptions.h>

namespace Pylada
{
  namespace error
  {
    //! Attribute error thrown explicitely by pylada.
    struct AttributeError: virtual root {};
    //! Key error thrown explicitely by pylada.
    struct KeyError: virtual out_of_range, virtual root {};
    //! Value error thrown explicitely by pylada.
    struct ValueError: virtual root {};
    //! Index error thrown explicitely by pylada.
    struct IndexError: virtual root {};
    //! Argument error thrown explicitely by pylada.
    struct TypeError: virtual root {};
    //! Not implemented error thrown explicitely by pylada.
    struct NotImplementedError: virtual root {};
    //! Subclasses python's ImportError.
    struct ImportError: virtual root {};

#   ifdef PYLADA_PYERROR
#     error PYLADA_PYERROR already  defined. 
#   endif
#   ifdef PYLADA_PYERROR_FORMAT
#     error PYLADA_PYERROR already  defined. 
#   endif
#   ifdef PYLADA_PYTHROW
#     error PYLADA_PYERROR already  defined. 
#   endif
    //! \def PYLADA_PYERROR(EXCEPTION, MESSAGE)
    //!      Raises a python exception with the interpreter, but no c++ exception.
    //!      EXCEPTION should be an unqualified declared in python/exceptions.h.
#   define PYLADA_PYERROR(EXCEPTION, MESSAGE)                                       \
      {                                                                           \
        PyObject* err_module = PyImport_ImportModule("pylada.error");               \
        if(err_module)                                                            \
        {                                                                         \
          PyObject *err_result = PyObject_GetAttrString(err_module, #EXCEPTION);  \
          if(not err_result) Py_DECREF(err_module);                               \
          else                                                                    \
          {                                                                       \
            PyErr_SetString(err_result, MESSAGE);                                 \
            Py_DECREF(err_module);                                                \
            Py_DECREF(err_result);                                                \
          }                                                                       \
        }                                                                         \
      }
#   define PYLADA_PYTHROW(EXCEPTION, MESSAGE)                                       \
      {                                                                           \
        PYLADA_PYERROR(EXCEPTION, MESSAGE);                                         \
        BOOST_THROW_EXCEPTION(error::EXCEPTION());      \
      }

    //! \def PYLADA_PYERROR(EXCEPTION, MESSAGE)
    //!      Raises a python exception with a formatted message, but no c++ exception.
    //!      For formatting, see PyErr_Format from the python C API.
    //!      EXCEPTION should be an unqualified declared in python/exceptions.h.
#   define PYLADA_PYERROR_FORMAT(EXCEPTION, MESSAGE, OTHER) \
      {                                                                           \
        PyObject* err_module = PyImport_ImportModule("pylada.error");               \
        if(err_module)                                                            \
        {                                                                         \
          PyObject *err_result = PyObject_GetAttrString(err_module, #EXCEPTION);  \
          if(not err_result) Py_DECREF(err_module);                               \
          else                                                                    \
          {                                                                       \
            PyErr_Format(err_result, MESSAGE, OTHER);                             \
            Py_DECREF(err_module);                                                \
            Py_DECREF(err_result);                                                \
          }                                                                       \
        }                                                                         \
      }
  }
}
# endif 
