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

#ifndef PYLADA_PYTHON_PYTHON_H
#define PYLADA_PYTHON_PYTHON_H
#ifndef __cplusplus
# error Pylada requires a cpp compiler
#endif

#ifndef PYLADA_PYTHON_MODULE
#  define PYLADA_PYTHON_MODULE 100
#endif

#if PYLADA_PYTHON_MODULE != 1

# include "PyladaConfig.h"
# include <Python.h>
# include <structmember.h>
# ifndef PYLADA_PYTHONTWOSIX
#   if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION < 7
#     define PYLADA_PYTHONTWOSIX
#   endif
# endif

# include <vector>
# include <string>
# include <iostream>

# include <boost/mpl/int.hpp>
# include <boost/type_traits/is_floating_point.hpp>
# include <boost/preprocessor/arithmetic/inc.hpp>
# include <python/ppslot.hpp>
# define BOOST_PP_VALUE 0
# include PYLADA_ASSIGN_SLOT(python)

//PG
# include <Eigen/Core>

# include <errors/exceptions.h>
# include "types.h"

  namespace Pylada
  {
    namespace python
    {

#     if PYLADA_PYTHON_MODULE == 100
        /* This section is used in modules that use pylada.python's API */
#       ifdef PYLADA_NO_IMPORT
          extern
#       endif 
        void **api_capsule;
        
        namespace 
        {
          // Return -1 on error, 0 on success.
          // PyCapsule_Import will set an exception if there's an error.
          inline bool import(void)
          {
            PyObject *module = PyImport_ImportModule("pylada.cppwrappers");
            if(not module) return false;
#           ifdef PYLADA_PYTHONTWOSIX
              PyObject* c_api_object = PyObject_GetAttrString(module, "_C_API");
	      if (c_api_object == NULL) { Py_DECREF(module); return false; }
              if (PyCObject_Check(c_api_object))
                api_capsule = (void **)PyCObject_AsVoidPtr(c_api_object);
              Py_DECREF(c_api_object);
#           else
              api_capsule = (void **)PyCapsule_Import("pylada.cppwrappers._C_API", 0);
#           endif
            Py_DECREF(module);
            return api_capsule != NULL;
          }
        }
#     endif
#else
# define BOOST_PP_VALUE 0
# include PYLADA_ASSIGN_SLOT(python)
#endif

#if PYLADA_PYTHON_MODULE != 1
#  ifdef PYLADA_INLINE
#    error PYLADA_INLINE already defined
#  endif
#  if PYLADA_PYTHON_MODULE == 100
#    define PYLADA_INLINE inline
#  elif PYLADA_PYTHON_MODULE == 0
#    define PYLADA_INLINE
#  endif
#  ifdef PYLADA_END
#    error PYLADA_END already defined
#  elif PYLADA_PYTHON_MODULE == 0
#    define PYLADA_END(X) ;
#  elif PYLADA_PYTHON_MODULE == 100
#    define PYLADA_END(X) { X }
#  endif
#endif

#if PYLADA_PYTHON_MODULE != 1
  class Object;
  namespace
  {
#endif
#if PYLADA_PYTHON_MODULE != 1
  //! Object reset function.
  //! Declared as friend to object so that it can be linked at runtime.
  PYLADA_INLINE void object_reset(PyObject*& _object, PyObject *_in)
    PYLADA_END( return ( *(void(*)(PyObject*&, PyObject*))
                       api_capsule[PYLADA_SLOT(python)])(_object, _in); ) 
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)((void(*)(PyObject*&, PyObject*)) object_reset);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)
  
#if PYLADA_PYTHON_MODULE != 1
  PYLADA_INLINE bool object_equality_op(Object const& _self, Object const &_b)
    PYLADA_END( return ( *(bool(*)(Object const&, Object const&))
                       api_capsule[PYLADA_SLOT(python)])(_self, _b); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)object_equality_op;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)

#if PYLADA_PYTHON_MODULE != 1
  PYLADA_INLINE std::ostream& operator<<(std::ostream &_stream, Object const &_ob)
    PYLADA_END( return ( (std::ostream&(*)(std::ostream&, Object const&)) 
                       api_capsule[PYLADA_SLOT(python)] )(_stream, _ob); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void*)
       ( (std::ostream&(*)(std::ostream&, Object const&)) operator<< );
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)

#if PYLADA_PYTHON_MODULE != 1
  }
#endif

// in namespace Pylada::python, but not anonymous.
# include "object.h"

#if PYLADA_PYTHON_MODULE != 1
  namespace
  {
    //! \brief Acquires a reference to an object.
    //! \details Input is XINCREF'ed before the return wrappers is created. 
    inline Object acquire(PyObject *_in) { Py_XINCREF(_in); return Object(_in); }
    //! \brief Steals a reference to an object.
    //! \details Input is XINCREF'ed before the return wrappers is created. 
    inline Object steal(PyObject *_in) { return Object(_in); }
    
    //! \brief Dumps representation of an object.
    //! \details Will throw c++ exceptions if python calls fail. Does not clear
    //!          python exceptions.
    inline std::ostream& operator<< (std::ostream &stream, PyObject* _ob)
      { return stream << Object::acquire(_ob); }
  }
#endif

// in namespace Pylada::python, but not anonymous.
# include "random_access_list_iterator.h"
# include "random_access_tuple_iterator.h"

#if PYLADA_PYTHON_MODULE != 1
  namespace numpy
  {
    namespace 
    {
#endif

#     include "numpy_types.h"
#     include "wrap_numpy.h"

#if PYLADA_PYTHON_MODULE != 1
    }
  }
  namespace 
  {
#endif

# include "quantity.h"

#if PYLADA_PYTHON_MODULE != 1
      }
    } // python
  } // Pylada
#endif

#ifdef PYLADA_INLINE
# undef PYLADA_INLINE
#endif
#ifdef PYLADA_END
# undef PYLADA_END
#endif
#if PYLADA_PYTHON_MODULE == 100
#  undef PYLADA_PYTHON_MODULE
#endif
// get ready for second inclusion
#ifdef PYLADA_PYTHON_MODULE 
# if PYLADA_PYTHON_MODULE == 0
#   undef PYLADA_PYTHON_MODULE 
#   define PYLADA_PYTHON_MODULE 1
# elif PYLADA_PYTHON_MODULE == 1
#   undef PYLADA_PYTHON_MODULE 
#   define PYLADA_PYTHON_MODULE 0
# endif
#endif 

#endif
