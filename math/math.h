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

#ifndef PYLADA_MATH_MODULE_H
#define PYLADA_MATH_MODULE_H
#ifndef __cplusplus
# error Pylada requires a cpp compiler
#endif


#ifndef PYLADA_MATH_MODULE
#  define PYLADA_MATH_MODULE 100
#endif 

#if PYLADA_MATH_MODULE != 1
# include "PyladaConfig.h"
# include <Python.h>
# ifndef PYLADA_PYTHONTWOSIX
#   if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION < 7
#     define PYLADA_PYTHONTWOSIX
#   endif
# endif

# include <boost/utility/enable_if.hpp>
# include <boost/type_traits/is_integral.hpp>
# include <boost/type_traits/is_arithmetic.hpp>
# include <boost/type_traits/is_floating_point.hpp>
# include <boost/numeric/conversion/converter.hpp>

# include <boost/preprocessor/arithmetic/inc.hpp>
# include <python/ppslot.hpp>
# define BOOST_PP_VALUE 0
# include PYLADA_ASSIGN_SLOT(math)

# include <python/types.h>
# include <Eigen/Core>
# include <Eigen/LU>
# include <Eigen/Geometry>

# include <errors/exceptions.h>

# if PYLADA_MATH_MODULE == 100

    namespace Pylada
    {

      namespace math
      {
        /* This section is used in modules that use pylada.math's API */
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
            PyObject *module = PyImport_ImportModule("pylada.math");
            if(not module) return false;
#           ifdef PYLADA_PYTHONTWOSIX
              PyObject* c_api_object = PyObject_GetAttrString(module, "_C_API");
	      if (c_api_object == NULL) { Py_DECREF(module); return false; }
              if (PyCObject_Check(c_api_object))
                api_capsule = (void **)PyCObject_AsVoidPtr(c_api_object);
              Py_DECREF(c_api_object);
#           else
              api_capsule = (void **)PyCapsule_Import("pylada.math._C_API", 0);
#           endif
            Py_DECREF(module);
            return api_capsule != NULL;
          }
        }
      }
    }
# endif
#else
# define BOOST_PP_VALUE 0
# include PYLADA_ASSIGN_SLOT(math)
#endif

#if PYLADA_MATH_MODULE != 1
#  ifdef PYLADA_INLINE
#    error PYLADA_INLINE already defined
#  endif
#  if PYLADA_MATH_MODULE == 100
#    define PYLADA_INLINE inline
#  elif PYLADA_MATH_MODULE == 0
#    define PYLADA_INLINE
#  endif
#  ifdef PYLADA_END
#    error PYLADA_END already defined
#  elif PYLADA_MATH_MODULE == 0
#    define PYLADA_END(X) ;
#  elif PYLADA_MATH_MODULE == 100
#    define PYLADA_END(X) { X }
#  endif
#endif

// Some stuff needs to be added to eigen, so do outside of namespace.
// Also, it is actually needed by python. Not very cool. 
# include "eigen.h"
// different namespace
# include "exceptions.h"

#if PYLADA_MATH_MODULE != 1
  namespace Pylada
  {
    namespace math 
    {
      namespace 
      {
#endif

#       include "fuzzy.h"
#       include "misc.h"
#       include "symmetry_operator.h"
#       include "algorithms.h"
        
#if PYLADA_MATH_MODULE != 1
      }
    }
  }
#endif


// get ready for second inclusion
#ifdef PYLADA_MATH_MODULE 
# if PYLADA_MATH_MODULE == 0
#   undef PYLADA_MATH_MODULE 
#   define PYLADA_MATH_MODULE 1
# elif PYLADA_MATH_MODULE == 1
#   undef PYLADA_MATH_MODULE 
#   define PYLADA_MATH_MODULE 0
# endif
#endif 
#ifdef PYLADA_INLINE
# undef PYLADA_INLINE
#endif
#ifdef PYLADA_END
# undef PYLADA_END
#endif

#if PYLADA_MATH_MODULE == 100
# undef PYLADA_MATH_MODULE
#endif

#endif 

