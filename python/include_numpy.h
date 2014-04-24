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

#ifndef PYLADA_INCLUDE_NUMPY

#if 0
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

// PG
#if 1
// Makes sure that NPY_ARRAY style stuff exists, and ENABLEFLAGS
#if NUMPY_VERSION_MINOR >= 7
#  define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>
#ifndef LADA_NPY_NEWDEFS
#  define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS 
#  define NPY_ARRAY_WRITEABLE    NPY_WRITEABLE
#endif
#ifndef LADA_NPY_HAS_ENABLEFLAGS
#  define PyArray_ENABLEFLAGS(ARRAY, FLAGS) ARRAY->flags |= FLAGS
#  define PyArray_CLEARFLAGS(ARRAY, FLAGS)  ARRAY->flags &= (!FLAGS)
#  define PyArray_SetBaseObject(ARRAY, BASE) ARRAY->base = BASE
#endif
#endif



#endif
