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

#ifndef PYLADA_ENUM_NDIMITERATOR_H
#define PYLADA_ENUM_NDIMITERATOR_H
#include "PyladaConfig.h"

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL pylada_enum_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <vector>

//! \def PyNDimIterator_Check(object)
//!      Returns true if an object is a struture or subtype.
#define PyNDimIterator_Check(object) PyObject_TypeCheck(object, Pylada::crystal::structure_type())
//! \def PyNDimIterator_CheckExact(object)
//!      Returns true if an object is a structure.
#define PyNDimIterator_CheckExact(object) object->ob_type == Pylada::crystal::structure_type()
      

namespace Pylada
{
  namespace enumeration
  {
    //! Type used internally by the counter.
    typedef npy_short t_ndim;
    extern "C" 
    {
      //! \brief Describes basic structure type. 
      //! \details Instances of this object are exactly those that are seen
      //!          within the python interface. C++, however, defines a
      //!          secondary NDimIterator object which wrapps around a python
      //!          refence to instances of this object. NDimIterator provides some
      //!          syntactic sugar for handling in c++. 
      struct NDimIterator
      {
        PyObject_HEAD 
        //! Holds beginning and end of range.
        std::vector<t_ndim> ends;
        //! Read-only numpy array referencing inner counter.
        PyArrayObject *yielded;
        //! Inner counter;
        std::vector<t_ndim> counter;
      };
      //! Creates a new structure.
      NDimIterator* PyNDimIterator_New();
      //! Creates a new structure with a given type.
      NDimIterator* PyNDimIterator_NewWithArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs);
      //! Creates a new structure with a given type, also calling initialization.
      NDimIterator* PyNDimIterator_NewFromArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs);
      // Returns pointer to structure type.
      PyTypeObject* ndimiterator_type();
    }
  }
}
#endif
