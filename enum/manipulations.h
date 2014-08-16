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

#ifndef PYLADA_ENUM_MANIPULATIONS_H
#define PYLADA_ENUM_MANIPULATIONS_H
#include "PyladaConfig.h"

#include <Python.h>
#include <structmember.h>
#define PY_ARRAY_UNIQUE_SYMBOL pylada_enum_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "ndimiterator.h"

//! \def PyManipulations_Check(object)
//!      Returns true if an object is a struture or subtype.
#define PyManipulations_Check(object) PyObject_TypeCheck(object, Pylada::crystal::structure_type())
//! \def PyManipulations_CheckExact(object)
//!      Returns true if an object is a structure.
#define PyManipulations_CheckExact(object) object->ob_type == Pylada::crystal::structure_type()
      

namespace Pylada
{
  namespace enumeration
  {
    //! Type used internally by the counter.
    extern "C" 
    {
      //! \brief Describes basic manipulations iterator. 
      struct Manipulations
      {
        PyObject_HEAD 
        //! Array object
        PyArrayObject *arrayout;
        PyObject *arrayin;
        //! Inner counter;
        std::vector<t_ndim> counter;
        //! Inner counter;
        std::vector<t_ndim> substituted;
        //! Inner counter;
        std::vector< std::vector<t_ndim> > substitutions;
        //! iterator over substitutions
        std::vector< std::vector<t_ndim> >:: iterator i_first;
        //! iterator over substitutions
        std::vector< std::vector<t_ndim> >:: iterator i_end;
        //! Whether this is the first iteration.
        bool is_first;
      };
      //! Creates a new structure.
      Manipulations* PyManipulations_New();
      //! Creates a new structure with a given type.
      Manipulations* PyManipulations_NewWithArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs);
      //! Creates a new structure with a given type, also calling initialization.
      Manipulations* PyManipulations_NewFromArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs);
      // Returns pointer to structure type.
      PyTypeObject* manipulations_type();
    }
  }
}
#endif
