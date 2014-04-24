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

#if PYLADA_PYTHON_MODULE != 1
  //! Checks whether this a quantity object.
  PYLADA_INLINE bool check_quantity(PyObject *_in)
    PYLADA_END( return ((bool(*)(PyObject*))api_capsule[PYLADA_SLOT(python)])(_in); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)check_quantity;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)


#if PYLADA_PYTHON_MODULE != 1
  //! Returns a quantity object from a number and a unit in string.
  PYLADA_INLINE PyObject* fromC_quantity(types::t_real const& _double, std::string const &_units)
    PYLADA_END( return ( (PyObject*(*)(types::t_real const&, std::string const&) )
                       api_capsule[PYLADA_SLOT(python)] )
                     (_double, _units); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)
     ( ( PyObject*(*)(types::t_real const &, std::string const&) )  
       fromC_quantity );
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)

#if PYLADA_PYTHON_MODULE != 1
  //! Returns a quantity object from a number and a quantity.
  PYLADA_INLINE PyObject* fromC_quantity(types::t_real const &_double, PyObject *_unittemplate)
    PYLADA_END( return ( (PyObject*(*)(types::t_real const &, PyObject*) )  
                       api_capsule[PYLADA_SLOT(python)] )
                     (_double, _unittemplate); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)
     ( ( PyObject*(*)(types::t_real const &, PyObject*) )  
       fromC_quantity );
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)

#if PYLADA_PYTHON_MODULE != 1
  //! \brief Returns a quantity object from a number and a quantity object.
  //! \details if _number has itself a unit, then it should be convertible to
  //! _units.  However, in that case, _number is returned as is
  //! (Py_INCREF'ed).
  PYLADA_INLINE PyObject* fromPy_quantity(PyObject *_number, PyObject *_units)
    PYLADA_END( return ( (PyObject*(*)(PyObject*, PyObject*) )  
                       api_capsule[PYLADA_SLOT(python)] )
                     (_number, _units); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *) fromPy_quantity;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)

#if PYLADA_PYTHON_MODULE != 1
  //! \brief Returns as a number in specified units.
  //! \details If the number is 0, then one should check whether exception
  //! was thrown.
  PYLADA_INLINE types::t_real get_quantity(PyObject *_number, std::string const &_units)
    PYLADA_END( return ( (types::t_real(*)(PyObject*, std::string const&) )  
                       api_capsule[PYLADA_SLOT(python)] )
                     (_number, _units); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)
     ( ( types::t_real(*)(PyObject*, std::string const&) )  
       get_quantity );
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)

#if PYLADA_PYTHON_MODULE != 1
  //! \brief Returns as a number in specified units.
  //! \details If the number is 0, then one should check whether exception
  //! was thrown.
  PYLADA_INLINE types::t_real get_quantity(PyObject *_number, PyObject *_units)
    PYLADA_END( return ( (types::t_real(*)(PyObject*, PyObject*) )  
                       api_capsule[PYLADA_SLOT(python)] )
                     (_number, _units); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)
     ( ( types::t_real(*)(PyObject*, PyObject*) )  
       get_quantity );
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)

#if PYLADA_PYTHON_MODULE != 1
  //! \brief Returns as float without conversion. 
  PYLADA_INLINE types::t_real get_quantity(PyObject *_in)
    PYLADA_END( return ( (types::t_real(*)(PyObject*) )  
                       api_capsule[PYLADA_SLOT(python)] )
                     (_in); )
#else
  api_capsule[PYLADA_SLOT(python)] = (void *)
     ( ( types::t_real(*)(PyObject*) ) get_quantity );
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(python))
#include PYLADA_ASSIGN_SLOT(python)
