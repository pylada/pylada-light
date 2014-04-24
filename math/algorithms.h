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

#if PYLADA_MATH_MODULE != 1
  //! \brief Computes the Gruber cell of a lattice.
  //! \details Shamelessly adapted from the computational rystallography
  //!          tool box.
  PYLADA_INLINE rMatrix3d gruber( rMatrix3d const &_in, 
                                size_t itermax = 0,
                                types::t_real _tol = types::tolerance )
    PYLADA_END( return ( (rMatrix3d(*)( rMatrix3d const &,
                                      size_t itermax, 
                                      types::t_real ))
                       api_capsule[PYLADA_SLOT(math)] )
                     (_in, itermax, _tol); )
#else
  api_capsule[PYLADA_SLOT(math)] = (void *)gruber;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(math))
#include PYLADA_ASSIGN_SLOT(math)

#if PYLADA_MATH_MODULE != 1
  //! \brief Computes the parameters of the metrical matrix associated with
  //!        the gruber cell.
  PYLADA_INLINE Eigen::Matrix<types::t_real, 6, 1>
    gruber_parameters( rMatrix3d const &_in,
                       size_t itermax = 0,
                       types::t_real _tol = types::tolerance)
    PYLADA_END( return ( (Eigen::Matrix<types::t_real, 6, 1>(*)( rMatrix3d const &,
                                      size_t itermax, 
                                      types::t_real ))
                       api_capsule[PYLADA_SLOT(math)] ) 
                     (_in, itermax, _tol); )
#else
  api_capsule[PYLADA_SLOT(math)] = (void *)gruber_parameters;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(math))
#include PYLADA_ASSIGN_SLOT(math)

#if PYLADA_MATH_MODULE != 1
  //! Computes smith normal form of a matrix \a  _S = \a _L \a _M \a _R.
  PYLADA_INLINE void smith_normal_form( iMatrix3d& _S, 
                                      iMatrix3d& _L, 
                                      const iMatrix3d& _M, 
                                      iMatrix3d& _R )
    PYLADA_END( ( (void(*)( iMatrix3d&, 
                          iMatrix3d&, 
                          const iMatrix3d&, 
                          iMatrix3d& ))
                  api_capsule[PYLADA_SLOT(math)] ) 
                (_S, _L, _M, _R); )
#else
  api_capsule[PYLADA_SLOT(math)] = (void *)smith_normal_form;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(math))
#include PYLADA_ASSIGN_SLOT(math)
