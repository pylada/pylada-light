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

#ifdef PYLADA_TYPEDEF
# error PYLADA_TYPEDEF already defined
#endif
#define PYLADA_TYPEDEF                                                          \
    (Pylada::math::rVector3d(*)( Pylada::math::rVector3d const&,                  \
                               Pylada::math::rMatrix3d const&,                  \
                               Pylada::math::rMatrix3d const& ))

#if PYLADA_CRYSTAL_MODULE != 1
  //! Refolds a periodic vector into the unit cell.
  PYLADA_INLINE math::rVector3d into_cell( math::rVector3d const &_vec, 
                                         math::rMatrix3d const &_cell, 
                                         math::rMatrix3d const &_inv)
    PYLADA_END(return (*PYLADA_TYPEDEF
                      api_capsule[PYLADA_SLOT(crystal)])(_vec, _cell, _inv);) 
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)(PYLADA_TYPEDEF into_cell);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)
  
#if PYLADA_CRYSTAL_MODULE != 1
  //! Refolds a periodic vector into the voronoi cell (eg first BZ or WignerSeitz).
  PYLADA_INLINE math::rVector3d into_voronoi( math::rVector3d const &_vec, 
                                            math::rMatrix3d const &_cell, 
                                            math::rMatrix3d const &_inv)
    PYLADA_END(return (*PYLADA_TYPEDEF
                      api_capsule[PYLADA_SLOT(crystal)])(_vec, _cell, _inv);) 
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)(PYLADA_TYPEDEF into_voronoi);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Refolds a periodic vector into a cell centered around zero (in
  //!        fractional coordinates).
  //! \details Since the vector is refolded in fractional coordinates, it may
  //!          or may not be the vector with smallest norm. Use math::rVector3d
  //!          into_voronoi() to get the equivalent vector with smallest norm.
  PYLADA_INLINE math::rVector3d zero_centered( math::rVector3d const &_vec, 
                                             math::rMatrix3d const &_cell, 
                                             math::rMatrix3d const &_inv)
    PYLADA_END(return (*PYLADA_TYPEDEF
                      api_capsule[PYLADA_SLOT(crystal)])(_vec, _cell, _inv);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)(PYLADA_TYPEDEF zero_centered);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#undef PYLADA_TYPEDEF

#if PYLADA_CRYSTAL_MODULE != 1
  //! Refolds a periodic vector into the unit cell.           
  inline math::rVector3d into_cell( math::rVector3d const &_vec, 
                                    math::rMatrix3d const &_cell )
    { return into_cell(_vec, _cell, _cell.inverse()); }
  //! \brief Refolds a periodic vector into the voronoi cell (eg first BZ or WignerSeitz).
  //! \details May fail if the matrix is a weird parameterization of the
  //!          lattice. It is best to use a grubber(...) cell . 
  inline math::rVector3d into_voronoi( math::rVector3d const &_vec, 
                                       math::rMatrix3d const &_cell )
    { return into_voronoi(_vec, _cell, _cell.inverse()); }

  //! \brief Refolds a periodic vector into a cell centered around zero (in
  //!        fractional coordinates).
  //! \details Since the vector is refolded in fractional coordinates, it may
  //!          or may not be the vector with smallest norm. Use math::rVector3d
  //!          into_voronoi() to get the equivalent vector with smallest norm.
  inline math::rVector3d zero_centered( math::rVector3d const &_vec, 
                                        math::rMatrix3d const &_cell )
    { return zero_centered(_vec, _cell, _cell.inverse()); }
#endif
