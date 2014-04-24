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
  //! \brief True if invariant by this transform. Ignores translation.
  inline bool is_invariant(Affine3d const&_a, rMatrix3d const &_m)
    { return is_integer(_m.inverse() * _a.linear() * _m); }
  //! \brief True if invariant by this transform. Ignores translation.
  inline bool is_invariant(Affine3d const&_a, rMatrix3d const &_m, types::t_real const &_tol) 
    { return is_integer(_m.inverse() * _a.linear() * _m, _tol); }
  //! True if invariant by this transform.
  //! \warning This is different from checking whether a lattice is
  //!          invariant, since we do not check for translation invariance.
  inline bool is_invariant(Affine3d const&_a, rVector3d const &_m) { return eq(_m, _a * _m); }
  //! True if invariant by this transform.
  //! \warning This is different from checking whether a lattice is
  //!          invariant, since we do not check for translation invariance.
  inline bool is_invariant(Affine3d const&_a, rVector3d const &_m, types::t_real const &_tol) 
    { return eq(_m, _a * _m, _tol); }
  //! Checks if this is a pure rotation. 
  inline bool is_rotation(Affine3d const &_a)
    { return is_null(_a.translation())
             and is_identity(_a.linear() * (~_a.linear()))
             and eq(_a.linear().determinant(), 1.); }
  //! Checks if this is a pure rotation. 
  inline bool is_rotation(Affine3d const &_a, types::t_real _tol) 
    { return is_null(_a.translation(), _tol)
             and is_identity(_a.linear() * (~_a.linear()), _tol)
             and eq(_a.linear().determinant(), 1., _tol); }
  //! Checks if this is an isometry.
  inline bool is_isometry(Affine3d const &_a)
    { return is_identity(_a.linear() * (~_a.linear()))
             and eq(_a.linear().determinant(), 1.)
             and eq(_a.matrix()(3,0), 0.)
             and eq(_a.matrix()(3,1), 0.)
             and eq(_a.matrix()(3,2), 0.)
             and eq(_a.matrix()(3,3), 1.); }
  //! Checks if this is an isometry.
  inline bool is_isometry(Affine3d const &_a, Affine3d::Scalar _tol)
    { return     is_identity(_a.linear() * (~_a.linear()), _tol)
             and eq(_a.linear().determinant(), 1., _tol)
             and eq(_a.matrix()(3,0), 0., _tol)
             and eq(_a.matrix()(3,1), 0., _tol)
             and eq(_a.matrix()(3,2), 0., _tol)
             and eq(_a.matrix()(3,3), 1., _tol); }
#endif
