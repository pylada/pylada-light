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
namespace Pylada
{
  namespace math
  {
    //! \f$\pi\f$
    const types::t_real pi = 3.1415926535897932384626433832795028841971693993751058209749445920;
    
    //! 3d-vector of reals. 
    typedef Eigen::Matrix<types::t_real, 3, 1> rVector3d;
    //! 3d-vector of integers. 
    typedef Eigen::Matrix<types::t_int, 3, 1> iVector3d;
    //! 6d-vector of reals. 
    typedef Eigen::Matrix<types::t_real, 4, 1> rVector4d;
    //! 6d-vector of reals. 
    typedef Eigen::Matrix<types::t_real, 5, 1> rVector5d;
    //! 6d-vector of reals. 
    typedef Eigen::Matrix<types::t_real, 6, 1> rVector6d;
    
    //! 3d-matrix of reals. 
    typedef Eigen::Matrix<types::t_real, 3, 3> rMatrix3d;
    //! 3d-vector of integers. 
    typedef Eigen::Matrix<types::t_int, 3, 3> iMatrix3d;
    
    //! \typedef type of the angle axis object to initialize roations.
    typedef Eigen::AngleAxis<types::t_real> AngleAxis;
    //! \typedef type of the translation objects.
    typedef Eigen::Translation<types::t_real, 3> Translation;
#   ifndef PYLADA_WITH_EIGEN3 
      //! \typedef type of the affine transformations.
      typedef Eigen::Transform<types::t_real, 3> Affine3d;
#   else
      //! \typedef type of the affine transformations.
      typedef Eigen::Transform<types::t_real, 3, Eigen::Isometry> Affine3d;
#   endif
  } // namespace math
} // namespace Pylada


namespace Eigen
{
  //! Real type.
  typedef Pylada::types::t_real t_real;
  //! Integer type.
  typedef Pylada::types::t_int t_int;

  //! Cross product of real vectors.
  inline Matrix<t_real, 3, 1> operator^(Matrix<t_real, 3, 1> const &_a, Matrix<t_real, 3, 1> const &_b)
    { return _a.cross(_b); }
  //! Cross product of integer vectors.
  inline Matrix<t_int, 3, 1> operator^(Matrix<t_int, 3, 1> const &_a, Matrix<t_int, 3, 1> const &_b)
    { return _a.cross(_b); }

  //! \brief Inverse operation of real matrix.
  //! \note Probably slower than using eigen because of return type.
  inline Matrix<t_real, 3, 3> operator!(Matrix<t_real, 3, 3> const &_mat)
    { return _mat.inverse(); }

# ifndef PYLADA_WITH_EIGEN3 
    //! Transpose operation of real matrix.
    inline Transpose< Matrix<t_real, 3, 3> > operator~(Matrix<t_real, 3, 3> const &_mat)
      { return _mat.transpose(); }
    //! Transpose operation of integer matrix.
    inline Transpose< Matrix<t_int, 3, 3> > operator~(Matrix<t_int, 3, 3> const &_mat)
      { return _mat.transpose(); }
# else
    //! Transpose operation of matrix.
    template<class T_DERIVED>
      inline typename MatrixBase<T_DERIVED>::ConstTransposeReturnType
        operator~(MatrixBase<T_DERIVED> const &_mat) { return _mat.transpose(); }
# endif
}
#endif
