/******************************
   This file is part of PyLaDa.

   Copyright (C) 2013 National Renewable Energy Lab

   PyLaDa is a high throughput computational platform for Physics. It aims to
   make it easier to submit large numbers of jobs on supercomputers. It
   provides a python interface to physical input, such as crystal structures,
   as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs.
   It is able to organise and launch computational jobs on PBS and SLURM.

   PyLaDa is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
   details.

   You should have received a copy of the GNU General Public License along with
   PyLaDa.  If not, see <http://www.gnu.org/licenses/>.
******************************/

#ifndef PYLADA_TYPES_H_
#define PYLADA_TYPES_H_

#include <Eigen/Core>
#include <complex>
#include <limits>

namespace pylada {
//! Names a few simple variable types and globals for portability purposes
namespace types {
//! the unsigned integer type
typedef unsigned t_unsigned;
//! the signed integer type
typedef int t_int;
//! the real value type
typedef double t_real;
//! the character type, unused
typedef char t_char;
//! a complex real type
typedef std::complex<types::t_real> t_complex;
//! \brief all-purpose global tolerance.
//! \warning Setting this to a very small value may lead to bizarre errors.
//!          For instance, some of the tests are known to fail occasionally
//!          on a 64bit linux when tolerance = 1e-12.
const t_real tolerance = 1.e-8;
//! roundoff term for numerical noise crap.
types::t_real const roundoff(5e3 * std::numeric_limits<types::t_real>::epsilon());

//! 3d-vector of reals.
typedef Eigen::Matrix<t_real, 3, 1> rVector3d;
//! 3d-vector of integers.
typedef Eigen::Matrix<t_int, 3, 1> iVector3d;
//! 6d-vector of reals.
typedef Eigen::Matrix<t_real, 4, 1> rVector4d;
//! 6d-vector of reals.
typedef Eigen::Matrix<t_real, 5, 1> rVector5d;
//! 6d-vector of reals.
typedef Eigen::Matrix<t_real, 6, 1> rVector6d;

//! 3d-matrix of reals.
typedef Eigen::Matrix<t_real, 3, 3> rMatrix3d;
//! 3d-vector of integers.
typedef Eigen::Matrix<t_int, 3, 3> iMatrix3d;
}
} // namespace Pylada
#endif
