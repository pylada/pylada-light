/******************************
   This file is part of PyLaDa.

   Copyright (C) 2013 National Renewable Energy Lab

   PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to
   submit large numbers of jobs on supercomputers. It provides a python interface to physical input,
   such as crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential
   programs. It is able to organise and launch computational jobs on PBS and SLURM.

   PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU
   General Public License as published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.

   PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.

   You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
   <http://www.gnu.org/licenses/>.
******************************/

#include "pylada/crystal/defects/third_order.h"
#include "pylada/crystal/types.h"
#include <Eigen/Dense>
#include <cstdlib>

namespace pylada {
template <class T> types::t_real third_order(Eigen::MatrixBase<T> const &_matrix, types::t_int _n) {
  typedef types::rVector3d rVector3d;
  typedef types::t_real t_real;
  t_real result = 0e0;
  t_real const ninv = 1e0 / t_real(_n);

  for(types::t_int i(0); i < _n; ++i) {
    for(types::t_int j(0); j < _n; ++j) {
      for(types::t_int k(0); k < _n; ++k) {
        t_real min_dist =
            (_matrix * rVector3d(i * ninv - 0.5, j * ninv - 0.5, k * ninv - 0.5)).squaredNorm();
        for(types::t_int l(-1); l < 2; ++l)
          for(types::t_int m(-1); m < 2; ++m)
            for(types::t_int n(-1); n < 2; ++n) {
              auto const v = rVector3d(i * ninv + l - 0.5, j * ninv + m - 0.5, k * ninv + n - 0.5);
              t_real const q = (_matrix * v).squaredNorm();
              if(q < min_dist)
                min_dist = q;
            }
        result += min_dist;
      }
    }
  }
  return result / (std::abs(_matrix.determinant()) * t_real(_n * _n * _n));
}

types::t_real third_order(types::t_real const *_matrix, types::t_int _n) {
  auto const matrix = types::rMatrix3d::Map(_matrix).transpose();
  return third_order(matrix, _n);
}
} // namespace pylada
