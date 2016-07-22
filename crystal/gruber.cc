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

#include "crystal/cutilities.h"
#include "crystal/fuzzy.h"
#include "crystal/types.h"
#include <Eigen/Dense>
#include <cstdlib>
#include <exception>
#include <utility>

namespace pylada {
namespace details {
bool no_opt_change_test(types::t_real _new, types::t_real _last);
}

types::t_int const max_no_change = 2;

//! Implementation of the Gruber algorithm.
//! Class implementations simplify the number of arguments to pass.
struct Gruber {
  types::rMatrix3d cell, rinv;
  types::t_real tol;
  types::t_int itermax;
  types::t_int nochange, iterations;
  types::t_real a, b, c, d, e, f;
  types::t_real last_a, last_b, last_c;
  Gruber(types::t_int _itermax, types::t_real _tol)
      : cell(types::rMatrix3d::Zero()), tol(_tol), itermax(_itermax), nochange(0) {}

  types::rMatrix3d operator()(types::rMatrix3d const &_in) {
    if(std::abs(_in.determinant()) < 1e-6)
      throw std::runtime_error("Singular matrix");
    iterations = 0;
    cell = _in;
    types::rMatrix3d metric = cell.transpose() * cell;
    a = metric(0, 0);
    b = metric(1, 1);
    c = metric(2, 2);
    d = 2e0 * metric(1, 2);
    e = 2e0 * metric(0, 2);
    f = 2e0 * metric(0, 1);
    rinv = types::rMatrix3d::Identity();
    nochange = 0;
    last_a = -a;
    last_b = -b;
    last_c = -c;
    while(step())
      ;
    return cell * rinv;
  };

  bool def_gt_0() {
    int z, p;
    def_test(z, p);
    return p == 3 or (z == 0 and p == 1);
  }

  void def_test(int &_zero, int &_positive) {
    _zero = 0;
    _positive = 0;
    if(lt(0e0, d, tol))
      _positive += 1;
    else if(eq(d, 0e0, tol))
      _zero += 1;
    if(lt(0e0, e, tol))
      _positive += 1;
    else if(eq(e, 0e0, tol))
      _zero += 1;
    if(lt(0e0, f, tol))
      _positive += 1;
    else if(eq(f, 0e0, tol))
      _zero += 1;
  }

  bool step() {
    if(gt(a, b, tol) or (eq(a, b, tol) and gt(std::abs(d), std::abs(e), tol)))
      n1_action();
    if(gt(b, c, tol) or (eq(b, c, tol) and gt(std::abs(e), std::abs(f), tol))) {
      n2_action();
      return true;
    }
    if(def_gt_0())
      n3_action();
    else {
      n4_action();
      if(not is_changing())
        return false;
    }
    if(gt(std::abs(d), b, tol) or (eq(d, b, tol) and lt(e + e, f, tol)) or
       (eq(d, -b, tol) and lt(f, 0e0, tol))) {
      n5_action();
      return true;
    }
    if(gt(std::abs(e), a, tol) or (eq(e, a, tol) and lt(d + d, f, tol)) or
       (eq(e, -a, tol) and lt(f, 0e0, tol))) {
      n6_action();
      return true;
    }
    if(gt(std::abs(f), a, tol) or (eq(f, a, tol) and lt(d + d, e, tol)) or
       (eq(f, -a, tol) and lt(e, 0e0, tol))) {
      n7_action();
      return true;
    }
    if(lt(d + e + f + a + b, 0e0, tol) or (eq(f, a, tol) and lt(d + d, e, tol)) or
       (eq(d + e + f + a + b, 0e0, tol) and gt(a + a + e + e + f, 0e0, tol))) {
      n8_action();
      return true;
    }
    return false;
  }

  bool is_changing() {
    if(details::no_opt_change_test(a, last_a) || details::no_opt_change_test(b, last_b) ||
       details::no_opt_change_test(c, last_c))
      nochange = 0;
    else
      ++nochange;
    last_a = a;
    last_b = b;
    last_c = c;
    return nochange < max_no_change;
  }

  void cb_update(types::t_real a00, types::t_real a01, types::t_real a02, types::t_real a10,
                 types::t_real a11, types::t_real a12, types::t_real a20, types::t_real a21,
                 types::t_real a22) {
    if(itermax != 0 and iterations >= itermax)
      throw std::runtime_error("Reached maximum iterations");
    types::rMatrix3d update;
    update << a00, a01, a02, a10, a11, a12, a20, a21, a22;
    rinv *= update;
    ++iterations;
  }

  void n1_action() {
    cb_update(0, -1, 0, -1, 0, 0, 0, 0, -1);
    std::swap(a, b);
    std::swap(d, e);
  }

  void n2_action() {
    cb_update(-1, 0, 0, 0, 0, -1, 0, -1, 0);
    std::swap(b, c);
    std::swap(e, f);
  }

  void n3_action() {
    types::t_int i(1), j(1), k(1);
    if(lt(d, 0e0, tol))
      i = -1;
    if(lt(e, 0e0, tol))
      j = -1;
    if(lt(f, 0e0, tol))
      k = -1;
    cb_update(i, 0, 0, 0, j, 0, 0, 0, k);
    d = std::abs(d);
    e = std::abs(e);
    f = std::abs(f);
  }
  void n4_action() {
    types::iVector3d vec(1, 1, 1);
    size_t z(6);
    if(gt(d, 0e0, tol))
      vec(0) = -1;
    else if(eq(d, 0e0, tol))
      z = 0;
    if(gt(e, 0e0, tol))
      vec(1) = -1;
    else if(eq(e, 0e0, tol))
      z = 1;
    if(gt(f, 0e0, tol))
      vec(2) = -1;
    else if(eq(f, 0e0, tol))
      z = 2;
    if(vec(0) * vec(1) * vec(2) < 0) {
      if(z == 6)
        throw std::runtime_error("Internal error");
      vec(z) = -1;
    }
    cb_update(vec(0), 0, 0, 0, vec(1), 0, 0, 0, vec(2));
    d = -std::abs(d);
    e = -std::abs(e);
    f = -std::abs(f);
  }

  void n5_action() {
    if(d > 0) {
      cb_update(1, 0, 0, 0, 1, -1, 0, 0, 1);
      c += b - d;
      d -= b + b;
      e -= f;
    } else {
      cb_update(1, 0, 0, 0, 1, 1, 0, 0, 1);
      c += b + d;
      d += b + b;
      e += f;
    }
  }
  void n6_action() {
    if(e > 0) {
      cb_update(1, 0, -1, 0, 1, 0, 0, 0, 1);
      c += a - e;
      d -= f;
      e -= a + a;
    } else {
      cb_update(1, 0, 1, 0, 1, 0, 0, 0, 1);
      c += a + e;
      d += f;
      e += a + a;
    }
  }
  void n7_action() {
    if(f > 0) {
      cb_update(1, -1, 0, 0, 1, 0, 0, 0, 1);
      b += a - f;
      d -= e;
      f -= a + a;
    } else {
      cb_update(1, 1, 0, 0, 1, 0, 0, 0, 1);
      b += a + f;
      d += e;
      f += a + a;
    }
  }
  void n8_action() {
    cb_update(1, 0, 1, 0, 1, 1, 0, 0, 1);
    c += a + b + d + e + f;
    d += b + b + f;
    e += a + a + f;
  }
};

//! Computes the Niggli cell of a lattice.
void gruber(types::t_real *_out, types::t_real const *_in, size_t itermax, types::t_real _tol) {
  Eigen::Map<types::rMatrix3d const> in(_in, 3, 3);
  Eigen::Map<types::rMatrix3d>(_out, 3, 3) = Gruber(itermax, _tol)(in);
}
}
