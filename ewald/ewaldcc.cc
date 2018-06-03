/******************************
   This file is part of PyLaDa.

   Copyright (C) 2013 National Renewable Energy Lab

   PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to
submit
   large numbers of jobs on supercomputers. It provides a python interface to physical input, such
as
   crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs.
It
   is able to organise and launch computational jobs on PBS and SLURM.

   PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU
General
   Public License as published by the Free Software Foundation, either version 3 of the License, or
(at
   your option) any later version.

   PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.

   You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
   <http://www.gnu.org/licenses/>.
******************************/

namespace {
extern "C" void ewaldf(const int *const,    // verbosity
                       double *const,       // Energy
                       double *const,       // forces (reduced)
                       double *const,       // forces (cartesian)
                       const double *const, // stress
                       const int *const,    // number of atoms
                       const double *const, // reduced atomic coordinates.
                       const double *const, // atomic charges
                       const double *const, // real space cutoff
                       const double *const, // cell vectors
                       const int *const,    // dimension of arrays.
                       int *const           // ERROR
);
}
namespace pylada {
inline int ewaldc(const int verbosity, double &energy, double *const reduced_forces,
           double *const cartesian_forces, const double *const stress, const int natoms,
           const double *const reduced_atomic_coords, const double *const atomic_charges,
           const double real_space_cutoff, const double *const cell_vectors) {
  int error;
  ewaldf(&verbosity, &energy, reduced_forces, cartesian_forces, stress, &natoms,
         reduced_atomic_coords, atomic_charges, &real_space_cutoff, cell_vectors, &natoms, &error);
  return error;
}
} // namespace pylada
