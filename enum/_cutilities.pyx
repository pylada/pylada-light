###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
#  crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################

__docformat__ = "restructuredtext en"
from libcpp cimport bool

cdef extern from "cmath" namespace "std":
    float floor(float)
    double floor(double)

cpdef bool _is_integer(double[:, ::1] vector, double tolerance=1e-8):
    cdef:
        int ni = vector.shape[0]
        int nj = vector.shape[1]
        int i, j
    for i in range(ni):
        for j in range(nj):
            if abs(floor(vector[i, j] + 1e-3) - vector[i, j]) > tolerance:
                return False
    return True
