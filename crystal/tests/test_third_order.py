###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to
#  submit large numbers of jobs on supercomputers. It provides a python interface to physical input,
#  such as crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential
#  programs. It is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU
#  General Public License as published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################
from pytest import mark

def random_matrix(n=10):
    """ Yields random invertible 3x3 matrices """
    from numpy.random import random
    from numpy.linalg import det
    from numpy import abs
    for i in range(n):
        matrix = 10 * (random((3, 3)) - 0.5)
        while abs(det(matrix)) < 1e-4:
            matrix = 10 * (random((3, 3)) - 0.5)
        yield matrix

@mark.parametrize('cell', random_matrix(10))
def test_third_order_regression(cell):
    from numpy import abs
    from pylada.crystal.defects import third_order as pyto
    from pylada.crystal.defects.cutilities import third_order as cto

    assert abs(pyto(cell, 10) - cto(cell, 10)) < 1e-8
