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
""" Checks that point group of cell is determined correctly. """

from pytest import mark
from pylada.crystal import Structure


def is_integer(o, tolerance=1e-8):
    from numpy import allclose, floor
    return allclose(o, floor(o + 0.1), tolerance)

parameters = [
    ([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], 48),
    ([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]], 48),
    ([[-0.6, 0.5, 0.5], [0.6, -0.5, 0.5], [0.6, 0.5, -0.5]], 4),
    ([[-0.7, 0.7, 0.7], [0.6, -0.5, 0.5], [0.6, 0.5, -0.5]], 8),
    ([[-0.765, 0.7, 0.7], [0.665, -0.5, 0.5], [0.6, 0.5, -0.5]], 2)
]
parameters += [(Structure(cell).add_atom(0, 0, 0, 'Si'), n) for cell, n in parameters]


def test_gvectors_fcc():
    from numpy import array, allclose, sqrt
    from numpy.linalg import norm
    from itertools import chain
    from pylada.crystal import space_group

    cell = array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    result = space_group.__gvectors(cell, 1e-12)
    assert len(result) == 3
    assert len(result[0]) > 0
    assert len(result[0]) == len(result[1])
    assert len(result[0]) == len(result[2])
    # check directions are equivalent and each vector is unique
    for vectors in result:
        for vector in vectors:
            assert sum([allclose(u, vector) for u in result[0]]) == 1

    for vector in result[0]:
        assert abs(norm(vector) - 0.5 * sqrt(2.)) < 1e-8


def test_gvectors_non_fcc():
    from numpy import array, allclose, sqrt
    from numpy.linalg import norm
    from itertools import chain
    from pylada.crystal import space_group

    cell = array([[0.6, 0.5, 0.5], [0.6, -0.5, 0.5], [0.6, 0.5, -0.5]])
    result = space_group.__gvectors(cell, 1e-12)
    assert len(result) == 3
    assert len(result[0]) > 0
    assert len(result[1]) == len(result[2])
    # check directions 1 and 2 are equivalent
    for vector in result[1]:
        assert sum([allclose(u, vector) for u in result[2]]) == 1
    # check unicity
    for vector in result[0]:
        assert sum([allclose(u, vector) for u in result[0]]) == 1
    for vector in result[1]:
        assert sum([allclose(u, vector) for u in result[1]]) == 1

    for vector in result[0]:
        assert abs(norm(vector) - 0.6 * sqrt(3.)) < 1e-8
    for vector in result[1]:
        assert abs(norm(vector) - 0.5 * sqrt(3.)) < 1e-8


@mark.parametrize("cell,numops", parameters)
def test_cellinvariants(cell, numops):
    """ Test number of symmetry operations. """
    from numpy import all, abs, dot, array
    from numpy.linalg import inv, det
    from pylada.crystal.space_group import cell_invariants

    ops = cell_invariants(cell)
    if isinstance(cell, Structure):
        cell = cell.cell.copy()
    assert len(ops) == numops
    for op in ops:
        assert op.shape == (3, 3)
        transformation = dot(dot(inv(cell), op), cell)
        assert is_integer(transformation)
        assert abs(abs(det(transformation)) - 1e0) < 1e-8
    if numops != 48:
        allops = cell_invariants(array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]))
        failed = 0
        for op in allops:
            transformation = dot(dot(inv(cell), op[:3]), cell)
            if not (is_integer(transformation) and abs(abs(det(transformation)) - 1e0) < 1e-8):
                failed += 1
        assert failed == 48 - numops
