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
from pytest import mark


def get_some_lattice(u):
    from pylada.crystal import Structure

    x, y = u, 0.25 - u
    return Structure([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])  \
        .add_atom(5.000000e-01, 5.000000e-01, 5.000000e-01, "A") \
        .add_atom(5.000000e-01, 2.500000e-01, 2.500000e-01, "A") \
        .add_atom(2.500000e-01, 5.000000e-01, 2.500000e-01, "A") \
        .add_atom(2.500000e-01, 2.500000e-01, 5.000000e-01, "A") \
        .add_atom(8.750000e-01, 8.750000e-01, 8.750000e-01, "B") \
        .add_atom(1.250000e-01, 1.250000e-01, 1.250000e-01, "B") \
        .add_atom(     x,     x,     x, "X") \
        .add_atom(     x,     y,     y, "X") \
        .add_atom(     y,     x,     y, "X") \
        .add_atom(     y,     y,     x, "X") \
        .add_atom(    -x,    -x,    -x, "X") \
        .add_atom(    -x,    -y,    -y, "X") \
        .add_atom(    -y,    -x,    -y, "X") \
        .add_atom(-y,    -y,    -x, "X")


def get_a_supercell(u):
    from random import randint
    from numpy import array
    from numpy.linalg import det
    from pylada.crystal import supercell
    lattice = get_some_lattice(u)
    while True:
        cell = [[randint(-2, 3) for j in range(3)] for k in range(3)]
        if det(cell) != 0:
            break
    structure = supercell(lattice, cell)
    copy = structure.copy()
    for atom in copy:
        del atom.site
        return structure, copy, lattice


@mark.parametrize('u', (0.25, 0.36))
def test_map_sites_ideal_structure(u):
    from pylada.crystal import map_sites

    structure0, structure1, lattice = get_a_supercell(u)
    assert map_sites(lattice, structure1)
    for a, b in zip(structure0, structure1):
        assert a.site == b.site


@mark.parametrize('u', (0.25, 0.36))
def test_map_sites_perturbed_structure(u):
    from numpy.random import random
    from pylada.crystal import map_sites

    structure0, structure1, lattice = get_a_supercell(u)
    for atom in structure1:
        atom.pos += random(3) * 1e-3

    assert map_sites(lattice, structure1, tolerance=1e-2)
    for a, b in zip(structure0, structure1):
        assert a.site == b.site
