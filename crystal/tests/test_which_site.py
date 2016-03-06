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


def get_a_b5_lattice(u):
    from pylada.crystal import Structure
    x, y = u, 0.25 - u
    return Structure([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) \
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


@mark.parametrize('u', (0.25, 0.36))
def test_position_to_atomic_site_correspondence(u):
    from numpy import dot
    from numpy.random import randint
    from pylada.crystal import which_site

    lattice = get_a_b5_lattice(u)
    assert which_site(lattice[6].pos + [0.5, -0.5, 2], lattice) == 6
    for i, atom in enumerate(lattice):
        assert which_site(atom.pos, lattice) == i
        for j in xrange(10):
            newpos = dot(lattice.cell, randint(10, size=(3,)) - 5)
            assert which_site(atom.pos + newpos, lattice) == i, (atom.pos, newpos, i)
