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

""" Tests for primitive and is_primitive. """
from pytest import mark


def itercells(nmax, prune=-1):
    """ Iterates over cells with ``nmax`` and fewer atoms

        :param nmax:
            Max number of atoms
        :param prune:
            Yields cell only if random number between 0 and 1 is larger than this parameter.
    """
    from itertools import product
    from random import random
    from numpy import zeros
    cell = zeros((3, 3), dtype="float64")
    # loop over all possible cells with less than 10 atoms.
    for n, a in product(range(2, nmax + 1), range(1, nmax + 1)):
        if a > n or n % a != 0:
            continue
        ndiv_a = n / a
        cell[0, 0] = a
        for b in range(1, ndiv_a + 1):
            if ndiv_a % b != 0:
                continue
            cell[1, 1] = b
            c = ndiv_a / b
            cell[2, 2] = c
            for d, e, f in product(range(b), range(c), range(c)):
                cell[1, 0] = d
                cell[2, 0] = e
                cell[2, 1] = f
                if random() > prune:
                    yield cell


def is_integer(o, tolerance=1e-8):
    from numpy import allclose, floor
    return allclose(o, floor(o + 0.1), tolerance)


def test_lattice_is_primitive():
    from pylada.crystal import Structure, is_primitive

    lattice = Structure(0.0, 0.5, 0.5,
                        0.5, 0.0, 0.5,
                        0.5, 0.5, 0.0, scale=2.0, m=True ) \
        .add_atom(0, 0, 0, "As")                           \
        .add_atom(0.25, 0.25, 0.25, ['In', 'Ga'], m=True)

    assert is_primitive(lattice)


@mark.parametrize('cell', itercells(10, prune=0.95))
def test_primitive(cell):
    """ Tests whether primitivization works. """
    from numpy import abs, dot
    from numpy.linalg import inv
    from pylada.crystal import supercell, Structure, are_periodic_images as api, primitive, \
        is_primitive

    lattice = Structure(0.0, 0.5, 0.5,
                        0.5, 0.0, 0.5,
                        0.5, 0.5, 0.0, scale=2.0, m=True ) \
        .add_atom(0, 0, 0, "As")                           \
        .add_atom(0.25, 0.25, 0.25, ['In', 'Ga'], m=True)

    structure = supercell(lattice, dot(lattice.cell, cell))
    assert not is_primitive(structure)
    structure = primitive(structure, 1e-8)
    assert is_primitive(structure)
    assert abs(structure.volume - lattice.volume) < 1e-8
    assert len(structure) == len(lattice)
    assert is_integer(dot(structure.cell, inv(lattice.cell)))
    assert is_integer(dot(lattice.cell, inv(structure.cell)))
    invcell = inv(lattice.cell)
    for atom in structure:
        assert api(lattice[atom.site].pos, atom.pos, invcell)
        assert atom.type == lattice[atom.site].type
        assert getattr(lattice[atom.site], 'm', False) == getattr(atom, 'm', False)
        assert (getattr(atom, 'm', False) or atom.site == 0)
