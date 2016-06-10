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
# -*- coding: utf-8 -*-
from pytest import fixture, mark
from pylada.espresso import structure_handling as sh
from quantities import angstrom, bohr_radius


@fixture
def pwscfin(tmpdir):
    string = """
        &system
            ibrav=5,
            celldm = 1.0 0.0 0.0 0.5,
        /

    ATOMIC_POSITIONS alat
    A 0 0 0
    B 1 2 3
    """
    tmpdir.join('pos.in').write(string)
    return tmpdir.join('pos.in')


def get_namelist(ibrav, **kwargs):
    from pylada.espresso import Namelist
    result = Namelist({'ibrav': ibrav})
    for key, value in kwargs.items():
        if value is not None:
            setattr(result, key, value)

    return result


@mark.parametrize('celldim, subtitle, expected_scale', [
    (None, 'bohr', bohr_radius),
    (None, 'angstrom', angstrom),
    (None, None, bohr_radius),
    ('A', None, angstrom),
    ([2.5], 'alat', 2.5 * bohr_radius),
    # card takes precedence over celldm
    ('A', 'bohr', bohr_radius),
    ([2.5], 'bohr', bohr_radius),
    ([2.5], 'angstrom', angstrom),
])
def test_free_cell(celldim, subtitle, expected_scale):
    from numpy import allclose, abs, array
    from pylada.espresso import Card, Namelist

    namelist = Namelist({'ibrav': 0})
    if celldim is not None:
        namelist.celldm = celldim

    card = Card('CELL_PARAMETERS', subtitle=subtitle, value="""
        1 2 3
        2 3 4
        4 4 6
    """)

    cell, scale = sh._read_free(namelist, card)
    assert allclose(cell, array([[1, 2, 4], [2, 3, 4], [3, 4, 6]], dtype='float64'))
    assert abs(scale - expected_scale) < 1e-8


@mark.parametrize('celldim, A, expected_scale', [
    (None, 0.75, 0.75 * angstrom),
    ('A', None, angstrom),
    ([1.1], None, 1.1 * bohr_radius)
])
def test_cubic_cell_and_scale(celldim, A, expected_scale):
    from numpy import abs, allclose
    namelist = get_namelist(1, celldm=celldim, A=A)
    cell, scale = sh.read_cell_and_scale(namelist, None)
    assert allclose(cell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert abs(scale - expected_scale) < 1e-8


@mark.parametrize('celldim, A, expected_scale', [
    (None, 0.75, 0.75 * angstrom),
    ('A', None, angstrom),
    ([1.1], None, 1.1 * bohr_radius)
])
def test_fcc_cell_and_scale(celldim, A, expected_scale):
    from numpy import abs, allclose
    namelist = get_namelist(2, celldm=celldim, A=A)
    cell, scale = sh.read_cell_and_scale(namelist, None)
    assert allclose(2 * cell, [[-1, 0, -1], [0, 1, 1], [1, 1, 0]])
    assert abs(scale - expected_scale) < 1e-8


@mark.parametrize('celldim, A, expected_scale', [
    (None, 0.75, 0.75 * angstrom),
    ('A', None, angstrom),
    ([1.1], None, 1.1 * bohr_radius)
])
def test_bcc_cell_and_scale(celldim, A, expected_scale):
    from numpy import abs, allclose
    namelist = get_namelist(3, celldm=celldim, a=A)
    cell, scale = sh.read_cell_and_scale(namelist, None)
    assert allclose(2 * cell, [[1, -1, -1], [1, 1, -1], [1, 1, 1]])
    assert abs(scale - expected_scale) < 1e-8


@mark.parametrize('celldim, A, expected_scale, C, c_over_a', [
    (None, 0.75, 0.75 * angstrom, 1, 1. / 0.75),
    ([0.75, 0, 1. / 0.75], None, 0.75 * bohr_radius, None, 1. / 0.75),
])
def test_hexa_cell_and_scale(celldim, A, expected_scale, C, c_over_a):
    from numpy import abs, allclose, sqrt
    namelist = get_namelist(4, celldm=celldim, a=A, c=C)
    cell, scale = sh.read_cell_and_scale(namelist, None)
    expected_cell = [[1, -0.5, 0], [0, sqrt(3.) / 2., 0], [0, 0, c_over_a]]
    assert allclose(cell, expected_cell)
    assert abs(scale - expected_scale) < 1e-8


def test_read_structure(pwscfin):
    from numpy import allclose, sqrt
    from quantities import bohr_radius
    from pylada.espresso.structure_handling import read_structure
    structure = read_structure(str(pwscfin))

    c = 0.5
    tx, ty, tz = sqrt((1. - c) / 2.), sqrt((1. - c) / 6.), sqrt((1. + 2. * c) / 3.)
    assert allclose(structure.cell.transpose(), [[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]])
    assert abs(structure.scale - bohr_radius) < 1e-8
    assert len(structure) == 2
    assert structure[0].type == 'A'
    assert allclose(structure[0].pos, [0, 0, 0.])
    assert structure[1].type == 'B'
    assert allclose(structure[1].pos, [1, 2, 3.])


def test_write_read_loop(tmpdir, pwscfin):
    from numpy import abs, allclose
    from f90nml import Namelist as F90Namelist
    from pylada.espresso.structure_handling import add_structure, read_structure
    from pylada.espresso.misc import write_pwscf_input
    structure = read_structure(str(pwscfin))

    namelist = F90Namelist()
    cards = []
    add_structure(structure, namelist, cards)
    write_pwscf_input(namelist, cards, str(tmpdir.join('other.in')))
    reread = read_structure(str(tmpdir.join('other.in')))

    assert allclose(reread.cell, structure.cell)
    assert abs(reread.scale - structure.scale) < 1e-12
    assert len(reread) == len(structure)
    for i in range(len(reread)):
        assert allclose(reread[i].pos, structure[i].pos)
        assert reread[i].type == structure[i].type


def test_read_forces(tmpdir):
    from quantities import Ry, bohr_radius as a0
    from numpy import allclose
    from pylada.espresso.structure_handling import read_structure
    string = """
        &system
            ibrav=5,
            celldm = 1.0 0.0 0.0 0.5,
        /

    ATOMIC_POSITIONS alat
    A 0 0 0
    B 1 2 3
    ATOMIC_FORCES
    A 0 0 0
    B 1 2 3
    """
    tmpdir.join('pos.in').write(string)
    structure = read_structure(str(tmpdir.join('pos.in')))
    for atom in structure:
        assert hasattr(atom, 'force')
        assert atom.force.units == (Ry / a0)
    assert allclose(structure[0].force.magnitude, [0, 0, 0])
    assert allclose(structure[1].force.magnitude, [1, 2, 3])
