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
from pytest import fixture


@fixture
def structure():
    from pylada.crystal import Structure

    u = 0.25
    x, y = u, 0.25 - u
    structure = Structure([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) \
        .add_atom(5.000000e-01, 5.000000e-01, 5.000000e-01, "Mg", magmom=-1e0) \
        .add_atom(5.000000e-01, 2.500000e-01, 2.500000e-01, "Mg", magmom=-1e0) \
        .add_atom(2.500000e-01, 5.000000e-01, 2.500000e-01, "Mg", magmom=-1e0) \
        .add_atom(2.500000e-01, 2.500000e-01, 5.000000e-01, "Mg", magmom=-1e0) \
        .add_atom(8.750000e-01, 8.750000e-01, 8.750000e-01, "Al") \
        .add_atom(1.250000e-01, 1.250000e-01, 1.250000e-01, "Al") \
        .add_atom(     x,     x,     x, "O") \
        .add_atom(     x,     y,     y, "O") \
        .add_atom(     y,     x,     y, "O") \
        .add_atom(     y,     y,     x, "O") \
        .add_atom(    -x,    -x,    -x, "O") \
        .add_atom(    -x,    -y,    -y, "O") \
        .add_atom(    -y,    -x,    -y, "O") \
        .add_atom(-y,    -y,    -x, "O")
    return structure


@fixture
def vasp():
    from pylada.vasp import Vasp
    return Vasp()


def test_none(vasp, structure):
    vasp.magmom = None
    assert vasp.magmom is None
    assert vasp._input['magmom'].keyword == 'MAGMOM'
    assert vasp._input['magmom'].output_map(vasp=vasp, structure=structure) is None


def test_ispin_disables_magmom(vasp, structure):
    vasp.magmom = True
    vasp.ispin = 1
    assert vasp._input['magmom'].output_map(vasp=vasp, structure=structure) is None


def test_magmom_disables_isping(vasp, structure):
    vasp.ispin = 2
    vasp.magmom = None
    assert vasp._input['magmom'].output_map(vasp=vasp, structure=structure) is None
    vasp.magmom = False
    assert vasp._input['magmom'].output_map(vasp=vasp, structure=structure) is None


def test_computed_magmom(vasp, structure):
    vasp.ispin = 2
    vasp.magmom = True
    assert 'MAGMOM' in vasp._input['magmom'].output_map(vasp=vasp, structure=structure)
    actual = vasp._input['magmom'].output_map(vasp=vasp, structure=structure)['MAGMOM']
    assert actual == '2*0.0 4*-1.0 8*0.0'


def test_magmom_from_string(vasp, structure):
    vasp.ispin = 2
    vasp.magmom = 'hello'
    assert vasp.magmom == 'hello'
    assert 'MAGMOM' in vasp._input['magmom'].output_map(vasp=vasp, structure=structure)
    assert vasp._input['magmom'].output_map(vasp=vasp, structure=structure)['MAGMOM'] == 'hello'


def test_repr(vasp, structure):
    vasp.ispin = 2
    vasp.magmom = 'hello'
    assert repr(vasp._input['magmom']) == "Magmom(value='hello')"


def test_pickling(vasp, structure):
    from pickle import loads, dumps
    vasp.ispin = 2
    vasp.magmom = 'hello'
    assert repr(loads(dumps(vasp._input['magmom']))) == "Magmom(value='hello')"


def test_complex_magmom(vasp, structure):
    for atom, mag in zip(structure, [1, -1, 1, 1]):
        if atom.type == 'Mg':
            atom.magmom = mag
    for atom, mag in zip(structure, [0.5, 0.5]):
        if atom.type == 'Al':
            atom.magmom = mag

    vasp.ispin = 2
    vasp.magmom = True
    assert 'MAGMOM' in vasp._input['magmom'].output_map(vasp=vasp, structure=structure)
    actual = vasp._input['magmom'].output_map(vasp=vasp, structure=structure)['MAGMOM']
    assert actual == '2*0.0 1.0 -1.0 2*1.0 8*0.0'
