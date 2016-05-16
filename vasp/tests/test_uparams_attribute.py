###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is vasp high throughput computational platform for Physics. It aims to make it easier to submit
#  large numbers of jobs on supercomputers. It provides vasp python interface to physical input, such as
#  crystal structures, as well as to vasp number of DFT (VASP, CRYSTAL) and atomic potential programs. It
#  is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at
#  your option) any later version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR vasp PARTICULAR PURPOSE.  See the GNU General
#  Public License for more details.
#
#  You should have received vasp copy of the GNU General Public License along with PyLaDa.  If not, see
#  <http://www.gnu.org/licenses/>.
###############################
from pytest import fixture


@fixture
def structure():
    from pylada.crystal import Structure
    u = 0.25
    x, y = u, 0.25 - u
    structure = Structure([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) \
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
    return structure


@fixture
def Specie():
    from collections import namedtuple
    return namedtuple('Specie', ['U'])


@fixture
def vasp(Specie):
    from pylada.vasp import Vasp
    from pylada.vasp.specie import U, nlep
    vasp = Vasp()
    vasp.species = {'A': Specie([]), 'B': Specie([]), 'X': Specie([])}
    for key in list(vasp._input.keys()):
        if key not in ['ldau']:
            del vasp._input[key]
    return vasp


def test_ldau_keyword_without_nlep(vasp, structure):
    from pickle import loads, dumps
    import pylada
    pylada.vasp_has_nlep = False
    assert vasp.ldau == True

    keyword = vasp._input['ldau']
    assert keyword.output_map(vasp=vasp, structure=structure) is None
    assert eval(repr(keyword), {'LDAU': keyword.__class__})._value == True
    assert eval(repr(keyword), {'LDAU': keyword.__class__}).keyword == 'LDAU'
    assert loads(dumps(keyword)).keyword == 'LDAU'
    assert loads(dumps(keyword))._value


def test_U_disabled(vasp, structure, Specie):
    from pylada.vasp.specie import U
    import pylada
    pylada.vasp_has_nlep = False
    vasp.ldau = False
    vasp.species = {'A': Specie([U(2, 0, 0.5)]), 'B': Specie([]), 'X': Specie([])}
    assert vasp.ldau == False
    assert vasp.output_map(vasp=vasp) is None


def test_enabled_U(vasp, Specie, structure):
    from numpy import all, abs, array
    from pylada.vasp.specie import U
    import pylada
    pylada.vasp_has_nlep = False
    vasp.species = {'A': Specie([U(2, 0, 0.5)]), 'B': Specie([]), 'X': Specie([])}
    vasp.ldau = True
    assert vasp.ldau == True
    map = vasp.output_map(vasp=vasp, structure=structure)
    assert map['LDAU'] == '.TRUE.'
    assert map['LDAUTYPE'] == '2'
    assert all(abs(array(map['LDUJ'].split(), dtype='float64')) < 1e-8)
    assert all(abs(array(map['LDUU'].split(), dtype='float64') - [0.5, 0, 0]) < 1e-8)
    assert all(abs(array(map['LDUL'].split(), dtype='float64') - [0, -1, -1]) < 1e-8)


def test_enabled_complex_U(vasp, Specie, structure):
    from numpy import all, abs, array
    from pylada.vasp.specie import U
    import pylada
    pylada.vasp_has_nlep = False
    vasp.species = {'A': Specie([U(2, 0, 0.5)]), 'B': Specie([U(2, 1, 0.6)]), 'X': Specie([])}
    vasp.ldau = True
    map = vasp.output_map(vasp=vasp, structure=structure)
    assert map['LDAU'] == '.TRUE.'
    assert map['LDAUTYPE'] == '2'
    assert all(abs(array(map['LDUJ'].split(), dtype='float64')) < 1e-8)
    assert all(abs(array(map['LDUU'].split(), dtype='float64') - [0.5, 0.6, 0]) < 1e-8)
    assert all(abs(array(map['LDUL'].split(), dtype='float64') - [0, 1, -1]) < 1e-8)


def test_disabled_nlep(vasp, Specie, structure):
    from numpy import all, abs, array
    from pylada.vasp.specie import U, nlep
    import pylada
    pylada.vasp_has_nlep = True

    vasp.species = {
        'A': Specie([U(2, 0, 0.5)]),
        'B': Specie([U(2, 0, -0.5), nlep(2, 1, -1.0)]),
        'X': Specie([])
    }
    vasp.ldau = False
    assert vasp.ldau == False
    assert vasp.output_map(vasp=vasp) is None


def test_enabled_nlep(vasp, Specie, structure):
    from numpy import all, abs, array
    from pylada.vasp.specie import U, nlep
    import pylada
    pylada.vasp_has_nlep = True

    vasp.species = {
        'A': Specie([U(2, 0, 0.5)]),
        'B': Specie([U(2, 0, -0.5), nlep(2, 1, -1.0)]),
        'X': Specie([])
    }
    vasp.ldau = True
    map = vasp.output_map(vasp=vasp, structure=structure)
    assert map['LDAU'] == '.TRUE.'
    assert map['LDAUTYPE'] == '2'
    assert all(abs(array(map['LDUL1'].split(), dtype='float64') - [0, 0, -1]) < 1e-8)
    assert all(abs(array(map['LDUU1'].split(), dtype='float64') - [0.5, -0.5, 0]) < 1e-8)
    assert all(abs(array(map['LDUJ1'].split(), dtype='float64') - [0, 0, 0]) < 1e-8)
    assert all(abs(array(map['LDUO1'].split(), dtype='float64') - [1, 1, 1]) < 1e-8)
    assert all(abs(array(map['LDUL2'].split(), dtype='float64') - [-1, 1, -1]) < 1e-8)
    assert all(abs(array(map['LDUU2'].split(), dtype='float64') - [0, -1.0, 0]) < 1e-8)
    assert all(abs(array(map['LDUJ2'].split(), dtype='float64') - [0, 0, 0]) < 1e-8)
    assert all(abs(array(map['LDUO2'].split(), dtype='float64') - [1, 2, 1]) < 1e-8)


def test_enabled_complex_nlep(vasp, structure, Specie):
    from numpy import all, abs, array
    from pylada.vasp.specie import U, nlep
    import pylada
    pylada.vasp_has_nlep = True

    vasp.species = {
        'A': Specie([U(2, 0, 0.5)]),
        'B': Specie([U(2, 0, -0.5), nlep(2, 2, -1.0, -3.0)]),
        'X': Specie([])
    }
    vasp.ldau = True
    map = vasp.output_map(vasp=vasp, structure=structure)
    assert map['LDAU'] == '.TRUE.'
    assert map['LDAUTYPE'] == '2'
    assert all(abs(array(map['LDUL1'].split(), dtype='float64') - [0, 0, -1]) < 1e-8)
    assert all(abs(array(map['LDUU1'].split(), dtype='float64') - [0.5, -0.5, 0]) < 1e-8)
    assert all(abs(array(map['LDUJ1'].split(), dtype='float64') - [0, 0, 0]) < 1e-8)
    assert all(abs(array(map['LDUO1'].split(), dtype='float64') - [1, 1, 1]) < 1e-8)
    assert all(abs(array(map['LDUL2'].split(), dtype='float64') - [-1, 2, -1]) < 1e-8)
    assert all(abs(array(map['LDUU2'].split(), dtype='float64') - [0, -1.0, 0]) < 1e-8)
    assert all(abs(array(map['LDUJ2'].split(), dtype='float64') - [0, -3.0, 0]) < 1e-8)
    assert all(abs(array(map['LDUO2'].split(), dtype='float64') - [1, 3, 1]) < 1e-8)
