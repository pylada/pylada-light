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
from pytest import fixture, mark


@fixture
def choice():
    from pylada.vasp.incar._params import Choices
    return Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]})


@mark.parametrize('interchangeable, expected', [
    ('a', 'A'), ('aa', 'A'), (0, 'A'), ('b', 'B'), ('bb', 'B'), (1, 'B'),
    (None, None)
])
def test_interchangeable(interchangeable, expected, choice):
    choice.value = interchangeable
    assert choice.value == expected
    if expected is not None:
        assert choice.incar_string() == 'ALGO = %s' % expected
    else:
        assert choice.incar_string() is None


@mark.parametrize('interchangeable, expected', [
    ('a', 'A'), ('aa', 'A'), (0, 'A'), ('b', 'B'), ('bb', 'B'), (1, 'B'),
    (None, None)
])
def test_interchangeable_and_pickle(interchangeable, expected, choice):
    from pickle import loads, dumps
    choice.value = interchangeable
    choice = loads(dumps(choice))
    assert choice.value == expected
    if expected is not None:
        assert choice.incar_string() == 'ALGO = %s' % expected
    else:
        assert choice.incar_string() is None


def test_fail_on_unexpected_value(choice):
    from pytest import raises

    with raises(ValueError):
        choice.value = 2

    with raises(ValueError):
        choice.value = 'D'


def test_none_value_means_none_incar(choice):
    choice.value = None
    assert choice.incar_string() is None


@mark.parametrize('interchangeable, expected', [
    ('a', 'A'), ('aa', 'A'), (0, 'A'), ('b', 'B'), ('bb', 'B'), (1, 'B'),
    (None, None)
])
def test_is_repreable(choice, interchangeable, expected):
    from pylada.vasp.incar._params import Choices
    choice.value = interchangeable
    choice = eval(repr(choice), {'Choices': Choices})
    assert choice.value == expected
    if expected is not None:
        assert choice.incar_string() == 'ALGO = %s' % expected
    else:
        assert choice.incar_string() is None


@mark.parametrize('value, expected', [('b', 'B'), ('a', 'A'), (1, 'B')])
def test_construction_with_input_value(value, expected):
    from pylada.vasp.incar._params import Choices
    choice = Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]}, value)
    assert choice.value == expected


def test_construction_fails_on_unexpected_value():
    from pylada.vasp.incar._params import Choices
    from pytest import raises

    with raises(ValueError):
        Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]}, 2)
