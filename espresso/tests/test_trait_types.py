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
import pylada  #  makes sure bohr_radius is in quantities
from pytest import fixture, mark
from pylada.espresso.trait_types import DimensionalTrait
from quantities import bohr_radius, angstrom, eV
from traitlets import HasTraits


@fixture
def Dimensional():
    class Dimensional(HasTraits):
        dimensional = DimensionalTrait(angstrom, default_value=1 * bohr_radius)
    return Dimensional


@fixture
def dimensional(Dimensional):
    return Dimensional()


def test_dimensional_has_default_value(dimensional):
    from numpy import all, abs
    assert dimensional.dimensional.units == angstrom
    assert all(abs(dimensional.dimensional - 1 * bohr_radius) < 1e-8)


def test_dimensional_can_set_dimensional_value(dimensional):
    from numpy import all, abs
    dimensional.dimensional = 5 * bohr_radius
    assert dimensional.dimensional.units == angstrom
    assert all(abs(dimensional.dimensional - 5 * bohr_radius) < 1e-8)


def test_dimensional_with_none():
    class Dimensional(HasTraits):
        dimensional = DimensionalTrait(angstrom, allow_none=True, default_value=None)

    a = Dimensional()
    assert a.dimensional is None


@mark.parametrize('value, expected', [
    (5 * bohr_radius, 5 * bohr_radius),
    (5, 5 * angstrom),
    ([5, 6] * bohr_radius, [5, 6] * bohr_radius),
    ([5, 6], [5, 6] * angstrom),
])
def test_dimensional_can_set_values(dimensional, value, expected):
    from numpy import all, abs
    dimensional.dimensional = value
    assert dimensional.dimensional.units == angstrom
    assert all(abs(dimensional.dimensional - expected) < 1e-8)


@mark.parametrize('value, exception', [
    (5 * eV, ValueError),
    ('a string', TypeError)
])
def test_fail_on_incorrect_input(dimensional, value, exception):
    from pytest import raises
    with raises(exception):
        dimensional.dimensional = value

def test_lowercasecard():
    from pylada.espresso.trait_types import LowerCaseUnicode
    class LowerCase(HasTraits):
        case = LowerCaseUnicode()
    lower = LowerCase()
    assert lower.case is None
    lower.case = "AAaa"
    assert lower.case == "aaaa"
