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


@fixture
def simple_namelist():
    from collections import OrderedDict
    return OrderedDict([
        ('ibrav', 2),
        ('celldm', [7.5]),
        ('nat', 1),
        ('ntyp', 1),
        ('ecutwfc', 12.0),
        ('occupations', 'smearing'),
        ('smearing', 'marzari-vanderbilt'),
        ('degauss', 0.06)
    ])


@mark.parametrize('name, type_, value', [
    ('ibrav', int, 2),
    ('nat', int, 1),
    ('ntyp', int, 1),
    ('ecutwfc', float, 12.0),
    ('celldm', list, [7.5]),
    ('occupations', str, 'smearing'),
    ('smearing', str, 'marzari-vanderbilt'),
    ('degauss', float, 0.06)
])
def test_scalar_namelist_attributes(simple_namelist, name, type_, value):
    from numpy import abs, allclose
    from collections import Sequence
    from pylada.espresso import Namelists

    nl = Namelists(simple_namelist)
    assert hasattr(nl, name)
    assert isinstance(getattr(nl, name), type_)
    if type_ == float:
        assert abs(getattr(nl, name) - value) < 1e-12
    elif type_ == list:
        assert allclose(getattr(nl, name), value, 1e-12)
    else:
        assert getattr(nl, name) == value
