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
from pylada.espresso import Namelist


@fixture
def recursive_namelist():
    from collections import OrderedDict
    return OrderedDict([
        ('control', OrderedDict([
            ('prefix', 'al'),
            ('outdir', 'temporary directory for large files'),
            ('pseudo_dir',
             'directory where pp-files are kept')
        ])),
        ('system', OrderedDict([
            ('ibrav', 2),
            ('celldm', [7.5]),
            ('nat', 1),
            ('ntyp', 1),
            ('ecutwfc', 12.0),
            ('occupations', 'smearing'),
            ('smearing', 'marzari-vanderbilt'),
            ('degauss', 0.06)
        ])),
        ('electrons', OrderedDict())])


@fixture
def simple_namelist(recursive_namelist):
    return recursive_namelist['system']


@fixture
def WithTraitLets():
    from traitlets import Enum

    class WithTraitLets(Namelist):
        ibrav = Enum([0, 1, 2, 3, 4, 5, -5, 6, 7, 8, 9, -9, 10, 11, 12, -12, 13, 14], 0,
                     help="Bravais class", allow_none=True)
    return WithTraitLets


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

    nl = Namelist(simple_namelist)
    assert hasattr(nl, name)
    assert isinstance(getattr(nl, name), type_)
    if type_ == float:
        assert abs(getattr(nl, name) - value) < 1e-12
    elif type_ == list:
        assert allclose(getattr(nl, name), value, 1e-12)
    else:
        assert getattr(nl, name) == value


def test_recursive_namelist_attributes(recursive_namelist):
    nl = Namelist(recursive_namelist)
    assert hasattr(nl, 'system')
    assert isinstance(getattr(nl, 'system'), Namelist)
    assert getattr(nl.system, 'ibrav', 0) == 2
    assert len(nl) == 3


def test_empty_namelists_do_appear(recursive_namelist):
    nl = Namelist(recursive_namelist)
    assert hasattr(nl, 'electrons')
    assert isinstance(getattr(nl, 'electrons'), Namelist)
    assert len(nl.electrons) == 0


def test_simple_back_to_ordered(simple_namelist):
    nl = Namelist(simple_namelist)
    assert len(nl) > 0

    back = nl.ordered_dict()
    assert len(back) == len(simple_namelist)
    for back_key, key in zip(back, simple_namelist):
        assert back_key == key
        assert back[key] == simple_namelist[key]


def test_recursive_back_to_ordered(recursive_namelist):
    from collections import OrderedDict
    nl = Namelist(recursive_namelist)
    assert len(nl) > 0

    back = nl.ordered_dict()
    assert len(back) == len(recursive_namelist)
    for back_key, key in zip(back, recursive_namelist):
        assert back_key == key
        assert isinstance(back[key], OrderedDict)


def test_set_known_attributes(recursive_namelist):
    nl = Namelist(recursive_namelist)
    nl.system.ibrav = 2
    assert nl.system.ibrav == 2


def test_add_namelist_attribute(recursive_namelist):
    nl = Namelist(recursive_namelist)
    nl.system.bravasi = 2
    assert nl.system.bravasi == 2
    assert 'bravasi' in nl.system.ordered_dict()


def test_add_private_attribute(recursive_namelist):
    nl = Namelist(recursive_namelist)
    nl.system._bravasi = 2
    assert nl.system._bravasi == 2
    assert '_bravasi' not in nl.system.ordered_dict()


def test_delete_namelist_attribute(recursive_namelist):
    from pytest import raises
    nl = Namelist(recursive_namelist)
    del nl.system.ibrav
    with raises(AttributeError):
        nl.system.ibrav
    assert 'ibrav' not in nl.system.ordered_dict()


def test_delete_private_attribute(recursive_namelist):
    from pytest import raises
    nl = Namelist(recursive_namelist)
    nl._private = 0
    del nl._private
    with raises(AttributeError):
        nl._private


def test_deleting_uknown_attribute_fails(recursive_namelist):
    from pytest import raises
    nl = Namelist(recursive_namelist)
    with raises(AttributeError):
        del nl.system.ibravi

    with raises(AttributeError):
        del nl.system._ibravi


def test_traitlets_from_empty(WithTraitLets):
    from pytest import raises
    from traitlets import TraitError
    nl = WithTraitLets()
    assert nl.ibrav == 0

    nl.ibrav = 2
    assert nl.ibrav == 2

    with raises(TraitError):
        nl.ibrav = 15


def test_traitlets_appear_in_dict(WithTraitLets):
    from pytest import raises
    from traitlets import TraitError
    nl = WithTraitLets()
    assert 'ibrav' in nl.ordered_dict()
    assert 'ibrav' not in nl.__dict__['_Namelist__inputs']


def test_traitlets_from_filled(simple_namelist, WithTraitLets):
    from pytest import raises
    from traitlets import TraitError
    nl = WithTraitLets(simple_namelist)
    assert nl.ibrav == 2

    nl.ibrav = 3
    assert nl.ibrav == 3
    assert 'ibrav' in nl.ordered_dict()
    with raises(TraitError):
        nl.ibrav = 15


def test_traitlets_cannot_be_deleted(WithTraitLets):
    from pytest import raises
    from traitlets import TraitError
    nl = WithTraitLets()
    with raises(AttributeError):
        del nl.ibrav


def test_none_arguments_do_not_appear_in_dict(simple_namelist):
    nl = Namelist(simple_namelist)
    nl.ibrav = None
    assert 'ibrav' not in nl.ordered_dict()


def test_none_traitelets_do_not_appear_in_dict(WithTraitLets):
    from pytest import raises
    from traitlets import TraitError
    nl = WithTraitLets()
    nl.ibrav = None
    assert 'ibrav' not in nl.ordered_dict()


def test_input_transform(WithTraitLets):
    from pylada.espresso.namelists import input_transform
    class Transformed(WithTraitLets):

        @input_transform
        def __transform_ibrav(self, dictionary, value):
            dictionary['ibrav'] = value

    nl = Transformed()
    assert nl.ordered_dict(value=5)['ibrav'] == 5
    assert nl.ordered_dict(value=6)['ibrav'] == 6
