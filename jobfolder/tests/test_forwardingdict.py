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


class A(object):

    def __init__(self, this, that):
      self.this = this
      self.that = that

    def __eq__(self, b):
      return b.__class__ is self.__class__ and b.this == self.this and b.that == self.that

    def __neq__(self, b): return not self.__eq__(b)

    def __repr__(self): return "A({0.this}, {0.that})".format(self)


@fixture
def first():
    return A(0, A(5, A('a', 'b')))


@fixture
def second():
    return A(1, A(6, A('c', 'd')))


@fixture
def dictionary(first, second):
    from pylada.jobfolder.forwarding_dict import ForwardingDict
    dictionary = ForwardingDict(ordered=True, readonly=True)
    dictionary['first'] = first
    dictionary['second'] = second
    return dictionary


@fixture
def single_item_dict(dictionary, first):
    # create dictionary with single item
    dictionary.readonly = False
    del dictionary.that.this
    first.that.this = 5
    return dictionary


def test_Aclass_fixture(first, second):
    third = A(0, A(5, A('a', 'b')))
    assert first == third
    third.that.that.that = 'd'
    assert first != third


def test_attribute_forwarding(first, second, dictionary):
    assert dictionary['first'].this == 0
    assert dictionary['first'].that.this == 5
    assert dictionary['first'].that.that.this == 'a'
    assert dictionary['first'].that.that.that == 'b'
    assert dictionary['second'].this == 1
    assert dictionary['second'].that.this == 6
    assert dictionary['second'].that.that.this == 'c'
    assert dictionary['second'].that.that.that == 'd'


def test_repr(first, second, dictionary):
    assert repr(dictionary) == \
        "{\n  'second': A(1, A(6, A(c, d))),\n  'first':  A(0, A(5, A(a, b))),\n}"
    assert repr(dictionary.this) == "{\n  'second': 1,\n  'first':  0,\n}"
    assert repr(dictionary.that) == "{\n  'second': A(6, A(c, d)),\n  'first':  A(5, A(a, b)),\n}"
    assert repr(dictionary.that.this) == "{\n  'second': 6,\n  'first':  5,\n}"
    assert repr(dictionary.that.that) == "{\n  'second': A(c, d),\n  'first':  A(a, b),\n}"
    assert repr(dictionary.that.that.this) == "{\n  'second': 'c',\n  'first':  'a',\n}"
    assert repr(dictionary.that.that.that) == "{\n  'second': 'd',\n  'first':  'b',\n}"


def test_iteration(first, second, dictionary):
    for key, value in dictionary.items():
        assert {'first': first, 'second': second}[key] == value

    for key, value in dictionary.this.items():
        assert {'first': first.this, 'second': second.this}[key] == value

    for key, value in dictionary.that.items():
        assert {'first': first.that, 'second': second.that}[key] == value

    for key, value in dictionary.that.this.items():
        assert {'first': first.that.this, 'second': second.that.this}[key] == value

    for key, value in dictionary.that.that.items():
        assert {'first': first.that.that, 'second': second.that.that}[key] == value

    for key, value in dictionary.that.that.this.items():
        assert {'first': first.that.that.this, 'second': second.that.that.this}[key] == value

    for key, value in dictionary.that.that.that.items():
        assert {'first': first.that.that.that, 'second': second.that.that.that}[key] == value


def test_fail_on_getting_missing_attribute(dictionary):
    from pytest import raises
    with raises(AttributeError):
        dictionary.this.that


def test_fail_on_setting_missing_attribute(dictionary):
    from pytest import raises
    with raises(RuntimeError):
        dictionary.this = 8


def test_fail_on_deleting_missing_attribute(dictionary):
    from pytest import raises
    with raises(RuntimeError):
        del dictionary.this


def test_writing_to_dict(dictionary):
    dictionary.readonly = False
    assert all([u != 8 for u in dictionary.this.values()])
    dictionary.this = 8
    assert all([u == 8 for u in dictionary.this.values()])
    assert all([u != 8 for u in dictionary.that.this.values()])
    dictionary.that.this = 8
    assert all([u == 8 for u in dictionary.that.this.values()])


def test_cannot_write_to_read_only(dictionary):
    from pytest import raises
    dictionary.readonly = True
    with raises(RuntimeError):
        dictionary.this = 8


def test_cannot_delete_attribute_from_readonly(dictionary):
    from pytest import raises
    dictionary.readonly = True
    with raises(RuntimeError):
        del dictionary.this


def test_deleting_attributes(first, second, dictionary):
    from pytest import raises
    dictionary.readonly = False
    del dictionary.that.this
    assert not hasattr(first.that, 'this')
    assert not hasattr(second.that, 'this')
    assert hasattr(first.that.that, 'this') and hasattr(first, 'this')
    assert hasattr(second.that.that, 'this') and hasattr(second, 'this')
    with raises(AttributeError):
        dictionary.that.this


def test_naked_end_false(first, second, single_item_dict):
    single_item_dict.naked_end = False
    assert single_item_dict.that.this.values()[0] == first.that.this


def test_naked_end_true(first, second, single_item_dict):
    single_item_dict.naked_end = True
    assert single_item_dict.that.this == first.that.this


def test_modify_only_existing(first, second, dictionary):
    from pytest import raises
    dictionary.readonly = False
    dictionary.only_existing = True
    with raises(AttributeError):
        dictionary.that.other = True


def test_add_missing_attributes(first, second, dictionary):
    dictionary.readonly = False
    dictionary.only_existing = False
    dictionary.that.other = True
    assert getattr(first.that, 'other', False) == True
    assert getattr(second.that, 'other', False) == True


def test_add_missing_nested_attributes(first, second, dictionary):
    dictionary.readonly = False
    dictionary.only_existing = False
    del first.that.that
    dictionary.that.that.other = True
    assert getattr(second.that.that, 'other', False) == True
