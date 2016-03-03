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
""" Checks atom methods and attributes. """

from pylada.crystal.atom import Atom


def test_init():
    """ Test atom initialization. """
    from numpy import all, abs, array

    # Try correct initialization. Check for garbage collection.
    a = Atom()
    assert a.type is None and all(abs(a.pos) < 1e-8) and len(a.__dict__) == 2
    assert a.type is None and all(abs(a.pos) < 1e-8) and len(a.__dict__) == 2

    a = Atom(0.1, 0.1, 0.1, 'Au')
    assert a.type == "Au"
    assert all(abs(a.pos - 0.1) < 1e-8)
    assert len(a.__dict__) == 2

    a = Atom(0.1, 0.1, 0.1, type=['Au', 'Pd'])
    assert a.type == ["Au", 'Pd']
    assert all(abs(a.pos - 0.1) < 1e-8)
    assert len(a.__dict__) == 2

    a = Atom(type='Au', pos=[0.1, 0.1, 0.1])
    assert a.type == "Au" and all(
        abs(a.pos - 0.1) < 1e-8) and len(a.__dict__) == 2
    a = Atom(type='Au', pos=[0.1, 0.1, 0.1], m=5)
    assert a.type == "Au" and all(
        abs(a.pos - 0.1) < 1e-8) and len(a.__dict__) == 3 and getattr(a, 'm', 3) == 5
    a = Atom(0.1, 0.1, 0.1, 0.1, 0.1)
    assert all(abs(array(a.type) - 0.1) < 1e-8) and all(abs(a.pos - 0.1)
                                                        < 1e-8) and len(a.__dict__) == 2
    l = [None]
    a = Atom(0.1, 0.1, 0.1, l)
    assert a.type is l
    assert all(abs(a.pos - 0.1) < 1e-8) and len(a.__dict__) == 2
    a.pos[0] = 0.2
    a.pos[1] = 0.3
    a.pos[2] = 0.4
    assert all(abs(a.pos - [0.2, 0.3, 0.4]) < 1e-8)


def test_fail_init():
    """ Test failures during initialization. """
    from pytest import raises
    with raises(TypeError):
      a = Atom(0.1, 0.1, 0.1, 'Au', type='Au')
    with raises(TypeError):
      a = Atom(0.1, 0.1, 0.1, pos=[0.1, 0.1, 0.1])


def test_repr():
    """ Test representability. """
    actual = repr(Atom(type='Au', pos=[1, 1, 1], m=1, dtype='int64'))
    assert actual == "{0.__name__}(1, 1, 1, 'Au', dtype='int64', m=1)".format(Atom)
    actual = str(Atom(type='Au', pos=[1., 1, 1], site=1))
    assert actual == "{0.__name__}(1.0, 1.0, 1.0, 'Au', site=1)".format(Atom)


def test_copy():
   """ Test copy and deep copy. """
   from copy import copy, deepcopy
   b = Atom(0, 0, 0, 'Au', m=0)
   a = copy(b)
   b.type = 'Pd'
   assert a.pos is b.pos
   assert a.type == 'Au'
   a = deepcopy(b)
   b.type = 'Au'
   b.pos += 1
   del b.m
   assert a.type == "Pd"
   assert all(abs(a.pos - 0.0) < 1e-8)
   assert len(a.__dict__) == 3
   assert getattr(a, 'm', 1) == 0


def test_todict():
    """ Test to_dict member. """
    a = Atom(0, 0, 0, 'Au', m=0)
    a = a.to_dict()
    assert a['type'] == "Au" and all(a['pos'] == 0) and a['m'] == 0


def test_pickle():
    """ Test pickling. """
    from pickle import loads, dumps
    a = Atom(pos=[0, 1, 2], type="Au", m=6)
    b = loads(dumps(a))
    assert b.__class__ is a.__class__
    assert all(abs(b.pos - [0, 1, 2]) < 1e-12)
    assert b.type == 'Au'
    assert len(b.__dict__) == 3
    assert b.__dict__.get('m', 0) == 6
    b = loads(dumps((a, a)))
    assert b[0] is b[1]
