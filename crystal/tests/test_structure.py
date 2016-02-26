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

""" Checks structure methods and attributes. """
from pytest import raises
from numpy import all, abs, array, identity
from quantities import angstrom, nanometer

from pylada.crystal.structure import Structure
from pylada.crystal.atom import Atom


def test_initialization():
    """ Test structure initialization. """
    a = Structure()
    assert all(abs(a.cell - identity(3)) < 1e-8)
    assert abs(a.scale - 1e0 * angstrom) < 1e0
    assert len(a.__dict__) == 3

    a = Structure(identity(3) * 2.5, scale=5.45)
    assert all(abs(a.cell - identity(3) * 2.5) < 1e-8)
    assert abs(a.scale - 5.45 * angstrom) < 1e0
    assert len(a.__dict__) == 3

    a = Structure(identity(3) * 2.5, scale=0.545 * nanometer)
    assert all(abs(a.cell - identity(3) * 2.5) < 1e-8)
    assert abs(a.scale - 5.45 * angstrom) < 1e0
    assert len(a.__dict__) == 3

    a = Structure(2.5, 0, 0, 0, 2.5, 0, 0, 0, 2.5, scale=5.45)
    assert all(abs(a.cell - identity(3) * 2.5) < 1e-8)
    assert abs(a.scale - 5.45 * angstrom) < 1e0
    assert len(a.__dict__) == 3

    a = Structure([2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5], scale=5.45)
    assert all(abs(a.cell - identity(3) * 2.5) < 1e-8)
    assert abs(a.scale - 5.45 * angstrom) < 1e0
    assert len(a.__dict__) == 3

    a = Structure(cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]], scale=5.45)
    assert all(abs(a.cell - identity(3) * 2.5) < 1e-8)
    assert abs(a.scale - 5.45 * angstrom) < 1e0
    assert len(a.__dict__) == 3

    a = Structure(identity(3) * 2.5, scale=5.45, m=True)
    assert all(abs(a.cell - identity(3) * 2.5) < 1e-8)
    assert abs(a.scale - 5.45 * angstrom) < 1e0
    assert len(a.__dict__) == 4 and getattr(a, 'm', False)


def test_representability():
    import quantities
    import numpy
    dictionary = {Structure.__name__: Structure}
    dictionary.update(numpy.__dict__)
    dictionary.update(quantities.__dict__)

    expected = Structure()
    actual = eval(repr(expected), dictionary)
    assert all(abs(expected.cell - actual.cell) < 1e-8)
    assert abs(expected.scale - actual.scale) < 1e-8
    assert len(expected) == len(actual)

    expected = Structure([1, 2, 0], [3, 4, 5], [6, 7, 8], m=True)
    actual = eval(repr(expected), dictionary)
    assert all(abs(expected.cell - actual.cell) < 1e-8)
    assert abs(expected.scale - actual.scale) < 1e-8
    assert len(expected) == len(actual)
    assert getattr(expected, 'm', False) == actual.m

    expected = Structure([1, 2, 0], [3, 4, 5], [6, 7, 8], m=True) \
        .add_atom(0, 1, 2, "Au", m=5) \
        .add_atom(0, -1, -2, "Pd")
    actual = eval(repr(expected), dictionary)
    assert all(abs(expected.cell - actual.cell) < 1e-8)
    assert abs(expected.scale - actual.scale) < 1e-8
    assert len(expected) == len(actual)
    assert all(abs(expected[0].pos - actual[0].pos) < 1e-8)
    assert getattr(expected[0], 'm', 0) == actual[0].m
    assert expected[0].type == actual[0].type
    assert all(abs(expected[1].pos - actual[1].pos) < 1e-8)
    assert expected[1].type == actual[1].type
    assert getattr(expected, 'm', False) == actual.m


def test_initerror():
    """ Checks initialization throws appropriately. """

    with raises(ValueError):
        Structure("A", 0, 0, 0, 2.5, 0, 0, 0, 2.5)

    with raises(TypeError):
        Structure(0, 0, 0, 2.5, 0, 0, 0, 2.5)

    with raises(TypeError):
        Structure(2.5, 0, 0, 0, 0, 2.5, 0, 0, 0, 2.5)

    with raises(ValueError):
        Structure([2.5, 0, 0, 0], [0, 2.5, 0], [0, 0, 2.5])

    with raises(ValueError):
        Structure([2.5, 0, 0], [0, 2.5], [0, 0, 2.5])

    with raises(ValueError):
        Structure([2.5, 0, 0], [0, 2.5, 0], [0, 0, 'A'])

    with raises(TypeError):
        Structure([2.5, 0, 0], [0, 2.5, 0], [0, 0, 0], cell='a')

    with raises(ValueError):
        Structure(cell='a')


def test_sequence_of_atoms():
    a = Structure(identity(3) * 2.5, scale=5.45, m=True)
    a.add_atom(0, 0, 0, "Au")\
     .add_atom(0.25, 0.5, 0.25, "Au", "Pd", m=True)

    assert len(a) == 2
    assert all(abs(a[0].pos) < 1e-8)
    assert a[0].type == "Au"

    assert all(abs(a[1].pos - (0.25, 0.5, 0.25)) < 1e-8)
    assert a[1].type == ("Au", "Pd")
    assert getattr(a[1], 'm', False) == True

    a.insert(1, 0.1, 0.1, 0.1, 6)
    assert len(a) == 3
    assert all(abs(a[1].pos - 0.1) < 1e-8)
    assert a[1].type == 6

    b = a.pop(1)
    assert all(abs(b.pos - 0.1) < 1e-8)
    assert b.type == 6
    assert len(a) == 2

    a.append(b)
    assert len(a) == 3
    assert all(abs(a[2].pos - 0.1) < 1e-8)
    assert a[2].type == 6

    b = a[0], a[1], a[2]
    a.clear()
    assert len(a) == 0
    a.extend(b)
    assert len(a) == 3 and a[0] is b[0] and a[1] is b[1] and a[2] is b[2]

    a.clear()
    b = Structure(identity(3) * 2.5, scale=5.45, m=True)
    b.add_atom(0, 0, 0, "Au")\
     .add_atom(0.25, 0.5, 0.25, "Au", "Pd", m=True)\
     .add_atom(0.1, 0.1, 0.1, 6, m=True)
    assert len(a) == 0
    a.extend(b)
    assert len(b) == 3
    assert a[0] is b[0]
    assert a[1] is b[1]
    assert a[2] is b[2]
    assert a is not b

    a[2] = Atom(-1, -1, -1, None)
    assert abs(all(a[2].pos + 1) < 1e-8) and a[2].type is None


def test_slicing():
    def create_al():
      types = 'ABCDEFGHIJKLMN'
      result = Structure(identity(3) * 2.5, scale=5.45,
                         m=True), list(range(10))
      for i in range(10):
        result[0].add_atom(Atom(i, i, i, types[i]))
      return result

    def check_al(*args):
      types = 'ABCDEFGHIJKLMN'
      for i, j in zip(*args):
        if not(all(abs(i.pos - j) < 1e-8) and i.type == types[j]):
          return False
        if i.__class__ is not Atom:
          return False
      return True
    # checks getting slices.
    a, l = create_al()
    assert check_al(a[::2], l[::2])

    a, l = create_al()
    assert check_al(a[4::2], l[4::2])
    a, l = create_al()
    assert check_al(a[3:8:3], l[3:8:3])
    a, l = create_al()
    assert check_al(a[3:8:-2], l[3:8:-2])

    # checks slice deletion.
    a, l = create_al()
    del a[::2]
    del l[::2]
    assert check_al(a, l)
    a, l = create_al()
    del a[3:8:2]
    del l[3:8:2]
    assert check_al(a, l)

    # checks settting slices.
    a, l = create_al()
    a[:] = a[::-1]
    l[:] = l[::-1]
    assert check_al(a, l)
    a, l = create_al()
    a[::-1] = a
    l[::-1] = l
    assert check_al(a, l)
    a, l = create_al()
    a[::2] = a[::-1][::2]
    l[::2] = l[::-1][::2]
    assert check_al(a, l)
    a, l = create_al()
    a[4::2] = a[::-1][4::2]
    l[4::2] = l[::-1][4::2]
    assert check_al(a, l)


def test_copy():
  """ Checks structure copy. """
  from numpy import all, abs, array, identity
  from copy import deepcopy
  a = Structure(identity(3) * 2.5, scale=5.45, m=True)\
      .add_atom(Atom(0, 0, 0, "Au"))\
      .add_atom(Atom(0.25, 0.5, 0.25, "Au", "Pd", m=True))\
      .add_atom(Atom(0.1, 0.1, 0.1, 6, m=True))
  b = deepcopy(a)
  assert a is not b
  assert b.__class__ is Structure
  assert all(abs(a.cell - b.cell) < 1e-8)
  assert abs(a.scale - b.scale) < 1e-8
  assert getattr(b, 'm', False) == True
  assert len(b) == 3

  for i, j in zip(a, b):
      assert i is not j
      assert i.__class__ is j.__class__
      assert all(abs(i.pos - j.pos) < 1e-8)
      assert i.type == j.type
      assert getattr(i, 'm', False) == getattr(j, 'm', False)


def test_pickle():
  """ Check pickling. """
  from numpy import all, abs, array, identity
  from pickle import loads, dumps
  a = Structure(identity(3) * 2.5, scale=5.45, m=True)\
      .add_atom(Atom(0, 0, 0, "Au"))\
      .add_atom(Atom(0.25, 0.5, 0.25, "Au", "Pd", m=True))\
      .add_atom(Atom(0.1, 0.1, 0.1, 6, m=True))
  b = loads(dumps(a))
  assert a is not b
  assert b.__class__ is Structure
  assert all(abs(a.cell - b.cell) < 1e-8)
  assert abs(a.scale - b.scale) < 1e-8
  assert getattr(b, 'm', False) == True
  assert len(b) == 3

  for i, j in zip(a, b):
      assert i is not j
      assert i.__class__ is j.__class__
      assert all(abs(i.pos - j.pos) < 1e-8)
      assert i.type == j.type
      assert getattr(i, 'm', False) == getattr(j, 'm', False)
