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
""" Checks atom methods and attributes. """
from nose_parameterized import parameterized
from pylada.crystal.cppwrappers import Atom
class Subclass(Atom):
  def __init__(self, *args, **kwargs):
    super(Subclass, self).__init__(*args, **kwargs)
classes = [(Atom,), (Subclass,)]

@parameterized(classes)
def test_init(Class):
  """ Test atom initialization. """
  from numpy import all, abs, array

  # Try correct initialization. Check for garbage collection.
  a = Class()
  assert a.type is None and all(abs(a.pos) < 1e-8) and len(a.__dict__) == 0
  assert a.type is None and all(abs(a.pos) < 1e-8) and len(a.__dict__) == 0
  a = Class(0.1, 0.1, 0.1, 'Au')
  assert a.type == "Au" and all(abs(a.pos-0.1) < 1e-8) and len(a.__dict__) == 0
  a = Class(0.1, 0.1, 0.1, type=['Au', 'Pd'])
  assert a.type == ["Au", 'Pd'] and all(abs(a.pos-0.1) < 1e-8) and len(a.__dict__) == 0
  a = Class(type='Au', pos=[0.1, 0.1, 0.1])
  assert a.type == "Au" and all(abs(a.pos-0.1) < 1e-8) and len(a.__dict__) == 0
  a = Class(type='Au', pos=[0.1, 0.1, 0.1], m=5)
  assert a.type == "Au" and all(abs(a.pos-0.1) < 1e-8) and len(a.__dict__) == 1 and getattr(a, 'm', 3) == 5
  a = Class(0.1, 0.1, 0.1, 0.1, 0.1)
  assert all(abs(array(a.type) - 0.1) < 1e-8) and all(abs(a.pos-0.1) < 1e-8) and len(a.__dict__) == 0
  l = [None]
  a = Class(0.1, 0.1, 0.1, l)
  assert a.type is l
  assert all(abs(a.pos-0.1) < 1e-8) and len(a.__dict__) == 0
  a.pos[0] = 0.2
  a.pos[1] = 0.3
  a.pos[2] = 0.4
  assert all(abs(a.pos-[0.2, 0.3, 0.4]) < 1e-8)
  
@parameterized(classes)
def test_fail_init(Class):
  """ Test failures during initialization. """
  # Try incorrect initialization
  try: a = Class(0.1, 0.1)
  except TypeError: pass
  else: raise RuntimeError("Should have failed.")
  try: a = Class(0.1, 'Au', 0.1, 0.1)
  except TypeError: pass
  else: raise RuntimeError("Should have failed.")
  try: a = Class(0.1, 0.1, 0.1, 'Au', type='Au')
  except TypeError: pass
  else: raise RuntimeError("Should have failed.")
  try: a = Class(0.1, 0.1, 0.1, pos=[0.1, 0.1, 0.1])
  except TypeError: pass
  else: raise RuntimeError("Should have failed.")

@parameterized(classes)
def test_repr(Class):
  """ Test representability. """
  assert repr(Class(type='Au', pos=[1, 1, 1], m=1)) == "{0.__name__}(1, 1, 1, 'Au', m=1)".format(Class)
  assert str(Class(type='Au', pos=[1, 1, 1], site=1)) == "{0.__name__}(1, 1, 1, 'Au', site=1)".format(Class)

@parameterized(classes)
def test_copy(Class):
  """ Test copy and deep copy. """
  from copy import copy, deepcopy
  b = Class(0,0,0, 'Au', m=0)
  a = copy(b)
  b.type = 'Pd'
  assert a is b
  a = deepcopy(b)
  b.type = 'Pd'
  b.pos += 1
  del b.m 
  assert a.type == "Pd" and all(abs(a.pos-0.0) < 1e-8) and len(a.__dict__) == 1 and getattr(a, 'm', 1) == 0
  assert a.__class__ is Class

@parameterized(classes)
def test_todict(Class):
  """ Test to_dict member. """
  a = Class(0,0,0, 'Au', m=0)
  a = a.to_dict()
  assert a['type'] == "Au" and all(a['pos']== 0) and a['m'] == 0

@parameterized(classes)
def test_pickle(Class):
  """ Test pickling. """
  from pickle import loads, dumps
  a = Class(pos=[0, 1, 2], type="Au", m=6)
  b = loads(dumps(a))
  assert b.__class__ is a.__class__
  assert all(abs(b.pos - [0, 1, 2]) < 1e-12) and b.type == 'Au' \
         and len(b.__dict__) == 1 and b.__dict__.get('m', 0) == 6
  b = loads(dumps((a, a)))
  assert b[0] is b[1]

@parameterized(classes)
def test_resize(Class):
  """ Tests that numpy does not allow resizing. """
  a = Class(pos=[0, 1, 4], type='Au')
  try: a.pos.resize(5)
  except ValueError: pass
  else: raise Exception()
  b = a.pos.reshape((1,3,1))
  b[0,1,0] = 0
  assert abs(a.pos[1]) < 1e-8
