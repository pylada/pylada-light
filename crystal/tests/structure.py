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
def test_init(Class):
  """ Test structure initialization. """
  import gc
  from sys import getrefcount
  from numpy import all, abs, array, identity
  from quantities import angstrom, nanometer

  a = Class()
  assert all(abs(a.cell - identity(3)) < 1e-8) and abs(a.scale - 1e0 * angstrom) < 1e0\
         and len(a.__dict__) == 0

  a = Class(identity(3)*2.5, scale=5.45)
  assert all(abs(a.cell - identity(3)*2.5) < 1e-8) and abs(a.scale - 5.45 * angstrom) < 1e0\
         and len(a.__dict__) == 0

  a = Class(identity(3)*2.5, scale=0.545*nanometer)
  assert all(abs(a.cell - identity(3)*2.5) < 1e-8) and abs(a.scale - 5.45 * angstrom) < 1e0\
         and len(a.__dict__) == 0


  a = Class(2.5, 0, 0, 0, 2.5, 0, 0, 0, 2.5, scale=5.45)
  assert all(abs(a.cell - identity(3)*2.5) < 1e-8) and abs(a.scale - 5.45 * angstrom) < 1e0\
         and len(a.__dict__) == 0
  
  a = Class([2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5], scale=5.45)
  assert all(abs(a.cell - identity(3)*2.5) < 1e-8) and abs(a.scale - 5.45 * angstrom) < 1e0\
         and len(a.__dict__) == 0

  a = Class(cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]], scale=5.45)
  assert all(abs(a.cell - identity(3)*2.5) < 1e-8) and abs(a.scale - 5.45 * angstrom) < 1e0\
         and len(a.__dict__) == 0

  a = Class(identity(3)*2.5, scale=5.45, m=True)
  assert all(abs(a.cell - identity(3)*2.5) < 1e-8) and abs(a.scale - 5.45 * angstrom) < 1e0\
         and len(a.__dict__) == 1 and getattr(a, 'm', False)
  assert all(abs(eval(repr(a), {'Structure': Structure}).cell - a.cell) < 1e-8)
  assert abs(eval(repr(a), {'Structure': Structure}).scale - a.scale) < 1e-8
  assert getattr(eval(repr(a), {'Structure': Structure}), 'm', False) 
  refcnt = getrefcount(a)
  a.add_atom(0,0,0, "Au")\
   .add_atom(0.25, 0.5, 0.25, "Au", "Pd", m=5)
  gc.collect() 
  # makes sure add_atom did not increase refcount innapropriately
  assert getrefcount(a) == refcnt 
  assert all(abs(eval(repr(a), {'Structure': Structure}).cell - a.cell) < 1e-8)
  assert abs(eval(repr(a), {'Structure': Structure}).scale - a.scale) < 1e-8
  assert getattr(eval(repr(a), {'Structure': Structure}), 'm', False) 
  assert all(abs(eval(repr(a), {'Structure': Structure})[0].pos - a[0].pos) < 1e-8)
  assert eval(repr(a), {'Structure': Structure})[0].type == a[0].type
  assert all(abs(eval(repr(a), {'Structure': Structure})[1].pos - a[1].pos) < 1e-8)
  assert eval(repr(a), {'Structure': Structure})[1].type == a[1].type
  assert getattr(eval(repr(a), {'Structure': Structure})[1], 'm', 6) == 5
  # make sure that add_atom did not increase a's ref count innapropriately.
  a.cell[0,0] = 1e0
  a.cell[1,:] = 1e0
  assert all(abs(a.cell - [[1, 0, 0], [1, 1, 1], [0, 0, 2.5]]) < 1e-8)
  assert all(abs(eval(repr(a), {'Structure': Structure}).cell - a.cell) < 1e-8)
  assert abs(eval(repr(a), {'Structure': Structure}).scale - a.scale) < 1e-8
  assert getattr(eval(repr(a), {'Structure': Structure}), 'm', False) 
  assert all(abs(eval(repr(a), {'Structure': Structure})[0].pos - a[0].pos) < 1e-8)
  assert eval(repr(a), {'Structure': Structure})[0].type == a[0].type
  assert all(abs(eval(repr(a), {'Structure': Structure})[1].pos - a[1].pos) < 1e-8)
  assert eval(repr(a), {'Structure': Structure})[1].type == a[1].type
  assert getattr(eval(repr(a), {'Structure': Structure})[1], 'm', 6) == 5

  a.scale = 0.5 * nanometer
  a.scale += 0.3 * a.scale.units

def test_initerror(Class, AtomClass):
  """ Checks initialization throws appropriately. """
  from numpy import identity
  from pylada.error import TypeError

  try: a = Class(identity(4)*2.5, scale=5.45)
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class("A", 0, 0, 0, 2.5, 0, 0, 0, 2.5)
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class(0, 0, 0, 2.5, 0, 0, 0, 2.5)
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class(2.5, 0, 0, 0, 0, 2.5, 0, 0, 0, 2.5)
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class([2.5, 0, 0, 0], [0, 2.5, 0], [0, 0, 2.5])
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class([2.5, 0, 0], [0, 2.5], [0, 0, 2.5])
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class([2.5, 0, 0], [0, 2.5, 0], [0, 0, 'A'])
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class([2.5, 0, 0], [0, 2.5, 0], [0, 0, 0], cell='a')
  except TypeError: pass
  else: raise Exception("Should have thrown.")
  try: a = Class(cell='a')
  except TypeError: pass
  else: raise Exception("Should have thrown.")

def test_sequence(Class, AtomClass):
  """ Test sequence methods. """
  from numpy import all, abs, array, identity

  a = Class(identity(3)*2.5, scale=5.45, m=True)
  a.add_atom(AtomClass(0,0,0, "Au"))\
   .add_atom(AtomClass(0.25, 0.5, 0.25, "Au", "Pd", m=True))
  assert len(a) == 2
  assert all(abs(a[0].pos) < 1e-8) and a[0].type == "Au" and len(a[0].__dict__) == 0
  assert all(abs(a[1].pos - (0.25, 0.5, 0.25)) < 1e-8) and a[1].type == ["Au", "Pd"]\
         and len(a[1].__dict__) == 1 and getattr(a[1], 'm', False) == True

  a.insert(1, AtomClass(0.1,0.1,0.1, 6))
  assert len(a) == 3
  assert all(abs(a[1].pos - 0.1) < 1e-8) and a[1].type == 6 and len(a[1].__dict__) == 0

  b = a.pop(1)
  assert all(abs(b.pos - 0.1) < 1e-8) and b.type == 6 and len(b.__dict__) == 0
  assert b.__class__ is AtomClass
  assert len(a) == 2

  a.append(b) 
  assert len(a) == 3
  assert all(abs(a[2].pos - 0.1) < 1e-8) and a[2].type == 6 and len(a[2].__dict__) == 0
  
  b = a[0], a[1], a[2]
  a.clear()
  assert len(a) == 0
  a.extend(b)
  assert len(a) == 3 and a[0] is b[0] and a[1] is b[1] and a[2] is b[2]
  a.clear()
  b = Class(identity(3)*2.5, scale=5.45, m=True)
  b.add_atom(AtomClass(0,0,0, "Au"))\
   .add_atom(AtomClass(0.25, 0.5, 0.25, "Au", "Pd", m=True))\
   .add_atom(AtomClass(0.1, 0.1, 0.1, 6, m=True))
  assert len(a) == 0
  a.extend(b)
  assert len(b) == 3 and a[0] is b[0] and a[1] is b[1] and a[2] is b[2] and a is not b
  a[2] = AtomClass(-1,-1,-1, None)
  assert abs(all(a[2].pos+1) < 1e-8) and a[2].type is None 

  def create_al():
    types = 'ABCDEFGHIJKLMN' 
    result = Class(identity(3)*2.5, scale=5.45, m=True), list(range(10))
    for i in range(10): 
      result[0].add_atom(AtomClass(i, i, i, types[i]))
    return result
  def check_al(*args):
    types = 'ABCDEFGHIJKLMN' 
    for i, j in zip(*args): 
      if not( all(abs(i.pos - j) < 1e-8) and i.type == types[j] ): return False
      if i.__class__ is not AtomClass: return False
    return True
  # checks getting slices.
  a, l = create_al(); a = a[::2];      l = l[::2];      assert check_al(a, l)
  a, l = create_al(); a = a[4::2];     l = l[4::2];     assert check_al(a, l)
  a, l = create_al(); a = a[3:8:2];    l = l[3:8:2];    assert check_al(a, l)
  a, l = create_al(); a = a[3:8:3];    l = l[3:8:3];    assert check_al(a, l)
  a, l = create_al(); a = a[3:8:6];    l = l[3:8:6];    assert check_al(a, l)
  a, l = create_al(); a = a[3:8:-2];   l = l[3:8:-2];   assert check_al(a, l)
  a, l = create_al(); a = a[5:3];      l = l[5:3];      assert check_al(a, l)
  a, l = create_al(); a = a[5:3];      l = l[5:3];      assert check_al(a, l)
  a, l = create_al(); a = a[::-1];     l = l[::-1];     assert check_al(a, l)
  a, l = create_al(); a = a[-5::2];    l = l[-5::2];    assert check_al(a, l)
  a, l = create_al(); a = a[-5::-2];   l = l[-5::-2];   assert check_al(a, l)
  a, l = create_al(); a = a[-5:-2:-2]; l = l[-5:-2:-2]; assert check_al(a, l)
  # checks slice deletion.
  a, l = create_al(); del a[::2];      del l[::2];      assert check_al(a, l)
  a, l = create_al(); del a[4::2];     del l[4::2];     assert check_al(a, l)
  a, l = create_al(); del a[3:8:2];    del l[3:8:2];    assert check_al(a, l)
  a, l = create_al(); del a[3:8:3];    del l[3:8:3];    assert check_al(a, l)
  a, l = create_al(); del a[3:8:6];    del l[3:8:6];    assert check_al(a, l)
  a, l = create_al(); del a[3:8:-2];   del l[3:8:-2];   assert check_al(a, l)
  a, l = create_al(); del a[5:3];      del l[5:3];      assert check_al(a, l)
  a, l = create_al(); del a[5:3];      del l[5:3];      assert check_al(a, l)
  a, l = create_al(); del a[-5::2];    del l[-5::2];    assert check_al(a, l)
  a, l = create_al(); del a[-5::-2];   del l[-5::-2];   assert check_al(a, l)
  a, l = create_al(); del a[-5:-2:-2]; del l[-5:-2:-2]; assert check_al(a, l)
  # checks settting slices.
  a, l = create_al(); a[:]        = a[::-1];           l[:]        = l[::-1];           assert check_al(a, l)
  a, l = create_al(); a[::-1]     = a;                 l[::-1]     = l;                 assert check_al(a, l)
  a, l = create_al(); a[::2]      = a[::-1][::2];      l[::2]      = l[::-1][::2];      assert check_al(a, l)
  a, l = create_al(); a[4::2]     = a[::-1][4::2];     l[4::2]     = l[::-1][4::2];     assert check_al(a, l)
  a, l = create_al(); a[3:8:2]    = a[::-1][3:8:2];    l[3:8:2]    = l[::-1][3:8:2];    assert check_al(a, l)
  a, l = create_al(); a[3:8:3]    = a[::-1][3:8:3];    l[3:8:3]    = l[::-1][3:8:3];    assert check_al(a, l)
  a, l = create_al(); a[3:8:6]    = a[::-1][3:8:6];    l[3:8:6]    = l[::-1][3:8:6];    assert check_al(a, l)
  a, l = create_al(); a[3:8:-2]   = a[::-1][3:8:-2];   l[3:8:-2]   = l[::-1][3:8:-2];   assert check_al(a, l)
  a, l = create_al(); a[5:3]      = a[::-1][5:3];      l[5:3]      = l[::-1][5:3];      assert check_al(a, l)
  a, l = create_al(); a[5:3]      = a[::-1][5:3];      l[5:3]      = l[::-1][5:3];      assert check_al(a, l)
  a, l = create_al(); a[-5::2]    = a[::-1][-5::2];    l[-5::2]    = l[::-1][-5::2];    assert check_al(a, l)
  a, l = create_al(); a[-5::-2]   = a[::-1][-5::-2];   l[-5::-2]   = l[::-1][-5::-2];   assert check_al(a, l)
  a, l = create_al(); a[-5:-2:-2] = a[::-1][-5:-2:-2]; l[-5:-2:-2] = l[::-1][-5:-2:-2]; assert check_al(a, l)

  a, l = create_al()
  try: a[2:5] = a.copy()[2:6]
  except: pass
  else: raise RuntimeError('shoulda failed')
  try: a[2:5] = a.copy()[2:3]
  except: pass
  else: raise RuntimeError('shoulda failed')

def test_copy(Class, AtomClass):
  """ Checks structure copy. """
  from numpy import all, abs, array, identity
  from copy import copy as shallow, deepcopy
  a = Class(identity(3)*2.5, scale=5.45, m=True)\
        .add_atom(AtomClass(0,0,0, "Au"))\
        .add_atom(AtomClass(0.25, 0.5, 0.25, "Au", "Pd", m=True))\
        .add_atom(AtomClass(0.1, 0.1, 0.1, 6, m=True))
  assert a is shallow(a)
  b = deepcopy(a)
  assert a is not b
  assert b.__class__ is Class
  assert all(abs(a.cell-b.cell) < 1e-8) and abs(a.scale-b.scale) < 1e-8 \
         and a.__dict__ is not b.__dict__ and len(b.__dict__) == 1 \
         and getattr(b, 'm', False) == True \
         and len(b) == 3 
  for i, j in zip(a, b):
    assert i is not j
    assert i.__class__ is j.__class__
    assert all(abs(i.pos - j.pos) < 1e-8) and i.type == j.type
    if len(i.__dict__) == 1: 
      assert len(j.__dict__) == 1 and getattr(j, 'm', False) == True

def test_pickle(Class, AtomClass):
  """ Check pickling. """
  from numpy import all, abs, array, identity
  from pickle import loads, dumps
  a = Class(identity(3)*2.5, scale=5.45, m=True)\
        .add_atom(AtomClass(0,0,0, "Au"))\
        .add_atom(AtomClass(0.25, 0.5, 0.25, "Au", "Pd", m=True))\
        .add_atom(AtomClass(0.1, 0.1, 0.1, 6, m=True))
  b = loads(dumps(a))
  assert a is not b
  assert b.__class__ is Class
  assert all(abs(a.cell-b.cell) < 1e-8) and abs(a.scale-b.scale) < 1e-8 \
         and a.__dict__ is not b.__dict__ and len(b.__dict__) == 1 \
         and getattr(b, 'm', False) == True \
         and len(b) == 3 
  for i, j in zip(a, b):
    assert i is not j
    assert i.__class__ is j.__class__
    assert all(abs(i.pos - j.pos) < 1e-8) and i.type == j.type
    if len(i.__dict__) == 1: 
      assert len(j.__dict__) == 1 and getattr(j, 'm', False) == True
  

def test_iterator(Class, AtomClass):
  """ Test structure iteration. """
  from numpy import all, abs, array, identity
  
  types = 'ABCDEFGHIJKLMN' 
  def create_al():
    result = Class(identity(3)*2.5, scale=5.45, m=True), list(range(10))
    for i in range(10): 
      result[0].add_atom(AtomClass(i, i, i, types[i]))
    return result
  a, l = create_al()
  for i, j in zip(a, l): 
    assert all(abs(i.pos - j) < 1e-8) and i.type == types[j]
    assert i.__class__ is AtomClass


if __name__ == "__main__":
  from pylada.crystal.cppwrappers import Structure, Atom
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  # tries to run test with normal class.
  test_init(Structure) 
  test_initerror(Structure, Atom)
  test_iterator(Structure, Atom) 
  test_sequence(Structure, Atom) 
  test_copy(Structure, Atom)
  test_pickle(Structure, Atom)


  # tries to run test with other class. 
  # check passage through init.
  check_passage = [False, False]
  class StructureSubclass(Structure):
    def __init__(self, *args, **kwargs):
      global check_passage
      check_passage[0] = True
      super(StructureSubclass, self).__init__(*args, **kwargs)
  class AtomSubclass(Atom):
    def __init__(self, *args, **kwargs):
      global check_passage
      check_passage[1] = True
      super(AtomSubclass, self).__init__(*args, **kwargs)

  test_initerror(StructureSubclass, AtomSubclass)
  test_iterator(StructureSubclass, AtomSubclass) 
  test_sequence(StructureSubclass, AtomSubclass) 
  test_copy(StructureSubclass, AtomSubclass)
  test_pickle(StructureSubclass, AtomSubclass)
  assert check_passage[0] and check_passage[1]
