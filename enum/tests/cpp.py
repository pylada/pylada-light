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

def test_isinteger():
  from numpy import array
  from pylada.enum.cppwrappers import is_integer
  assert not is_integer(array([0.5, 0.5]))
  assert not is_integer(array([1.0, 0.5]))
  assert not is_integer(array(0.1))
  assert is_integer(array([1.0, -1.0])) 
  assert is_integer(array(-5))
  assert is_integer(array([ [1.0, -1.0], [5.0, 1111.0] ]))

def test_ndimiterator():
  from numpy import all 
  from itertools import product
  from pylada.enum.cppwrappers import NDimIterator
  from pylada.error import TypeError, ValueError as LadaValueError

  i = 0
  for u in NDimIterator(3,3,4): i += 1; continue
  assert i == 3*3*4
  assert all(u == 1) # went round
  for u in NDimIterator(3,3,4): continue
  assert all(u == 1) # went round
  iterator = NDimIterator(5)
  for v in xrange(1, 6):
    u = iterator.next()
    assert  u == v
  assert u == 5 # didn't go round cos xrange finished first.
  iterator = NDimIterator(2, 3)
  for v in product(xrange(1, 3), xrange(1, 4)):
    u = iterator.next()
    assert all(u == v)
  assert all(u == [2, 3])
  iterator = NDimIterator(2, 3)
  a = iterator.next()
  assert all(a == [1, 1])
  b = iterator.next()
  assert all(b == [1, 2])
  assert all(a == b)
  assert a is b
  for u, v in zip( NDimIterator(5, 5, 5), 
                   product(xrange(1,6), repeat=3) ): 
    assert all(u == 1)

  iterator = NDimIterator(5, 5, 5)
  u = iterator.next()
  try: u[1] = 2
  except ValueError: pass   # numpy >= 1.7
  except RuntimeError: pass # numpy < 1.7
  else: raise Exception()

  try: NDimIterator(5, 0, 1)
  except LadaValueError: pass
  else: raise Exception()

  try: NDimIterator(5, 'a', 1)
  except TypeError: pass
  else: raise Exception()

def test_lexcompare():
  from pylada.enum.cppwrappers import _lexcompare, NDimIterator
  r = [u.copy() for u in NDimIterator(3, 4, 5)]
  for i in xrange(len(r)):
    assert _lexcompare(r[i], r[i]) == 0
    assert all(_lexcompare(r[i], r[j]) == -1 for j in xrange(i+1, len(r)))
    assert all(_lexcompare(r[i], r[j]) == 1 for j in xrange(1, i))

def test_fciterator():
  from pylada.enum.cppwrappers import FCIterator
  result = [False, False, False, True, True], \
           [False, False, True, False, True], \
           [False, True, False, False, True], \
           [True, False, False, False, True], \
           [False, False, True, True, False], \
           [False, True, False, True, False], \
           [True, False, False, True, False], \
           [False, True, True, False, False], \
           [True, False, True, False, False], \
           [True, True, False, False, False]
  iterator = FCIterator(5, 2)
  for i, u in enumerate(iterator):
    assert all(u == result[i])
  iterator.reset()
  reit = False
  for i, u in enumerate(iterator):
    assert all(u == result[i])
    reit = True
  assert reit

def test_manipulations():
  """ Test manipulation iterator. """
  from numpy import array, arange, all
  from itertools import permutations
  from pylada.enum.cppwrappers import Manipulations
  from pylada.error import TypeError

  x = arange(5, dtype='int16')
  perms = array([u for u in permutations(x)])
  manips = Manipulations(perms)
  for i, u in enumerate(manips(x)): 
    assert all(u == perms[i])
    assert all(x == arange(5))
  for i, u in enumerate(manips(x)): 
    assert all(u == perms[i])
    assert all(x == arange(5))

if __name__ == '__main__':
  test_isinteger()
  test_ndimiterator()
  test_lexcompare()
  test_fciterator()
  test_manipulations()
