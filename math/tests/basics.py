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
  from pylada.math import is_integer
  from pylada.error import TypeError

  assert is_integer(array([0, 1, 0, 2]))
  assert is_integer(array([0L, 1, 0, 2]))
  assert is_integer(array([0, 1.0, 0.0, 2.0]))
  assert not is_integer(array([0, 1.1, 0.0, 2.0]))
  try: is_integer([5, 6, 7])
  except TypeError: pass
  else: raise Exception()

def test_floorint():
  from numpy import array, all 
  from pylada.math import floor_int
  from pylada.error import TypeError

  assert floor_int(array([0.1, -0.1, -0.5, 0.5, -0.55, 0.55])).dtype == 'int64'
  assert all( floor_int(array([0.1, -0.1, -0.5, 0.5, -0.55, 0.55, 1.999, -1.99]))
               == [0, -1, -1, 0, -1, 0, 1, -2] )
  assert all( floor_int(array([[0.1, -0.1, -0.5, 0.5], [-0.55, 0.55, 1.999, -1.99]]))
               == [[0, -1, -1, 0], [-1, 0, 1, -2]] )
  
  try: floor_int([5, 6, 7])
  except TypeError: pass
  else: raise Exception()

def test_rotation():
  from numpy import all, abs, pi
  from pylada.math import Rotation
  from pylada.error import TypeError

  assert all(abs(Rotation(0, [1, 0, 0]) - [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]) < 1e-8)
  assert all(abs(Rotation(pi, [1, 0, 0]) - [[1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]]) < 1e-8)
  assert all(abs(Rotation(pi/2, [1, 0, 0]) - [[1, 0, 0], [0, 0, -1], [0, 1, 0], [0, 0, 0]]) < 1e-8)

  try: Rotation(pi/2, [0, 0])
  except TypeError: pass
  else: raise Exception()
  try: Rotation(pi/2, [0, 0, 1, 0])
  except TypeError: pass
  else: raise Exception()

def test_translation():
  from numpy import all, abs, identity, pi
  from pylada.math import Translation
  from pylada.error import TypeError

  assert all(abs(Translation([2, 2, 2])[:3] - identity(3)) < 1e-8)
  assert all(abs(Translation([2, 2, 2])[3] - 2) < 1e-8)
  assert all(abs(Translation([pi, pi/2., 2])[:3] - identity(3)) < 1e-8)
  assert all(abs(Translation([pi, pi/2., 2])[3] - [pi, pi/2., 2]) < 1e-8)

  try: Translation([0, 0])
  except TypeError: pass
  else: raise Exception()
  try: Translation([0, 0, 1, 0])
  except TypeError: pass
  else: raise Exception()

if __name__ == '__main__': 
  test_isinteger()
  test_floorint()
  test_rotation()
  test_translation()
