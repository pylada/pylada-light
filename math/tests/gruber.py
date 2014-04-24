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

def test_gruber():
  from numpy import dot
  from numpy.linalg import inv
  from pylada.math import gruber, is_integer
  from pylada.error import internal, input
  
  cell = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
  lim = 5
   
  for a00 in [-1, 1]: 
    for a10 in xrange(-lim, lim+1): 
      for a11 in [-1, 1]:
        for a20 in xrange(-lim, lim+1): 
          for a21 in xrange(-lim, lim+1): 
            for a22 in [-1, 1]:
              a = [[a00, 0, 0], [a10, a11, 0], [a20, a21, a22]]
              g = gruber(dot(cell, a))
              assert is_integer(dot(inv(cell), g))
              assert is_integer(dot(inv(g), cell))

  try: gruber([[0, 0, 0], [1, 2, 0], [4, 5, 6]])
  except input: pass
  else: raise Exception()

  try: gruber([[1, 0, 0], [1, 1, 0], [4, 5, 1]], itermax=2)
  except internal: pass
  else: raise Exception()

def test_capi():
  from _gruber import testme
  assert testme()

if __name__ == '__main__':
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  test_capi()
  test_gruber()
