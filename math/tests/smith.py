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

def test_smith():
  from numpy import dot, all
  from numpy.random import randint
  from numpy.linalg import det
  from pylada.math import smith_normal_form
  from pylada.error import input
  
  for i in xrange(50):
    cell = randint(-5, 5, size=(3,3))
    while abs(det(cell)) < 1e-2: cell = randint(-5, 5, size=(3,3))
    s, l, r = smith_normal_form(cell)
    assert all(dot(dot(l, cell), r) == s)

  try: smith_normal_form([[0, 0, 0], [1, 2, 0], [3, 4, 5]])
  except input: pass
  else: raise Exception()

def test_capi():
  from _gruber import testme
  assert testme()

if __name__ == '__main__':
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  test_capi()
  test_smith()
