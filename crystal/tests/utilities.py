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

""" Test some utilities from crystal extension module. """
def test_periodic():
  """ Test periodic images. """
  from pylada.crystal.cppwrappers import are_periodic_images
  from numpy import array, dot
  from numpy.linalg import inv
  from random import randint, random
  cell = array([ [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0] ])
  invcell = inv(cell)
  for i in range(10):
    vec = dot(cell, array([random(), random(), random()]))
    for j in range(10):
      vec2 = vec + dot(cell, array([randint(-10, 11), randint(-10, 11), randint(-10, 11)]))
      assert are_periodic_images(vec, vec2, invcell)
      vec3 = vec2 + dot(cell, array([random()+0.0001, random(), random()]))
      assert not are_periodic_images(vec, vec3, invcell)

def test_into_cell():
  """ Test that vector is folded to fractional coordinates >= 0 and < 1. """
  from pylada.crystal.cppwrappers import are_periodic_images, into_cell
  from numpy import array, dot, all
  from numpy.linalg import inv
  from random import uniform
  cell = array([ [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0] ])
  invcell = inv(cell)
  for i in range(100):
    vec = dot(cell, array([uniform(-10, 10), uniform(-10, 10), uniform(-10, 10)]))
    vec2 = into_cell(vec, cell, invcell)
    assert are_periodic_images(vec, vec2, invcell)
    assert all(dot(invcell, vec2) >= 0e0) and all(dot(invcell, vec2) < 1e0)
  
def test_zero_centered():
  """ Test that vector is folded to fractional coordinates >= -0.5 and < 0.5. """
  from pylada.crystal.cppwrappers import are_periodic_images, zero_centered
  from numpy import array, dot, all
  from numpy.linalg import inv
  from random import uniform
  cell = array([ [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0] ])
  invcell = inv(cell)
  for i in range(100):
    vec = dot(cell, array([uniform(-10, 10), uniform(-10, 10), uniform(-10, 10)]))
    vec2 = zero_centered(vec, cell, invcell)
    assert are_periodic_images(vec, vec2, invcell)
    assert all(dot(invcell, vec2) >= -0.5) and all(dot(invcell, vec2) < 0.5)

def test_into_voronoi():
  """ Test that vector is folded into first Brillouin zone. """
  from pylada.crystal.cppwrappers import are_periodic_images, into_voronoi
  from numpy import array, dot
  from numpy.linalg import inv
  from random import uniform
  cell = array([ [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0] ])
  invcell = inv(cell)
  for i in range(100):
    vec = dot(cell, array([uniform(-10, 10), uniform(-10, 10), uniform(-10, 10)]))
    vec2 = into_voronoi(vec, cell, invcell)
    assert are_periodic_images(vec, vec2, invcell)
    n = dot(vec2, vec2)
    assert dot(vec, vec) >= n
    for j in range(-3, 4):
      for k in range(-3, 4):
        for l in range(-3, 4):
          o = vec2 + dot(cell, [j, k, l])
          assert dot(o, o) >= n

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])

  test_periodic()
  test_into_cell()
  test_zero_centered()
  test_into_voronoi()
