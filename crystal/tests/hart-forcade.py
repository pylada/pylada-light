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

""" Regression tests for hf stuff. """
def test1():
  from random import randint
  from numpy import all, abs, dot, array
  from pylada.crystal.cppwrappers import HFTransform
  
  unitcell = array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
  supercell = array([[1,0,0],[0,1,0],[0,0,1]])
  a = HFTransform(unitcell, supercell)
  assert all(abs(a.transform-[[1, 1, -1],[0, 2, 0],[0, 0, 2]]) < 1e-8)
  assert all(abs(a.quotient-[1,2,2]) < 1e-8)
  for i in xrange(20):
    vec = dot(supercell, array([randint(-20, 20), randint(-20, 20), randint(-20, 20)], dtype="float64"))
    vec += [0, -0.5, 0.5]
    assert all(abs(a.indices(vec)-[0, 1, 1]) < 1e-8)
    try: a.indices(vec + [0.1, 0.1, 0])
    except ValueError: pass
    else: raise RuntimeError("Should have failed.")

def test2():
  from random import randint
  from numpy import all, abs, dot, array
  from pylada.crystal.cppwrappers import HFTransform, Structure, supercell
  
  unitcell = array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
  lattice = Structure(unitcell).add_atom(0,0,0, "Si")
  supercell = supercell(lattice, dot(lattice.cell, [[3,0,5],[0,0,-1],[-2,1,2]]))

  a = HFTransform(unitcell, supercell)

  assert all(abs(a.transform-[[0, 2, 0],[1, 5, -1],[-2, -4, 0]]) < 1e-8)
  assert all(abs(a.quotient-[1,1,3]) < 1e-8)
  all_indices = set()
  for atom in supercell:
    indices = a.indices(atom.pos)
    index = a.index(atom.pos)
    assert index not in all_indices, (index, all_indices )
    assert all(indices >= 0)
    assert all(indices <= a.quotient )
    assert index == a.flatten_indices(*indices)
    all_indices.add(index)
    for i in xrange(20):
      vec = dot(supercell.cell, array([randint(-20, 20), randint(-20, 20), randint(-20, 20)], dtype="float64"))
      vec += atom.pos
      assert all(abs(a.indices(vec)-indices) < 1e-8)
      try: a.indices(vec + [0.1, 0.1, 0])
      except ValueError: pass
      else: raise RuntimeError("Should have failed.")
      assert index == a.index(vec)
      try: a.index(vec + [0.1, 0.1, 0])
      except ValueError: pass
      else: raise RuntimeError("Should have failed.")
  assert len(all_indices) == len(supercell)

def b5(u=0.25):
  from pylada.crystal.cppwrappers import Structure
  x, y = u, 0.25-u
  structure = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) \
                       .add_atom(5.000000e-01, 5.000000e-01, 5.000000e-01, "A") \
                       .add_atom(5.000000e-01, 2.500000e-01, 2.500000e-01, "A") \
                       .add_atom(2.500000e-01, 5.000000e-01, 2.500000e-01, "A") \
                       .add_atom(2.500000e-01, 2.500000e-01, 5.000000e-01, "A") \
                       .add_atom(8.750000e-01, 8.750000e-01, 8.750000e-01, "B") \
                       .add_atom(1.250000e-01, 1.250000e-01, 1.250000e-01, "B") \
                       .add_atom(     x,     x,     x, "X") \
                       .add_atom(     x,     y,     y, "X") \
                       .add_atom(     y,     x,     y, "X") \
                       .add_atom(     y,     y,     x, "X") \
                       .add_atom(    -x,    -x,    -x, "X") \
                       .add_atom(    -x,    -y,    -y, "X") \
                       .add_atom(    -y,    -x,    -y, "X") \
                       .add_atom(    -y,    -y,    -x, "X") 
  return structure
def test3(u):
  from random import randint
  from numpy import all, abs, dot, array, concatenate
  from pylada.crystal.cppwrappers import HFTransform, supercell
  
  lattice = b5(u)
  supercell = supercell(lattice, dot(lattice.cell, [[2,2,0],[0,2,2],[4,0,4]]))

  a = HFTransform(lattice.cell, supercell)

  assert all(abs(a.transform-[[-1, 1, 1],[1, -1, 1],[5, -3, -1]]) < 1e-8)
  assert all(abs(a.quotient-[2,2,8]) < 1e-8)
  all_indices = set()
  others = set()
  for atom in supercell:
    indices = a.indices(atom.pos - lattice[atom.site].pos)
    index = a.index(atom.pos - lattice[atom.site].pos, atom.site)
    assert index not in all_indices, (index, all_indices )
    assert all(indices >= 0)
    assert all(indices <= a.quotient )
    all_indices.add(index)
    assert str(concatenate((indices, [atom.site]))) not in others
    others.add(str(concatenate((indices, [atom.site]))))
    for i in xrange(20):
      vec = dot(supercell.cell, array([randint(-20, 20), randint(-20, 20), randint(-20, 20)], dtype="float64"))
      vec += atom.pos - lattice[atom.site].pos
      assert all(abs(a.indices(vec)-indices) < 1e-8)
      try: a.indices(vec + [0.1, 0.1, 0])
      except ValueError: pass
      else: raise RuntimeError("Should have failed.")
      assert index == a.index(vec, atom.site)
      try: a.index(vec + [0.1, 0.1, 0])
      except ValueError: pass
      else: raise RuntimeError("Should have failed.")
  assert len(all_indices) == len(supercell)

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  test1()
  test2()
  test3(0.25)
  test3(0.33)
