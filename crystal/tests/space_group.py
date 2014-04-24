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

""" Checks that space group is correct. """
def test_fcc():
  """ Test fcc space-group and equivalents """
  from numpy import all, abs, dot
  from pylada.crystal.cppwrappers import space_group, Structure, equivalent, transform

  structure = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]], m=True).add_atom(0,0,0,"Si", m=True)
  ops = space_group(structure)
  assert len(ops) == 48
  for op in ops:
    assert op.shape == (4, 3)
    assert all(abs(op[3, :]) < 1e-8) 

    other = transform(structure, op)
    assert all(abs(dot(op[:3], structure.cell)-other.cell) < 1e-8)
    assert getattr(other, 'm', False) 
    for a, atom in zip(structure, other):
      assert all(abs(dot(op[:3], a.pos) + op[3] - atom.pos) < 1e-8)
      assert a.type == atom.type
      assert getattr(atom, 'm', False) 

    assert equivalent(structure, other, cartesian=False)
    assert equivalent(other, structure, cartesian=False)

def test_b5(u):
  """ Test b5 space-group and equivalents """
  from random import random, randint
  from numpy import all, abs, dot, pi
  from numpy.linalg import inv, norm
  from numpy.random import random_sample
  from pylada.crystal.cppwrappers import space_group, Structure, equivalent,     \
                                       transform
  from pylada.math import Rotation
  from pylada.crystal import which_site


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
  ops = space_group(structure)
  assert len(ops) == 48
  invcell = inv(structure.cell)
  for op in ops:
    assert op.shape == (4, 3)

    other = transform(structure, op)
    assert all(abs(dot(op[:3], structure.cell)-other.cell) < 1e-8)
    for a, atom in zip(structure, other):
      assert all(abs(dot(op[:3], a.pos) + op[3] - atom.pos) < 1e-8)
      assert a.type == atom.type
    sites = []
    for i, atom in enumerate(structure):
      pos = dot(op[:3], atom.pos) + op[3]
      j = which_site(pos, structure, invcell)
      assert j != -1
      sites.append(j)
    assert len(set(sites)) == len(structure)

    assert equivalent(structure, other, cartesian=False)
    assert equivalent(other, structure, cartesian=False)

  structure[0], structure[-1] = structure[-1], structure[0]
  ops = space_group(structure)
  assert len(ops) == 48
  for op in ops:
    assert op.shape == (4, 3)

    other = transform(structure, op)
    assert all(abs(dot(op[:3], structure.cell)-other.cell) < 1e-8)
    for a, atom in zip(structure, other):
      assert all(abs(dot(op[:3], a.pos) + op[3] - atom.pos) < 1e-8)
      assert a.type == atom.type
    sites = []
    for i, atom in enumerate(structure):
      pos = dot(op[:3], atom.pos) + op[3]
      j = which_site(pos, structure, invcell)
      assert j != -1, (i, atom, op)
      sites.append(j)
    assert len(set(sites)) == len(structure)

    assert equivalent(structure, other, cartesian=False)
    assert equivalent(other, structure, cartesian=False)

  # try random rotation, translations, atom swap
  structure[0], structure[-1] = structure[-1], structure[0]
  for u in xrange(10):
    axis = random_sample((3,))
    axis /= norm(axis)
    rotation = Rotation( pi*random(), axis)  
    rotation[3] = random_sample((3,))
    other = transform(structure, rotation)
    for u in xrange(10):
      l, m = randint(0, len(structure)-1), randint(0, len(structure)-1)
      a, b = other[l], other[m]
      other[l], other[m] = b, a
    invcell = inv(other.cell)
    ops = space_group(other)
    for z, op in enumerate(ops):
      assert op.shape == (4, 3)
    
      other2 = transform(other, op)
      assert all(abs(dot(op[:3], other.cell)-other2.cell) < 1e-8)
      for a, atom in zip(other, other2):
        assert all(abs(dot(op[:3], a.pos) + op[3] - atom.pos) < 1e-8)
        assert a.type == atom.type
      sites = []
      for i, atom in enumerate(other):
        pos = dot(op[:3], atom.pos) + op[3]
        j = which_site(pos, other, invcell)
        if j == -1:
          print i, z
          print atom 
          print op
          print pos
          print other
          raise Exception()
        sites.append(j)
      assert len(set(sites)) == len(other)

      assert equivalent(other, other2, cartesian=False)
      assert equivalent(other2, other, cartesian=False)

    
def test_zb():
  from numpy import all, abs, dot
  from pylada.crystal import space_group, transform, binary
  from pylada.crystal.cppwrappers import equivalent

  structure = binary.zinc_blende()
  ops = space_group(structure)
  assert len(ops) == 24
  for op in ops:
    assert op.shape == (4, 3)

    other = transform(structure, op)
    assert all(abs(dot(op[:3], structure.cell)-other.cell) < 1e-8)
    for a, atom in zip(structure, other):
      assert all(abs(dot(op[:3], a.pos) + op[3] - atom.pos) < 1e-8)
      assert a.type == atom.type

    assert equivalent(structure, other, cartesian=False)
    assert equivalent(other, structure, cartesian=False)
     
  for atom in structure: atom.type = ['A', 'B']
  ops = space_group(structure)
  assert len(ops) == 48
  for op in ops:
    assert op.shape == (4, 3)

    other = transform(structure, op)
    assert all(abs(dot(op[:3], structure.cell)-other.cell) < 1e-8)
 #  for a, atom in zip(structure, other):
 #    assert all(abs(dot(op[:3], a.pos) + op[3] - atom.pos) < 1e-8)
 #    assert a.type == atom.type

 #  assert equivalent(structure, other, cartesian=False)
 #  assert equivalent(other, structure, cartesian=False)


if __name__ == "__main__":
  test_fcc()
  test_b5(0.25)
  test_b5(0.36)
  test_zb()
