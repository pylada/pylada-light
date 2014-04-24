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

""" Checks crystal method `pylada.crystal.equivalent`. """
def scale(A, B):
  """ Check changes to scale. """
  from pylada.crystal.cppwrappers import equivalent
  B = B.copy()
  A.name = "A"
  B.name = "B"
  print A
  print B
  assert equivalent(A, B)
  B.scale = 3.0;
  assert not equivalent(A, B)
  assert equivalent(A, B, scale=False)
  B.cell *= 0.5;
  for atom in B: atom.pos *= 0.5
  assert not equivalent(A, B)
  assert equivalent(A, B, scale=False)

def motif(A, B):
  """ Check changes in motif. """
  from numpy import dot
  from pylada.crystal.cppwrappers import cell_invariants
  for op in cell_invariants(A)[1:]:
    print "op: ", op
    C = B.copy()
    for atom in C:
      atom.pos = dot(op[:3], atom.pos)
    scale(A, C)

def basis(A, B):
  """ Adds rotation and translation of cartesian basis. """
  from numpy import dot, pi
  from pylada.crystal.cppwrappers import cell_invariants, transform
  from pylada.math import Translation, Rotation
  from random import random
  motif(A, B)
# motif(A, transform(B, Rotation(0.5 * pi, [1,0,0])))
# motif(A, transform(B, Rotation(-pi, [1,0,0])))
# motif(A, transform(B, Rotation(-0.13*pi, [1,0,0])))
# motif(A, transform(B, Translation([0.25, 0.25, 0.25])))
# motif(A, transform(B, Rotation(random()*2*pi, [1, 0, 0]) \
#                       + Translation([random()-0.5,random()-0.5,random()-0.5])))
def decoration(A, B, lattice):
  """ Adds changes to the motif. """
  from numpy import dot
  from numpy.linalg import inv
  from pylada.crystal.cppwrappers import SmithTransform
  from pylada.math import is_integer
  basis(A, B)
  return

  smith = SmithTransform(lattice, A)

  # create map of atoms.
  indices = [-1] * len(A)
  for i, atom in enumerate(A):
    u = smith.index(atom.pos-lattice[atom.site].pos, atom.site)
    indices[u] = i

  # transform A according to all possible atom-atom translations.
  for atom in A:
    if atom.site != A[0].site: continue # only primitive translations.
    trans = atom.pos - A[0].pos
    B = A.copy()
    for index in indices:
      vec = A[index].pos + trans - lattice[A[index].site].pos
      assert is_integer(dot(inv(lattice.cell), vec)) # vector should be on lattice
      B[ indices[smith.index(vec, A[index].site)] ] = A[index]
    basis(A, B)

def test0():
  from pylada.crystal.cppwrappers import Structure

  zb = Structure( 0,0.5,0.5,
                  0.5,0,0.5,
                  0.5,0.5,0 )\
                .add_atom(0,0,0, "Si")\
                .add_atom(0.25,0.25,0.25, "Si", "Ge")
  basis(zb, zb)

def test1():
  from pylada.crystal.cppwrappers import Structure, supercell

  zb = Structure( 0,0.5,0.5,
                  0.5,0,0.5,
                  0.5,0.5,0 )\
                .add_atom(0,0,0, "Si")\
                .add_atom(0.25,0.25,0.25, "Si")
  sc = supercell(zb, [[3, 0, 0], [0, 0.5,-0.5], [0, 0.5, 0.5]])
  sc[0].type = "Ge"
  sc[1].type = "Ge"
  sc[3].type = "Ge"
  decoration(sc, sc, zb)

def test2():
  from random import random
  from pylada.crystal.cppwrappers import Structure, supercell

  zb = Structure( 0,0.5,0.5,
                  0.5,0,0.5,
                  0.5,0.5,0 )\
                .add_atom(0,0,0, "Si")\
                .add_atom(0.25,0.25,0.25, "Si")
  sc = supercell(zb, [[2, 0, 0], [0, 2,0], [0, 0, 2]])
  del sc.lattice
  for i in xrange(1):
    x = random() * 0.5
    for atom in sc:
      atom.type = "Si" if x > random() else "Ge"
    decoration(sc, sc, zb)

if __name__ == "__main__":
  from sys import argv, path 
  from numpy import array
  if len(argv) > 0: path.extend(argv[1:])
  
# test0() # no decoration.
# test1()
  test2()
