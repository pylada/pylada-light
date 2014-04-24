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

def b5(u=0.25):
  """ Test b5 space-group and equivalents """
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

def indices(invcell, pos, n):
  from numpy import cast, dot, array
  from pylada.math import floor_int
  int_fractional = cast["int64"](floor_int(dot(invcell, pos)))
  int_fractional = array([u + (ni if u < 0 else (-ni if u >= ni else 0)) for u, ni in zip(int_fractional, n)])
  neg = int_fractional % n
  return array([u+ni if u < 0 else u for u, ni in zip(neg, n)]) 

def check(structure):
  from numpy import multiply, cast, any
  from numpy.linalg import inv
  from pylada.crystal.cppwrappers import periodic_dnc
  from pylada.math import gruber

  mesh, boxes = periodic_dnc(structure, nperbox=30, overlap=0.125, return_mesh = True)
  invcell = gruber(structure.cell)
  invcell[:, 0] /= float(mesh[0])
  invcell[:, 1] /= float(mesh[1])
  invcell[:, 2] /= float(mesh[2])
  invcell = inv(invcell)
  for box in boxes:
    assert any(u[2] for u in box)
    for atom, trans, inbox in box:
      if inbox: break
    index = indices(invcell, trans+atom.pos, mesh) 
    for atom, trans, inbox in box:
      pi = indices(invcell, trans+atom.pos, mesh) 
      if inbox: assert all(abs(pi - index) < 1e-8), (pi, index)
      else: 
        assert any(    abs(u-v) == 1 \
                    or (w>1 and abs(u-v) == w-1) \
                    or  w == 1 for u, v, w in zip(pi, index, mesh) )

def newstructure(i=10):
  from numpy import zeros
  from numpy.linalg import det
  from random import randint
  from pylada.crystal.cppwrappers import supercell
  
  lattice = b5()
  cell = zeros((3,3))
  while det(cell) == 0:
    cell[:] = [[randint(-i, i+1) for j in xrange(3)] for k in xrange(3)]
  if det(cell) < 0: cell[:, 0], cell[:, 1] = cell[:, 1].copy(), cell[:, 0].copy()
  return supercell(lattice, cell)



def mayavi(structure, N=10):
  """ import this module and run mayavi to see the divide and conquer boxes. 
  
      Enjoy and play around.
  """
  from enthought.mayavi.mlab import points3d
  from pylada.crystal.cppwrappers import periodic_dnc

  mesh, boxes = periodic_dnc(structure, nperbox=30, overlap=0.25, return_mesh = True)

  x, y, z, s = [], [], [], []
  for i, box in enumerate(boxes):
    if i == N: continue
    x.extend([(atom.pos[0] + trans[0]) for atom, trans, inbox in box if inbox])
    y.extend([(atom.pos[1] + trans[1]) for atom, trans, inbox in box if inbox])
    z.extend([(atom.pos[2] + trans[2]) for atom, trans, inbox in box if inbox])
    s.extend([0.5 for atom, trans, inbox in box if inbox])
  points3d(x,y,z,s, scale_factor=0.1, colormap="copper")

  x, y, z, s = [], [], [], []
  for i, box in enumerate(boxes):
    if i < N: continue
    x.extend([(atom.pos[0] + trans[0]) for atom, trans, inbox in box if inbox])
    y.extend([(atom.pos[1] + trans[1]) for atom, trans, inbox in box if inbox])
    z.extend([(atom.pos[2] + trans[2]) for atom, trans, inbox in box if inbox])
    s.extend([float(i+1) + (0. if inbox else 0.4) for atom, trans, inbox in box if inbox])
    break
  points3d(x,y,z,s, scale_factor=0.004, colormap="autumn")
  x, y, z, s = [], [], [], []
  for i, box in enumerate(boxes):
    if i < N: continue
    x.extend([(atom.pos[0] + trans[0]) for atom, trans, inbox in box if not inbox])
    y.extend([(atom.pos[1] + trans[1]) for atom, trans, inbox in box if not inbox])
    z.extend([(atom.pos[2] + trans[2]) for atom, trans, inbox in box if not inbox])
    s.extend([float(i+2) + (0. if inbox else 0.4) for atom, trans, inbox in box if not inbox])
    break
  points3d(x,y,z,s, scale_factor=0.01, opacity=0.3)


if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])

  from random import randint
  from numpy import zeros
  from numpy.linalg import det
  from pylada.crystal.cppwrappers import supercell
  
  lattice = b5()
  check(lattice)

  for i in xrange(10): 
    cell = zeros((3,3))
    while det(cell) == 0:
      cell[:] = [[randint(-10, 11) for j in xrange(3)] for k in xrange(3)]
    if det(cell) < 0: cell[:, 0], cell[:, 1] = cell[:, 1].copy(), cell[:, 0].copy()
    structure = supercell(lattice, cell)
    check(structure)


