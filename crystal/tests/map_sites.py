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

""" Checks that map_sites is correct. """
def test_b5(u):
  """ Test b5 space-group and equivalents """
  from random import randint, random
  from numpy import array
  from numpy.linalg import det
  from pylada.crystal.cppwrappers import Structure, map_sites, supercell

  x, y = u, 0.25-u
  lattice = Structure([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]) \
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
  for i in xrange(5):
    while True:
      cell = [ [randint(-2, 3) for j in xrange(3)] for k in xrange(3)]
      if det(cell) != 0: break
    structure0 = supercell(lattice, cell)
    structure1 = structure0.copy()

    for atom in structure1: del atom.site
    assert map_sites(lattice, structure1)
    for a, b in zip(structure0, structure1): 
      assert a.site == b.site

    for atom in structure1:
      del atom.site
      atom.pos += array([random() * 1e-3, random() * 1e-3, random() * 1e-3])
    assert map_sites(lattice, structure1, tolerance=1e-2)
    for a, b in zip(structure0, structure1): 
      assert a.site == b.site

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  test_b5(0.25)
  test_b5(0.36)
