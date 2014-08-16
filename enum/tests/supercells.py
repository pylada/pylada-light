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

def test_supercells():
  from pylada.crystal.bravais import fcc
  from pylada.enum import supercells

  # First, a lattice without symmetry
  lattice = fcc()
  lattice.cell = [[1.0, 0.2, 0], [0, 0.9, -0.1], [0, 0, 0.8]]
  scs = supercells(lattice, range(17))
  results = [ 1, 7, 13, 35, 31, 91, 57, 155, 130, 217, 133, 455, 183, 399,
              403, 651 ]
  for r, s in zip(results, scs.itervalues()):
    assert r == len(s)
  # Then, an fcc lattice.
  lattice = fcc()
  scs = supercells(lattice, range(11))
  results = [1, 2, 3, 7, 5, 10, 7, 20, 14, 18]
  for r, s in zip(results, scs.itervalues()):
    assert r, len(s)
  
def test_hfgroups():
  from pylada.crystal.bravais import fcc
  from pylada.enum import hf_groups

  # First, a lattice without symmetry
  lattice = fcc()
  lattice.cell = [[1.0, 0.2, 0], [0, 0.9, -0.1], [0, 0, 0.8]]
  a = [1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 1, 1, 4]
  b = [ 1, 7, 13, 35, 31, 91, 57, 155, 130, 217, 133, 455, 183, 399, 403,
        651 ]
  results = [u for u in zip(a, b)]
  for r, s in zip(results, hf_groups(lattice, range(17))):
    assert len(s) == r[0]
    assert sum(len(u) for u in s) == r[1]

if __name__ == '__main__':
  test_supercells()
  test_hfgroups()
