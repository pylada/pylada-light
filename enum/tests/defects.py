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

def test_iterator():
  """ Test defect iterator. """
  from numpy import ones, logical_not
  from pylada.enum.defects import Iterator

  size = 8
  mask = ones(size, dtype='bool')
  mask0 = mask.copy()
  mask0[::4] = False
  mask0[1::4] = False
  mask2 = mask.copy()
  mask2[2:] = False


  results = [0, 4, 0, 0, 3, 3, 2, 2], [4, 0, 0, 0, 3, 3, 2, 2], \
            [0, 4, 0, 2, 3, 3, 0, 2], [4, 0, 0, 2, 3, 3, 0, 2], \
            [0, 4, 2, 0, 3, 3, 0, 2], [4, 0, 2, 0, 3, 3, 0, 2], \
            [0, 4, 0, 2, 3, 3, 2, 0], [4, 0, 0, 2, 3, 3, 2, 0], \
            [0, 4, 2, 0, 3, 3, 2, 0], [4, 0, 2, 0, 3, 3, 2, 0], \
            [0, 4, 2, 2, 3, 3, 0, 0], [4, 0, 2, 2, 3, 3, 0, 0], \
            [3, 4, 0, 0, 0, 3, 2, 2], [4, 3, 0, 0, 0, 3, 2, 2], \
            [3, 4, 0, 2, 0, 3, 0, 2], [4, 3, 0, 2, 0, 3, 0, 2], \
            [3, 4, 2, 0, 0, 3, 0, 2], [4, 3, 2, 0, 0, 3, 0, 2], \
            [3, 4, 0, 2, 0, 3, 2, 0], [4, 3, 0, 2, 0, 3, 2, 0], \
            [3, 4, 2, 0, 0, 3, 2, 0], [4, 3, 2, 0, 0, 3, 2, 0], \
            [3, 4, 2, 2, 0, 3, 0, 0], [4, 3, 2, 2, 0, 3, 0, 0], \
            [3, 4, 0, 0, 3, 0, 2, 2], [4, 3, 0, 0, 3, 0, 2, 2], \
            [3, 4, 0, 2, 3, 0, 0, 2], [4, 3, 0, 2, 3, 0, 0, 2], \
            [3, 4, 2, 0, 3, 0, 0, 2], [4, 3, 2, 0, 3, 0, 0, 2], \
            [3, 4, 0, 2, 3, 0, 2, 0], [4, 3, 0, 2, 3, 0, 2, 0], \
            [3, 4, 2, 0, 3, 0, 2, 0], [4, 3, 2, 0, 3, 0, 2, 0], \
            [3, 4, 2, 2, 3, 0, 0, 0], [4, 3, 2, 2, 3, 0, 0, 0]
  
  iterator = Iterator(size, (1, 4, mask2), (2, 2, mask0), (2, 3, logical_not(mask0)))
  for i, x in enumerate(iterator):
    assert all(x == results[i])
  assert i == len(results) - 1
  iterator.reset()
  doiter = False
  for i, x in enumerate(iterator):
    doiter = True
    assert all(x == results[i])
  assert doiter
  assert i == len(results) - 1

def test_defects():
  from pylada.crystal.bravais import fcc
  from pylada.enum.defects import defects
  lattice = fcc()
  lattice[0].type = 'Zr', 'Ti'
  lattice.add_atom(0.25, 0.25, 0.25, 'O', 'A')
  lattice.add_atom(0.75, 0.75, 0.75, 'O', 'A')


  i = 0
  for i, (x, hft, hermite) in enumerate(defects(lattice, 32, {'A': 2, 'Ti': 2})):
    continue; print x
  print i


if __name__ == '__main__':
  test_iterator()
  test_defects()
