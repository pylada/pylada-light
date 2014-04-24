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

""" Test for supercell function. """
def test_supercell():
  """ Simple supercell test. """
  from numpy import identity, abs, all, dot
  from numpy.linalg import inv
  from pylada.crystal.cppwrappers import supercell, Structure, are_periodic_images as api
  from quantities import angstrom
  lattice = Structure( 0.0, 0.5, 0.5,
                       0.5, 0.0, 0.5,
                       0.5, 0.5, 0.0, scale=2.0, m=True ) \
                     .add_atom(0, 0, 0, "As")           \
                     .add_atom(0.25, 0.25, 0.25, ['In', 'Ga'], m = True)
  result = supercell(lattice, dot(lattice.cell, [ [-1, 1, 1],
                                [1, -1, 1], 
                                [1, 1, -1] ] ) )
  assert all(abs(result.cell - identity(3)) < 1e-8)
  assert abs(result.scale - 2 * angstrom) < 1e-8
  assert getattr(result, 'm', False) 
  assert all(abs(result[0].pos - [0.00, 0.00, 0.00]) < 1e-8) and result[0].type == "As" \
         and getattr(result[0], 'site', -1) == 0 and api(result[0].pos, lattice[0].pos, inv(lattice.cell))
  assert all(abs(result[1].pos - [0.25, 0.25, 0.25]) < 1e-8) and result[1].type == ["In", "Ga"] \
         and getattr(result[1], 'm', False) and getattr(result[1], 'site', -1) == 1 \
         and api(result[1].pos, lattice[1].pos, inv(lattice.cell))
  assert all(abs(result[2].pos - [0.50, 0.00, 0.50]) < 1e-8) and result[2].type == "As" \
         and getattr(result[2], 'site', -1) == 0 and api(result[2].pos, lattice[0].pos, inv(lattice.cell))
  assert all(abs(result[3].pos - [0.75, 0.25, 0.75]) < 1e-8) and result[3].type == ["In", "Ga"] \
         and getattr(result[3], 'm', False) and getattr(result[3], 'site', -1) == 1 \
         and api(result[3].pos, lattice[1].pos, inv(lattice.cell))
  assert all(abs(result[4].pos - [0.50, 0.50, 0.00]) < 1e-8) and result[4].type == "As" \
         and getattr(result[4], 'site', -1) == 0 and api(result[4].pos, lattice[0].pos, inv(lattice.cell))
  assert all(abs(result[5].pos - [0.75, 0.75, 0.25]) < 1e-8) and result[5].type == ["In", "Ga"] \
         and getattr(result[5], 'm', False) and getattr(result[5], 'site', -1) == 1 \
         and api(result[5].pos, lattice[1].pos, inv(lattice.cell))
  assert all(abs(result[6].pos - [0.00, 0.50, 0.50]) < 1e-8) and result[6].type == "As" \
         and getattr(result[6], 'site', -1) == 0 and api(result[6].pos, lattice[0].pos, inv(lattice.cell))
  assert all(abs(result[7].pos - [0.25, 0.75, 0.75]) < 1e-8) and result[7].type == ["In", "Ga"] \
         and getattr(result[7], 'm', False) and getattr(result[7], 'site', -1) == 1 \
         and api(result[7].pos, lattice[1].pos, inv(lattice.cell))

def get_cell(n=5):
  from numpy.random import randint
  from numpy.linalg import det
  cell = randint(2*n, size=(3,3)) - n
  while abs(det(cell)) < 1e-8:
    cell = randint(2*n, size=(3,3)) - n
  return cell

def test_manysupercell():
  from numpy import dot
  from numpy.linalg import inv, det
  from pylada.crystal import supercell, binary
  from pylada.crystal.cppwrappers import are_periodic_images as api
  lattice = binary.zinc_blende()
  invlat = inv(lattice.cell)
  for i in xrange(100):
    cell = get_cell()
    struc = supercell(lattice, dot(lattice.cell, cell))
    assert len(struc) == len(lattice) * int(abs(det(cell))+0.01)
    invcell = inv(struc.cell)
    for i, atom in enumerate(struc):
      # compare to lattice
      tolat = [api(atom.pos, site.pos, invlat) for site in lattice]
      assert tolat.count(True) == 1
      assert tolat.index(True) == atom.site
      assert lattice[tolat.index(True)].type == atom.type
      # compare to self
      tolat = [api(atom.pos, site.pos, invcell) for site in struc]
      assert tolat.count(True) == 1
      assert i == tolat.index(True)
         
if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])

  test_supercell()
  test_manysupercell()
