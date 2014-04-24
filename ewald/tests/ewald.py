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

def test():
  from numpy import all, abs, sqrt
  from pylada.crystal import Structure
  from pylada.ewald import ewald
  from quantities import angstrom, a0, Ry
  structure = Structure( [ [1,0,0],
                           [0,1,0],
                           [0,0,1] ], scale=50 )                               \
                       .add_atom(0, 0, 0, 'A', charge=1e0)                     \
                       .add_atom( float(a0.rescale(angstrom)/50.0), 0, 0, 'A',  
                                  charge=-1e0 )
  result = ewald(structure, cutoff = 80)
  assert abs(result.energy + 2e0*Ry) < 1e-3
  assert all(abs(abs(result[0].force) - [2e0, 0, 0]*Ry/a0) < 1e-3)
  assert all(abs(abs(result[1].force) - [2e0, 0, 0]*Ry/a0) < 1e-3)

  a = float(a0.rescale(angstrom)/50.0) / sqrt(2.)
  structure = Structure( [ [1,0,0],
                           [0,1,0],
                           [0,0,1] ], scale=50 )                               \
                       .add_atom(0, 0, 0, 'A', charge=1e0)                     \
                       .add_atom(0, a, a, 'A', charge=-1e0 )
  result = ewald(structure, cutoff = 80)
  assert abs(result.energy + 2e0*Ry) < 1e-3
  assert all(abs(abs(result[0].force) - [0, 2./sqrt(2), 2./sqrt(2)]*Ry/a0) < 1e-3)
  assert all(abs(abs(result[1].force) - [0, 2./sqrt(2), 2./sqrt(2)]*Ry/a0) < 1e-3)

if __name__ == '__main__':
  test()
