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

""" Checks split-config routine. """
def check_bcc():
  from numpy import abs
  from pylada.crystal.cppwrappers import Structure, splitconfigs, supercell
  structure = Structure([[-0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5]])\
                       .add_atom(0,0,0,"Mo")
  configs = splitconfigs(structure, structure[0], 12)
  assert len(configs) == 1
  assert abs(configs[0][1] - 1e0) < 1e-8
  for u in configs[0][0]: assert u[0] is structure[0]
  assert all(abs(configs[0][0][0][1] - [ 0.,  0.,  0.]) < 1e-8)
  assert all(abs(configs[0][0][1][1] - [  8.66025404e-01,  -1.11022302e-16,   0.00000000e+00]) < 1e-8)
  assert all(abs(configs[0][0][2][1] - [ 0.28867513,  0.81649658,  0.        ]) < 1e-8)
  assert all(abs(configs[0][0][3][1] - [ 0.28867513, -0.40824829,  0.70710678]) < 1e-8)
  assert all(abs(configs[0][0][4][1] - [ 0.28867513, -0.40824829, -0.70710678]) < 1e-8)
  assert all(abs(configs[0][0][5][1] - [-0.28867513,  0.40824829,  0.70710678]) < 1e-8)
  assert all(abs(configs[0][0][6][1] - [-0.28867513,  0.40824829, -0.70710678]) < 1e-8)
  assert all(abs(configs[0][0][7][1] - [-0.28867513, -0.81649658,  0.        ]) < 1e-8)
  assert all(abs(configs[0][0][8][1] - [ -8.66025404e-01,   1.11022302e-16,   0.00000000e+00]) < 1e-8)
  assert all(abs(configs[0][0][9][1] - [ 0.57735027,  0.40824829,  0.70710678]) < 1e-8)
  assert all(abs(configs[0][0][10][1] - [ 0.57735027,  0.40824829, -0.70710678]) < 1e-8)
  assert all(abs(configs[0][0][11][1] - [ 0.57735027, -0.81649658,  0.        ]) < 1e-8)

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  check_bcc()
