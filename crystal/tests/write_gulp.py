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

def testzb():
  """ Tries and writes a gulp file. """
  from numpy import array, abs, all
  from pylada.crystal.binary import zinc_blende
  from pylada.crystal.write import gulp

  a = zinc_blende() 
  string = [u.rstrip().lstrip() for u in gulp(a).splitlines()]
  string = [u for u in string if len(u) > 0]
  assert string[0] == 'name'
  assert string[1] == 'Zinc-Blende'
  assert string[2] == 'vectors'
  assert all(abs(array(string[3].split(), dtype='float64') - [0, 0.5, 0.5]) < 1e-8)
  assert all(abs(array(string[4].split(), dtype='float64') - [0.5, 0, 0.5]) < 1e-8)
  assert all(abs(array(string[5].split(), dtype='float64') - [0.5, 0.5, 0]) < 1e-8)
  assert string[6] == 'cartesian'
  assert string[7].split()[:2] == ['A', 'core']
  assert all(abs(array(string[7].split()[2:], dtype='float64')) < 1e-8)
  assert string[8].split()[:2] == ['B', 'core']
  assert all(abs(array(string[8].split()[2:], dtype='float64') - [0.25, 0.25, 0.25]) < 1e-8)

  string2 = gulp(a, symmgroup=216).splitlines()
  string2 = [u.rstrip().lstrip() for u in string2]
  string2 = [u for u in string2 if len(u) > 0]
  assert string2[:6] == string[:6]
  assert string2[6] == 'spacegroup'
  assert string2[7] == '216'
  assert string2[8:] == string[6:]
  
  a.symmgroup = 216
  string = gulp(a).splitlines()
  string = [u.rstrip().lstrip() for u in string]
  string = [u for u in string if len(u) > 0]
  assert string2 == string
  
  a[0].asymmetric = True
  a[1].asymmetric = False
  string = gulp(a).splitlines()
  string = [u.rstrip().lstrip() for u in string]
  string = [u for u in string if len(u) > 0]
  assert string2[:-1] == string
  del a[0].asymmetric
  string = gulp(a).splitlines()
  string = [u.rstrip().lstrip() for u in string]
  string = [u for u in string if len(u) > 0]
  assert string2 == string
  del a[1].asymmetric

  a[1].type = 'A'
  a.symmgroup = 227
  string = gulp(a).splitlines()
  string = [u.rstrip().lstrip() for u in string]
  string = [u for u in string if len(u) > 0]
  assert string2[:7] == string[:7]
  assert string[7] == '227'
  assert string2[8:-1] == string[8:]

if __name__ == '__main__':
  testzb()
