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
  from pickle import loads, dumps
  from pylada.vasp import Vasp
  from pylada.error import ValueError
 
  a = Vasp()

  for key in a._input.keys():
    if key not in ['isif', 'nsw', 'ibrion', 'relaxation']: 
      del a._input[key]
  assert a.relaxation == 'static'
  assert len(a.output_map(vasp=a)) == 1
  assert a.output_map(vasp=a)['ibrion'] == str(-1)

  a.relaxation = 'cellshape'
  assert a.relaxation == 'cellshape'
  assert a.isif == 5
  assert a.nsw == 50
  assert a.ibrion == 2
  assert len(a.output_map(vasp=a)) == 3
  assert a.output_map(vasp=a)['ibrion'] == str(2)
  assert a.output_map(vasp=a)['isif'] == str(5)
  assert a.output_map(vasp=a)['nsw'] == str(50)
  a = loads(dumps(a))
  assert len(a.output_map(vasp=a)) == 3
  assert a.output_map(vasp=a)['ibrion'] == str(2)
  assert a.output_map(vasp=a)['isif'] == str(5)
  assert a.output_map(vasp=a)['nsw'] == str(50)

  a.relaxation = 'cellshape volume'
  a.nsw = 25
  assert a.relaxation == 'cellshape volume'
  assert a.isif == 6
  assert a.nsw == 25
  assert a.ibrion == 2
  assert len(a.output_map(vasp=a)) == 3
  assert a.output_map(vasp=a)['ibrion'] == str(2)
  assert a.output_map(vasp=a)['isif'] == str(6)
  assert a.output_map(vasp=a)['nsw'] == str(25)

  a.relaxation = 'ions'
  assert a.relaxation == 'ionic'
  assert a.isif == 2
  a.relaxation = 'ionic'
  assert a.relaxation == 'ionic'
  assert a.isif == 2

  a.relaxation = 'cellshape, volume ions'
  print a.relaxation
  assert a.relaxation == 'cellshape ionic volume'
  assert a.isif == 3

  a.relaxation = 'cellshape, ionic'
  assert a.relaxation == 'cellshape ionic'
  assert a.isif == 4

  a.relaxation = 'volume'
  assert a.relaxation == 'volume'
  assert a.isif == 7

  a.relaxation = 'static'
  assert a.ibrion == -1
  assert a.nsw == 0
  assert a.isif == 2

  try: a.relaxation = 'ions, volume'
  except ValueError: pass
  else: raise Exception

if __name__ == "__main__": test()

