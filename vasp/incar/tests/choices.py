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
  from pylada.vasp.incar._params import Choices

  a = Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]})
  a.value = 'a'
  assert a.value == 'A'
  assert loads(dumps(a)).incar_string() == 'ALGO = A'
  a.value = 'aa'
  assert a.value == 'A'
  assert loads(dumps(a)).incar_string() == 'ALGO = A'
  a.value = 0
  assert a.value == 'A'
  assert loads(dumps(a)).incar_string() == 'ALGO = A'
  a.value = 'b'
  assert a.value == 'B'
  assert loads(dumps(a)).incar_string() == 'ALGO = B'
  a.value = 'bb'
  assert a.value == 'B'
  assert loads(dumps(a)).incar_string() == 'ALGO = B'
  a.value = 1
  assert a.value == 'B'
  assert loads(dumps(a)).incar_string() == 'ALGO = B'
  try: a.value = 2
  except: pass
  else: raise RuntimeError()
  a.value = None
  assert a.incar_string() is None
  assert repr(a) == "Choices('algo', {'A': ['aa', 0, 'a'], 'B': ['bb', 1, 'b']}, None)"

  a = Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]}, 'b')
  assert a.value == 'B'
  a = Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]}, 'bb')
  assert a.value == 'B'
  a = Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]}, 1)
  assert a.value == 'B'
  try: a = Choices('algo', {'A': ['aa', 0], 'B': ['bb', 1]}, 2)
  except: pass
  else: raise RuntimeError()

if __name__ == "__main__":
  from sys import argv, path 
  from numpy import array
  if len(argv) > 0: path.extend(argv[1:])
  
  test()

