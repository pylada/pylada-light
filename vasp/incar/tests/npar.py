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
  from collections import namedtuple
  from pickle import loads, dumps
  from pylada.vasp.incar._params import Npar

  Comm = namedtuple('Comm', ['n'])

  # ispin == 1
  assert Npar(0).incar_string(comm=Comm(16)) is None
  assert Npar(1).incar_string(comm=Comm(16)) == "NPAR = 1"
  assert Npar(2).incar_string(comm=Comm(16)) == "NPAR = 2"
  assert Npar('power of two').incar_string(comm=Comm(16)) == "NPAR = 4"
  assert Npar('power of two').incar_string(comm=Comm(17)) is None
  assert Npar('power of two').incar_string(comm=Comm(2)) == "NPAR = 1"
  assert Npar('sqrt').incar_string(comm=Comm(16)) == "NPAR = 4"
  assert repr(Npar(1)) == "Npar(1)"
  assert repr(loads(dumps(Npar(2)))) == "Npar(2)"
  assert repr(loads(dumps(Npar("power of two")))) == "Npar('power of two')"

if __name__ == "__main__":
  from sys import argv, path 
  from numpy import array
  if len(argv) > 0: path.extend(argv[1:])
  
  test()

