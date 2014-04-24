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

def test_system():
  from pylada.crystal.cppwrappers import Structure
  from pylada.vasp import Vasp
  a = Vasp()
  b = Structure()

  assert a.system is None
  assert a._input['system'].keyword == 'system'
  assert a._input['system'].output_map(vasp=a, structure=b) is None

  a.system = 'system'
  assert a.system == 'system'
  assert 'system' in a._input['system'].output_map(vasp=a, structure=b)
  assert a._input['system'].output_map(vasp=a, structure=b)['system'] == 'system'

  b.name = 'hello'
  assert 'system' in a._input['system'].output_map(vasp=a, structure=b)
  assert a._input['system'].output_map(vasp=a, structure=b)['system'] == 'system: hello'

  a.system = None
  assert a.system is None
  assert 'system' in a._input['system'].output_map(vasp=a, structure=b)
  assert a._input['system'].output_map(vasp=a, structure=b)['system'] == 'hello'

  a.system = None
  assert a.system is None
  assert 'system' in a._input['system'].output_map(vasp=a, structure=b)
  assert a._input['system'].output_map(vasp=a, structure=b)['system'] == 'hello'

if __name__ == "__main__": 
  test_system()
