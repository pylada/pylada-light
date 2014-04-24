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
  from pylada.vasp import Vasp

  Restart = namedtuple('Restart', ['success', 'lmaxmix'])
  a = Vasp()
  o = a._input['lsorbit']
  d = {'LSorbit': o.__class__}
  assert a.lsorbit is None
  assert a.nonscf == False
  assert a._input['lsorbit'].keyword == 'lsorbit'
  assert a._input['nonscf'].keyword is None
  assert o.output_map(vasp=a) is None
  assert eval(repr(o), d).output_map(vasp=a) is None
  assert eval(repr(o), d).value is None
  assert loads(dumps(o)).value is None

  a.lsorbit = True
  assert a.nonscf
  assert a.lsorbit
  try: a._input['lsorbit'].output_map(vasp=a)
  except ValueError: pass
  else: raise Exception()
  a.restart = Restart(False, 7)
  try: a._input['lsorbit'].output_map(vasp=a)
  except ValueError: pass
  else: raise Exception()
  a.restart = Restart(True, 7)
  assert 'lsorbit' in o.output_map(vasp=a)
  assert o.output_map(vasp=a)['lsorbit'] == '.TRUE.'
  assert a.lmaxmix == 7
  a.lmaxmix = 5
  a.restart = Restart(True, 6)
  assert 'lsorbit' in o.output_map(vasp=a)
  assert o.output_map(vasp=a)['lsorbit'] == '.TRUE.'
  assert a.lmaxmix == 6
  assert loads(dumps(o)).value is True
  assert eval(repr(o), d).value is True
  

if __name__ == "__main__": test()
  
