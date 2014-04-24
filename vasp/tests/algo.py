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
  import pylada
  
  pylada.is_vasp_4 = True
  a = Vasp()

  # default.
  assert a.algo == 'Fast'
  # wrong argument.
  try: a.algo = 0
  except: pass
  else: raise RuntimeError()
  try: a.algo = "WTF"
  except: pass
  else: raise RuntimeError()

  # possible inputs and some.
  d = {
      'Very_Fast': ['very fast', 'VERY-fAst', 'very_FAST', 'v'],
      'VeryFast': ['very fast', 'VERY-fAst', 'very_FAST', 'v'],
      'Fast': ['fast', 'f'],
      'Normal': ['normal', 'n'],
      'Damped': ['damped', 'd'],
      'Diag': ['diag'],
      'All': ['all', 'a'],
      'Nothing': ['nothing'],
      'chi': ['chi'],
      'GW': ['gw'],
      'GW0': ['gw0'],
      'scGW': ['scgw'],
      'scGW0': ['scgw0'],
      'Conjugate': ['conjugate', 'c'],
      'Subrot': ['subrot', 's'],
      'Eigenval': ['eigenval', 'e']
  }
  vasp5 = 'Subrot', 'chi', 'GW', 'GW0', 'scGW', 'scGW0', 'Conjugate', 'Eigenval', 'Exact', 'Nothing'
  dictionary = {'Algo': a._input['algo'].__class__}
  for isvasp4 in [True, False]:
    pylada.is_vasp_4 = isvasp4
    for key, items in d.iteritems():
      for value in items:
        if key in vasp5 and isvasp4:
          try: a.algo = value
          except: pass 
          else: raise RuntimeError((value, key))
          continue
        a.algo = value
        o = a._input['algo']
        if key == 'VeryFast' and isvasp4:
          assert a.algo == 'Very_Fast'
          assert o.output_map()['algo'] == 'Very_Fast'
          assert loads(dumps(o)).output_map()["algo"] == 'Very_Fast'
          assert eval(repr(o), dictionary).output_map()["algo"] == 'Very_Fast'
        elif key == 'Very_Fast' and not isvasp4:
          assert a.algo == 'VeryFast'
          assert o.output_map()['algo'] == 'VeryFast'
          assert loads(dumps(o)).output_map()["algo"] == 'VeryFast'
          assert eval(repr(o), dictionary).output_map()["algo"] == 'VeryFast'
        else:
          assert a.algo == key
          assert o.output_map()['algo'] == key
          assert loads(dumps(o)).output_map()["algo"] == key
          assert eval(repr(o), dictionary).output_map()["algo"] == key

  a.algo = None
  assert a.algo is None
  assert o.output_map() is None
  assert loads(dumps(o)).output_map() is None
  assert eval(repr(o), dictionary).output_map() is None


if __name__ == '__main__': 
  test()

 
