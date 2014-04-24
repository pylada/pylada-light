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

def test_bool():
  from pickle import loads, dumps
  from pylada.vasp import Vasp
  a = Vasp()

  assert a._input['addgrid'].keyword == 'addgrid'
  assert a._input['addgrid'].output_map() is None
  assert a.addgrid is None
  a.addgrid = False
  assert a.addgrid is False
  assert 'addgrid' in a._input['addgrid'].output_map()
  assert a._input['addgrid'].output_map()['addgrid'] == '.FALSE.'
  a.addgrid = True
  assert a.addgrid is True
  assert 'addgrid' in a._input['addgrid'].output_map()
  assert a._input['addgrid'].output_map()['addgrid'] == '.TRUE.'
  a.addgrid = None
  assert a._input['addgrid'].keyword == 'addgrid'
  assert a._input['addgrid'].output_map() is None
  a.addgrid = 0
  assert a.addgrid is False

  a.addgrid = False
  o = a._input['addgrid']
  d = {'BoolKeyword': o.__class__}
  assert repr(eval(repr(o), d)) == repr(o)
  assert eval(repr(o), d).output_map()['addgrid'] == '.FALSE.'
  assert repr(loads(dumps(o))) == repr(o)
  a.addgrid = True
  o = a._input['addgrid']
  assert repr(eval(repr(o), d)) == repr(o)
  assert eval(repr(o), d).output_map()['addgrid'] == '.TRUE.'
  assert repr(loads(dumps(o))) == repr(o)
  a.addgrid = None
  o = a._input['addgrid']
  assert repr(eval(repr(o), d)) == repr(o)
  assert eval(repr(o), d).output_map() is None
  assert repr(loads(dumps(o))) == repr(o)


def test_choice():
  from pickle import loads, dumps
  from pylada.vasp import Vasp
  a = Vasp()

  assert a.ispin is None
  assert a._input['ispin'].keyword == 'ispin'
  assert a._input['ispin'].output_map() is None
  a.ispin = 1
  assert a.ispin == 1
  assert 'ispin' in a._input['ispin'].output_map()
  assert a._input['ispin'].output_map()['ispin'] == '1'
  a.ispin = 2
  assert a.ispin == 2
  assert 'ispin' in a._input['ispin'].output_map()
  assert a._input['ispin'].output_map()['ispin'] == '2'
  a.ispin = None
  assert a.ispin is None
  assert a._input['ispin'].keyword == 'ispin'
  assert a._input['ispin'].output_map() is None

  try: a.ispin = 5
  except: pass
  else: raise RuntimeError()

  a.ispin = '1'
  assert a.ispin == 1
  a.ispin = '2'
  assert a.ispin == 2

  try: a.ispin = '3'
  except: pass
  else: raise RuntimeError()

  a.ispin = None
  o = a._input['ispin']
  d = {'ChoiceKeyword': o.__class__}
  assert repr(eval(repr(o), d)) == repr(o)
  assert eval(repr(o), d).output_map() is None
  assert repr(loads(dumps(o))) == repr(o)
  a.ispin = 2
  o = a._input['ispin']
  assert repr(eval(repr(o), d)) == repr(o)
  assert eval(repr(o), d).output_map()['ispin'] == '2'
  assert repr(loads(dumps(o))) == repr(o)

def test_alias():
  from pylada.vasp import Vasp
  from pylada.error import ValueError
  a = Vasp()

  assert a.ismear is None
  assert a._input['ismear'].keyword == 'ismear'
  assert a._input['ismear'].output_map() is None
  map = a._input['ismear'].aliases
  assert len(map) != 0
  for i, items in map.iteritems():
    for item in items:
      a.ismear = item
      assert a.ismear == items[0]
      assert 'ismear' in a._input['ismear'].output_map()
      assert a._input['ismear'].output_map()['ismear'] == str(i)
      a.ismear = i
      assert a.ismear == items[0]
    a.ismear = str(i) 
    assert a.ismear == items[0]

  try: a.lmaxmix = 'a'
  except ValueError: pass
  else: raise Exception()

def test_typed():
  from pylada.vasp import Vasp
  from pylada.error import ValueError
  a = Vasp()

  assert a.nbands is None
  assert a._input['nbands'].keyword == 'nbands'
  assert a._input['nbands'].output_map() is None

  a.nbands = 50
  assert a.nbands == 50
  assert 'nbands' in a._input['nbands'].output_map()
  assert a._input['nbands'].output_map()['nbands'] == str(a.nbands)
  a.nbands = '51'
  assert a.nbands == 51
  assert 'nbands' in a._input['nbands'].output_map()
  assert a._input['nbands'].output_map()['nbands'] == str(a.nbands)
  a.nbands = None
  assert a.nbands is None
  assert a._input['nbands'].output_map() is None

  try: a.nbands = 'a'
  except ValueError: pass
  else: raise Exception()

  assert a.smearings is None
  assert a._input['smearings'].keyword == 'smearings'
  assert a._input['smearings'].output_map() is None
  a.smearings = [1.5, 1.0, 0.5]
  assert len(a.smearings) == 3
  assert all(abs(i-v) < 1e-8 for i, v in zip(a.smearings, [1.5, 1.0, 0.5]))
  assert 'smearings' in a._input['smearings'].output_map()
  assert all(abs(float(i)-v) < 1e-8 for i, v in zip(a._input['smearings'].output_map()['smearings'].split(), [1.5, 1.0, 0.5]))
  a.smearings = ['1.2', '0.2']
  assert len(a.smearings) == 2
  assert all(abs(i-v) < 1e-8 for i, v in zip(a.smearings, [1.2, 0.2]))
  assert 'smearings' in a._input['smearings'].output_map()
  assert all(abs(float(i)-v) < 1e-8 for i, v in zip(a._input['smearings'].output_map()['smearings'].split(), [1.2, 0.2]))
  a.smearings = '1.3 0.3'
  assert len(a.smearings) == 2
  assert all(abs(i-v) < 1e-8 for i, v in zip(a.smearings, [1.3, 0.3]))
  assert 'smearings' in a._input['smearings'].output_map()
  assert all(abs(float(i)-v) < 1e-8 for i, v in zip(a._input['smearings'].output_map()['smearings'].split(), [1.3, 0.3]))
  a.smearings = '1.3, 0.3'
  assert len(a.smearings) == 2
  assert all(abs(i-v) < 1e-8 for i, v in zip(a.smearings, [1.3, 0.3]))
  a.smearings = '1.3; 0.3'
  assert len(a.smearings) == 2
  assert all(abs(i-v) < 1e-8 for i, v in zip(a.smearings, [1.3, 0.3]))
  a.smearings = None
  assert a.smearings is None
  assert a._input['smearings'].output_map() is None
  
  try: a.smearings = 5.5
  except ValueError: pass
  else: raise Exception()
  try: a.smearings = [5.5, 'a']
  except ValueError: pass
  else: raise Exception()


if __name__ == '__main__':
  test_bool()
  test_choice()
  test_alias()
  test_typed()
