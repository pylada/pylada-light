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

def test_boolkeyword():
  from pylada.tools.input import BoolKeyword

  class Dummy(BoolKeyword): 
    keyword = 'whatever'
    pass

  a = Dummy()
  a.value = None
  assert a.value is None
  assert a.output_map() is None
  b = eval(repr(a), {'Dummy': Dummy})
  assert type(b) is type(a)
  assert b.value == a.value
  a.value = 1
  assert a.value is True
  assert a.output_map() == {'whatever': 'True'}
  b = eval(repr(a), {'Dummy': Dummy})
  assert type(b) is type(a)
  assert b.value == a.value

def test_valuekeyword():
  from pickle import loads, dumps
  from pylada.tools.input import ValueKeyword

  a = ValueKeyword('whatever')
  d = {'ValueKeyword': ValueKeyword}
  assert a.keyword == 'whatever'
  assert a.value is None
  assert a.output_map() is None
  assert eval(repr(a), d).__class__ is ValueKeyword
  assert eval(repr(a), d).value is None
  assert loads(dumps(a)).__class__ is ValueKeyword
  assert loads(dumps(a)).value is None

  a.raw = '5'
  assert type(a.value) is int and a.value == 5 and a.raw == '5'
  assert eval(repr(a), d).value == 5
  assert loads(dumps(a)).value == 5
  a.raw = '5.0'
  assert type(a.value) is float and abs(a.value - 5.0) < 1e-8 and a.raw == str(5.0)
  a.raw = 'five'
  assert type(a.value) is str and a.value == 'five' and a.raw == 'five'
  a.raw = '5 5.0 five'
  assert hasattr(a.value, '__iter__') and hasattr(a.value, '__len__')
  assert len(a.value) == 3
  assert type(a.value[0]) is int and a.value[0] == 5 
  assert type(a.value[1]) is float and abs(a.value[1] - 5.0) < 1e-8
  assert type(a.value[2]) is str and a.value[2] == 'five'
  assert type(a.raw) is str and len(a.raw.split('\n')) == 1 and len(a.raw.split()) == 3
  assert a.raw.split()[0] == '5' and a.raw.split()[1] == str(5.0) and a.raw.split()[2] == 'five' 

  assert 'whatever' in a.output_map()
  assert a.output_map()['whatever'] == ' '.join(str(u) for u in a.value)
  
  a.value = None
  assert a.value is None
  assert a.output_map() is None

def test_typedkeyword():
  from pickle import loads, dumps
  from pylada.tools.input import TypedKeyword

  a = TypedKeyword('whatever', int)
  d = {'TypedKeyword': TypedKeyword}
  assert a.keyword == 'whatever'
  assert a.value is None
  assert a.output_map() is None
  assert eval(repr(a), d).__class__ is TypedKeyword
  assert eval(repr(a), d).type is int
  assert eval(repr(a), d).value is None
  assert eval(repr(a), d).keyword == a.keyword
  assert loads(dumps(a)).__class__ is TypedKeyword
  assert loads(dumps(a)).type is int
  assert loads(dumps(a)).value is None
  assert loads(dumps(a)).keyword == a.keyword
  a.raw = '5'
  assert type(a.value) is int and a.value == 5 and a.raw == '5'
  a.value = '5'
  assert type(a.value) is int and a.value == 5 and a.raw == '5'
  assert type(eval(repr(a), d).value) is int 
  assert eval(repr(a), d).value == 5
  assert repr(loads(dumps(a))) == repr(a)
  a.value = None
  assert a.value is None
  assert a.output_map() is None
  a.value = 5.1
  assert type(a.value) is int and a.value == 5 and a.raw == '5'
  assert 'whatever' in a.output_map()
  assert a.output_map()['whatever'] == str(5)
  try: a.value = 'five'
  except ValueError: pass
  else: raise Exception()
  
  a = TypedKeyword('whatever', float)
  assert a.keyword == 'whatever'
  assert a.value is None
  assert a.output_map() is None
  a.raw = '5.1'
  assert type(a.value) is float and abs(a.value - 5.1) < 1e-8 and a.raw == str(5.1)
  a.value = '5.1'
  assert type(a.value) is float and abs(a.value - 5.1) < 1e-8 and a.raw == str(5.1)
  a.value = None
  assert a.value is None
  assert a.output_map() is None
  a.value = 5.1
  assert type(a.value) is float and abs(a.value - 5.1) < 1e-8 and a.raw == str(5.1)
  assert a.output_map()['whatever'] == str(5.1)
  try: a.value = 'five'
  except ValueError: pass
  else: raise Exception()
  
  a = TypedKeyword('whatever', str)
  assert a.keyword == 'whatever'
  assert a.value is None
  assert a.output_map() is None
  a.raw = '5.1'
  assert type(a.value) is str and a.value == '5.1' and a.raw == '5.1'
  a.value = '5.1'
  assert type(a.value) is str and a.value == '5.1' and a.raw == '5.1'
  a.value = None
  assert a.value is None
  assert a.output_map() is None
  a.value = 5.1
  assert type(a.value) is str and a.value == str(5.1) and a.raw == str(5.1)
  assert a.output_map()['whatever'] == str(5.1)
  a.value = 'five'
  assert type(a.value) is str and a.value == 'five' and a.raw == 'five'

  a = TypedKeyword('whatever', [int])
  a.raw = '5 5'
  assert hasattr(a.value, '__iter__') 
  assert len(a.value) == 2
  assert all(type(v) is int for v in a.value)
  assert all(v == 5 for v in a.value)
  a.value = 5, 5.3, 5.2
  assert hasattr(a.value, '__iter__') 
  assert len(a.value) == 3
  assert all(type(v) is int for v in a.value)
  assert all(v == 5 for v in a.value)
  assert all(v == '5' for v in a.raw.split())
  assert all(v == '5' for v in a.output_map()['whatever'].split())
  a.raw = ''
  assert a.value is None
  a.value = []
  assert a.value is None
  try: a.raw = '0 FIVE'
  except ValueError: pass
  else: raise Exception()
  assert eval(repr(a), d).type == [int]
  assert eval(repr(a), d).type != int
  assert eval(repr(a), d).type != float
  assert repr(loads(dumps(a))) == repr(a)

  a = TypedKeyword('whatever', [str, int])
  a.raw = 'five 5'
  assert hasattr(a.value, '__iter__') 
  assert len(a.value) == 2
  assert type(a.value[0]) is str and a.value[0] == 'five'
  assert type(a.value[1]) is int and a.value[1] == 5
  assert a.output_map()['whatever'].split()[0] == 'five' 
  assert a.output_map()['whatever'].split()[1] == '5' 
  try: a.value = 'five', 6, 7
  except ValueError: pass
  else: raise Exception()
  try: a.raw = '0 five'
  except ValueError: pass
  else: raise Exception()

def test_variablelistkeyword():
  from numpy import all, array, arange
  from pylada.tools.input import VariableListKeyword

  a = VariableListKeyword(keyword='whatever', type=int)
  assert a.keyword == 'whatever'
  assert a.value is None
  assert a.output_map() is None
  assert eval(repr(a), {'VariableListKeyword': VariableListKeyword}).__class__ \
               is VariableListKeyword
  a.raw = '5 0 1 2 3 4'
  assert type(a.value) is list and len(a.value) == 5
  assert all(type(v) is int for v in a.value)
  assert all(array(a.value) - arange(5) == 0)
  assert len(a.raw.split('\n')) == 2
  assert a.raw.split('\n')[0] == '5'
  a.value = 5.1, 4, 3, 2, 1, 0
  assert type(a.value) is list and len(a.value) == 6
  assert all(type(v) is int for v in a.value)
  assert all(array(a.value) - arange(6)[::-1] == 0)
  assert all(array(a.raw.split(), dtype='int32') - ([6]+range(6)[::-1]) == 0)
  assert len(a.raw.split('\n')) == 2
  assert a.raw.split('\n')[0] == '6'

  try: a.value = 5, 'a'
  except ValueError: pass
  else: raise Exception()

def test_choicekeyword():
  from pickle import loads, dumps
  from pylada.tools.input import ChoiceKeyword

  a = ChoiceKeyword(['five', 5], keyword='whatever')
  d = {'ChoiceKeyword': ChoiceKeyword}
  assert a.value is None
  assert a.output_map() is None
  assert eval(repr(a), d).__class__ is ChoiceKeyword
  assert eval(repr(a), d).keyword == a.keyword
  assert eval(repr(a), d).values == a.values
  assert eval(repr(a), d).output_map() is None
  assert repr(loads(dumps(a))) == repr(a)

  a.value = 5
  assert a.value == 5
  assert len(a.output_map()) == 1
  assert 'whatever' in a.output_map()
  assert a.output_map()['whatever'] == '5'
  assert eval(repr(a), d).__class__ is ChoiceKeyword
  assert eval(repr(a), d).keyword == a.keyword
  assert eval(repr(a), d).values == a.values
  assert eval(repr(a), d).value == 5
  assert repr(loads(dumps(a))) == repr(a)

  a.value = 'five'
  assert a.value == 'five'
  assert len(a.output_map()) == 1
  assert 'whatever' in a.output_map()
  assert a.output_map()['whatever'] == 'five'
  assert repr(eval(repr(a), d)) == repr(a)
  assert repr(loads(dumps(a))) == repr(a)
  a.value = 'FIVE'
  assert a.value == 'five'
  assert len(a.output_map()) == 1
  assert 'whatever' in a.output_map()
  assert a.output_map()['whatever'] == 'five'
  assert repr(eval(repr(a), d)) == repr(a)
  assert repr(loads(dumps(a))) == repr(a)

  try: a.value = 6
  except: pass
  else: raise Exception()
  try: a.value = 'six'
  except: pass
  else: raise Exception()

def test_quantitykeyword():
  from itertools import chain
  from numpy import all, abs, array
  from quantities import hartree, eV, kbar
  from pylada.tools.input import QuantityKeyword
  import quantities
  import numpy
  d = locals().copy()
  d.update(quantities.__dict__)
  d.update(numpy.__dict__)

  a = QuantityKeyword(units=eV, keyword='whatever')
  assert a.output_map() is None
  assert a.units is eV
  a.value = 5*hartree
  assert a.value.units == eV
  try: assert a.value.units == hartree
  except: pass
  else: raise Exception()
  assert abs(a.value - 5*hartree) < 1e-8
  assert abs(a.value.magnitude - (5*hartree.rescale(eV).magnitude)) < 2e-8
  try: a.value = 5*kbar
  except: pass
  else: raise Exception()
  a.value = 5
  assert a.value.units == eV
  assert abs(a.value.magnitude - 5) < 2e-8
  a.raw = '5.3\n'
  assert a.value.units == eV
  assert abs(a.value.magnitude - 5.3) < 2e-8
  a.raw = '5.3'
  assert a.value.units == eV
  assert abs(a.value.magnitude - 5.3) < 2e-8
  assert abs(eval(repr(a), d).value - a.value) < 1e-8
  assert eval(repr(a), d).units == a.units
  assert not hasattr(eval(repr(a), d), 'shape')

  assert 'whatever' in a.output_map()
  assert a.output_map()['whatever'] == str(float(a.value))


  a = QuantityKeyword(units=eV, keyword='whatever', shape=(3,3))
  assert a.output_map() is None
  assert a.units is eV
  value = [[1.0, 1.2, 5], [5.6, 0, 0], [0.1, 0.2, 5]]
  a.value = value * hartree
  assert all(abs(a.value - value*hartree)) < 1e-8
  assert all(abs(a.value.magnitude - value*hartree.rescale(eV).magnitude)) < 1e-8
  a.raw = ' '.join(str(u) for u in chain(*value))
  assert all(abs(a.value - value*eV)) < 1e-8
  assert all(abs(a.value.magnitude - value)) < 1e-8
  assert all(array(a.raw.split(), dtype='float64') == [u for u in chain(*value)])
  assert len(a.raw.split('\n')) == 3
  assert all(abs(eval(repr(a), d).value - a.value) < 1e-8)
  assert eval(repr(a), d).units == a.units
  assert eval(repr(a), d).shape == a.shape
  a.raw = a.raw
  assert all(abs(a.value - value*eV)) < 1e-8
  assert len(a.output_map()['whatever'].split('\n')) == 3
  assert len(a.output_map()['whatever'].split('\n')[0].split()) == 3
  assert len(a.output_map()['whatever'].split('\n')[1].split()) == 3
  assert len(a.output_map()['whatever'].split('\n')[2].split()) == 3
  assert all(abs(array(a.output_map()['whatever'].split('\n')[0].split(), dtype='float64') - value[0]) < 1e-8)
  assert all(abs(array(a.output_map()['whatever'].split('\n')[1].split(), dtype='float64') - value[1]) < 1e-8)
  assert all(abs(array(a.output_map()['whatever'].split('\n')[2].split(), dtype='float64') - value[2]) < 1e-8)

  try: a.raw = '0 1 2\n2 3 4\n\n'
  except: pass
  else: raise Exception()
  try: a.raw = '0 a 2\n2 3 4\n56 6 7\n'
  except: pass
  else: raise Exception()
  try: a.raw = '0 1 2 4\n2 3 4 4\n56 6 7 4\n'
  except: pass
  else: raise Exception()
  try: a.value = value * kbar
  except: pass
  else: raise Exception()
  
 
if __name__ == "__main__":
  test_boolkeyword()
  test_choicekeyword()
  test_valuekeyword()
  test_typedkeyword()
  test_variablelistkeyword()
  test_quantitykeyword()

