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

""" Makes it easier to create pretty representations """
def uirepr(object, name=None, defaults=None, exclude=None):
  """ Returns string of representation. """
  if not hasattr(object, '__ui_repr__'): return repr(object)
  imports = {}
  collected = object.__ui_repr__(imports, name, defaults, exclude)

  result = ''
  for key in sorted(imports.iterkeys()):
    values = list(imports[key])
    result += 'from {0} import {1}\n'.format(key, ', '.join(values))

  # print headers
  result += '\n{0}'.format(collected[None])
  del collected[None]
  if len(collected) == 0: return result
  notnone = [key for key, v in collected.iteritems() if v is not None]
  if len(notnone): 
    length = max(len(key) for key in notnone)
    for key in sorted(collected.keys()):
      value = collected[key]
      if value is not None:
        value = value.rstrip()
        if value.count('\n') != 0:
          value = value.replace('\n', '\n' + ''.join([' ']*(length+4)))
        result += '\n{0:<{length}} = {1}'.format(key, value, length=length)
  for key in sorted(collected.keys()):
    value = collected[key]
    if value is None: result += '\n{0}'.format(key)
    
  return result

def add_to_imports(object, imports):
  from inspect import isclass, isfunction
  if object is None: return
  if isclass(object) or isfunction(object): 
    key   = object.__module__
    value = object.__name__
  else:
    key   = object.__class__.__module__
    value = object.__class__.__name__
  if key == '__builtin__': return
  if key in imports: imports[key].add(value)
  else: imports[key] = set([value])

def template_ui_repr(self, imports, name=None, defaults=None, exclude=None):
  """ Creates user friendly representation. """
  from inspect import isdatadescriptor

  results = {}
  if name is None:
    name = getattr(self, '__ui_name__', self.__class__.__name__.lower())
    results[None] = '{1} = {0.__class__.__name__}()'.format(self, name)
    add_to_imports(self, imports)

  # loop through dictionary attributes first.
  for key, value in self.__dict__.iteritems():
    if key[0] == '_': continue
    if exclude is not None and key in exclude: continue
    if hasattr(value, '__ui_repr__'): 
      default = None if defaults is None                                       \
                else defaults.__dict__.get(key, None)
      newname = name + '.' + key
      partial = value.__ui_repr__(imports, newname, default)
      results.update(partial) 
    else: 
      string = repr(value)
      if defaults is not None and key in defaults.__dict__                     \
         and type(value) is type(defaults.__dict__[key])                       \
         and string == repr(defaults.__dict__[key]): continue
      key = '{0}.{1}'.format(name, key)
      results[key] = string
      add_to_imports(string, imports)

  # then loops through class properties.
  for key in dir(self):
    if key[0] == '_': continue
    if key in self.__dict__: continue
    if exclude is not None and key in exclude: continue
    if not hasattr(self.__class__, key): continue
    value = getattr(self.__class__, key)
    if not isdatadescriptor(value): continue
    string = repr(getattr(self, key))
    if defaults is None or not hasattr(defaults.__class__, key): pass
    elif not isdatadescriptor(getattr(defaults.__class__, key)):   pass
    else: 
      default = getattr(defaults, key)
      if type(getattr(self, key)) is type(default) and repr(default) == string:
        continue
    key = '{0}.{1}'.format(name, key)
    results[key] = string
    add_to_imports(string, imports)
  return results
