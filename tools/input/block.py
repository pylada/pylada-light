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

from .keywords import BaseKeyword
class AttrBlock(BaseKeyword):
  """ Defines block input to CRYSTAL. 
  
      A block is a any set of input which starts with a keyword and ends with
      an END. It can contain other sub-keywords.
      This particular flavor creates attributes from the  inner input keywords.
      This supposes that each keyword is only ever inputed once. 

      It can contain subitems.
  """
  def __init__(self, keyword=None, raw=None):
    """ Creates a block. 

        :param str keyword:
          Keyword indicating the name of the block.
    """
    # first add this to avoid infinite recursion from redifining __setattr__
    self.__dict__['_input'] = {}

    # then call base constructor
    super(AttrBlock, self).__init__(keyword=keyword, raw=raw)

    # now so we get doctstrings right.
    self._input = {}
    """ Dictionary of crystal inputs. """
    

  def __getattr__(self, name):
    """ passes through the input keywords in :py:attr:`_input`. """
    from ...error import AttributeError
    if name not in self._input: 
      raise AttributeError('Unknown attribute {0}.'.format(name))
    result = self._input[name]
    return result.__get__(self) if hasattr(result, '__get__') else result
  def __setattr__(self, name, value):
    """ passes through the input keywords in :py:attr:`_input`. 
    
        If the input value is derived from
        :py:class:`~pylada.tools.keyword.BaseKeyword`, then it is added to
        :py:attr:`_input`. Otherwise, super is called.
    """
    if isinstance(value, BaseKeyword): 
      self._input[name] = value
      if not hasattr(value, 'keyword'): self._input[name].keyword = name
    elif name in self._input:
      result = self._input[name]
      if hasattr(result, '__set__'): result.__set__(self, value)
      else: self._input[name] = value
    else: super(AttrBlock, self).__setattr__(name, value)
  def __delattr__(self, name):
    """ passes through the input keywords in :py:attr:`_input`.  """
    if name in self._input: del self._input[name]
    else: super(AttrBlock, self).__delattr__(name)
  def __dir__(self):
    """ List of attributes and members. """
    return list( set(self._input.iterkeys())                                   \
                 | set(self.__dict__.iterkeys())                               \
                 | set(dir(self.__class__)) )

  def add_keyword(self, name, value=None):
    """ Adds/Sets input keyword. """
    # if known keyword, then go through setattr mechanism.
    # this makes sure we recognize the type of value and the already registered
    # keyword.
    if name in self._input:  setattr(self, name, value)
    # if value is None, then transform it to True. 
    # This is a keyword which is either there or not there, like EXTPRT.
    elif value is None: self._input[name] = True
    # boolean case
    elif value is True or value is False:
      self._input[name] = value
    # if a string, tries to guess what it is.
    elif isinstance(value, str):
      # split along line and remove empty lines
      lines = value.rstrip().lstrip().split('\n')
      # if only one line left, than split into a list and guess type of each
      # element.
      if len(lines) == 1 and len(lines[0]) > 0:
        lines = lines[0]
        n = []
        for u in lines.split():
          try: v = int(u)
          except:
            try: v = float(u)
            except: v = u
          n.append(v)
        # if made up of string, then go back to string.
        if all(isinstance(u, str) for u in n): n = [lines]
        # if only one element use that rather than list
        if len(n) == 1: n = n[0]
      # if empty string, set to True
      elif len(lines) == 1: n = True
      # if multiple line, keep as such
      else: n = value
      self._input[name] = n
    # otherwise, just set the keyword.
    else: self._input[name] = value
    # return self to allow chaining calls.
    return self

  def __repr__(self, defaults=False, name=None):
    """ Representation of this instance. """
    from ..uirepr import uirepr
    defaults = self.__class__() if defaults else None
    return uirepr(self, name=name, defaults=defaults)

  def __ui_repr__(self, imports, name=None, defaults=None, exclude=None):
    """ Creates user friendly representation. """
    from ..uirepr import template_ui_repr, add_to_imports

    results = template_ui_repr(self, imports, name, defaults, exclude)
    if name is None:
      name = getattr(self, '__ui_name__', self.__class__.__name__.lower())

    for key, value in self._input.iteritems():
      if exclude is not None and key in exclude: continue
      if hasattr(value, '__ui_repr__'): 
        default = None if defaults is None                                     \
                  else defaults._input.get(key, None)
        newname = name + '.' + key
        partial = value.__ui_repr__(imports, newname, default)
        results.update(partial)
        if newname in results:              doinit = False
        elif default is None:               doinit = True
        else: doinit = type(value) is not type(default)
        if doinit:
          results[newname] = '{0.__class__.__name__}()'.format(value)
          add_to_imports(value, imports)
      elif isinstance(value, BaseKeyword):
        value = getattr(self, key)
        string = repr(value)
        if defaults is not None and key in defaults._input                     \
           and type(value) is type(getattr(defaults, key))                     \
           and string == repr(getattr(defaults, key)): continue
        key = '{0}.{1}'.format(name, key) 
        results[key] = string
        add_to_imports(value, imports)
      elif value is None:
        if defaults is not None and key in defaults._input                     \
           and defaults._input[key] is None: continue
        results['{0}.add_keyword({1!r})'.format(name, key)] = None
      else:
        if defaults is not None and key in defaults._input                     \
           and type(value) is type(defaults._input[key])                       \
           and repr(value) == repr(defaults._input[key]): continue
        results['{0}.add_keyword({1!r}, {2!r})'.format(name, key, value)]      \
            = None
        add_to_imports(value, imports)
    
    return results
  
  def __getstate__(self):
    d = self.__dict__.copy()
    crysinput = d.pop('_input')
    return d, crysinput
  def __setstate__(self, value):
    self.__dict__['_input'] = value[1]
    self.__dict__.update(value[0])



  def output_map(self, **kwargs):
    """ Map of keyword, value """
    from .tree import Tree

    # At this point kwargs is a map with entries:
    #   'vasp': vasp/functional.Vasp.__init__: species:  None
    #       vasp/functional.Vasp.__init__: kpoints:  None
    #       vasp/functional.Vasp.__init__: kwargs:  {}
    #       keywords: IStruct.init for CONTCAR: value: auto
    #       from pylada.vasp.relax import Relax
    #       from quantities.quantity import Quantity
    #       relax = Relax()
    #       relax.addgrid        = True
    #       relax.ediff          = 6e-05
    #       relax.encut          = 0.9
    #       ...
    #   'comm': None,
    #   'overwrite': False,
    #   'structure': Structure( 2.64707e-16, 2.1615, 2.1615,
    #       2.1615, 1.32354e-16, 2.1615,
    #       2.1615, 2.1615, 0,
    #       scale=1, name='icsd_633029.cif' )\
    #      .add_atom(0, 0, 0, 'Fe')\
    #      .add_atom(2.1615, 2.1615, 2.1615, 'O'),
    #  'outdir': '/tmp/temp.test/icsd_633029/icsd_633029.cif/non-magnetic/relax_cellshape/0'

    root = Tree()
    result = root if getattr(self, 'keyword', None) is None \
             else root.descend(self.keyword)
    for key, value in self._input.iteritems():
      self._output_map(result, key, value, **kwargs)
    if len(result) == 0: return None
    return root


  @staticmethod
  def _output_map(_tree, _key, _value, **kwargs):
    """ Modifies output tree for given keyword/value. """
    if _value is None: return False
    elif isinstance(_value, bool): _tree[_key] = _value 
    elif hasattr(_value, 'output_map'):
      dummy = _value.output_map(**kwargs)
      if dummy is None: return False
      _tree.update(dummy)
    elif getattr(_value, 'raw', None) is not None:
      _tree[_key] = str(_value.raw)
    elif hasattr(_value, '__iter__'):
      _tree[_key] =  ' '.join(str(u) for u in _value)
    else: _tree[_key] = str(_value)
    return True

  def read_input(self, tree, owner=None, **kwargs):
    """ Sets object from input tree. """
    from ...error import internal, IndexError
    from .tree import Tree
    for key, value in tree.iteritems():
      key = key.lower()
      try: 
        if isinstance(value, Tree) and key not in self._input:
          if hasattr(self, '_read_nested_group'): 
            dummy = self._read_nested_group(value, owner=self, **kwargs)
            self.add_keyword(key, dummy)
          else: raise IndexError('Unknown group {0}'.format(key))
        if key in self._input:
          newobject = self._input[key]
          if hasattr(newobject, 'read_input'):
            newobject.read_input(value, owner=self, **kwargs)
          elif hasattr(self._input[key], 'raw'):
            newobject.raw = value
          elif hasattr(self._input[key], '__set__'):
            newobject.__set__(self, value)
          else: raise internal('Cannot read INCAR for {0}'.format(key))
        else: self.add_keyword(key.lower(), value)
      except:
        from sys import exc_info
        type, value, traceback = exc_info()
        message = 'ERROR when reading {0}.'.format(key)
        if value is None: type.args = type.args, message
        else: value = value, message
        raise type, value, traceback
