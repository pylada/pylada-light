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

class BaseKeyword(object):
  """ Defines keyword input to different functionals. 
  
      The object is to make functionals act and behave as close as possible to
      the original input-file based approach, while allowing some automation.
      We want functional wrappers to both be able to read the original input
      files meaningfully, while also providing a more pythonic interface.
      

      At this juncture, it would seem that input files can be in some way
      understood as pairs of keywords and values. This class is the base of all
      such keyword-value pairs. The basic class contains little more than the
      ability to represent itself (:py:method:`__repr__`) and the to emit a
      dictionary representing the keyword-value pair (:py:method:`output_map`).

      .. note:: 

        More precisely, input files  are generally groups of keyword-value
        pairs. The groups are dealt with by
        :py:class:`~pylada.tools.block.AttrBlock`.
  """
  def __init__(self, keyword=None, raw=None):
    """ Creates a block. 

        :param str keyword:
          Keyword indicating the name of the block.
    """
    super(BaseKeyword, self).__init__()
    if keyword != None: 
      self.keyword = keyword
      """ Input keyword. """
    if raw is not None:
      self.raw = raw
      """ Extra input to keyword. """

  def __repr__(self): 
    """ Dumps representation to string. """
    args = []
    if 'keyword' in self.__dict__:
      args.append("keyword={0.keyword!r}".format(self))
    if 'raw' in self.__dict__: args.append("raw={0.raw!r}".format(self))
    return "{0.__class__.__name__}(".format(self) + ', '.join(args) + ')'

  def output_map(self, **kwargs):
    """ Keyword - Value dictionary
    
        This function returns a dictionary from which a text input file can be written.
        For instance, it could emit {'SIGMA': '0.6'}, which the
        :py:class:`~pylada.vasp.functional.Vasp` functional would then print to
        an INCAR file  as "SIGMA = 0.6", and the :py:class:`Crystal
        <pylada.dftcrystal.functional.Functional>` functional would render as
        "SIGMA\n0.6\n".
    """
    #print "tools/keywords: BaseKeyword.output: keyword: %s" \
    #  % (self.keyword,)
    if getattr(self, 'keyword', None) is None: return None
    return {self.keyword: getattr(self, 'raw', None)}

class ValueKeyword(BaseKeyword):
  """ Keywords which expect a value of some kind. 
  
      Instances of this class make it easy to declare and use CRYSTAL_ keywords
      which define a single value::

        functional.keyword = ValueKeyword(value=5)

      The above would print to the CRYSTAL_ input as follows:

        | KEYWORD
        | 5

      And by VASP_ as follows:

        | KEYWORD = 5

      The keyword can be then be given any value::

        functional.keyword = abc
      
      would result in:

        | KEYWORD
        | ABC

      In practice, the value is printed by first transforming it to a string
      via str_ and putting it in upper case.

      If given a string, Pylada will attempt to guess the type of the value,
      depending on whether it is an ``int``, ``float``, ``str``, or a list of
      the same, in that order. A list is created if and only if the string is
      does not contian multiple lines. If the string contains more than one
      line, it will always be kept as a string, so that complex pre-formatted
      input is not messed with. :py:meth:`~ValueKeyword.__set__`. Finally, the
      keyword will not appear in the input if its value is None. 

      .. _str: http://docs.python.org/library/functions.html#str
      .. CRYSTAL_: http://www.crystal.unito.it/
      .. VASP_: http://www.vasp.at/
  """
  def __init__(self, keyword=None, value=None):
    """ Initializes a keyword with a value. """
    super(ValueKeyword, self).__init__(keyword=keyword)
    #print "tools/keywords: ValueKeyword.const: keyword: %s  value: %s" \
    #  % (keyword, value,)
    self.value = value
    """ The value to print to input. 

        If None, then this keyword will not appear.
    """
  @property
  def raw(self):
    """ Returns raw value for CRYSTAL input. """
    if self.value == None: return '' # otherwise, fails to find attribute.
    return str(self.value) if not hasattr(self.value, '__iter__')              \
           else ' '.join(str(u) for u in self.value)
  @raw.setter
  def raw(self, value):
    """ Guesses value from raw input. """
    from ...error import ValueError
    if not isinstance(value, str):
      raise ValueError( 'Expected a string as input to {0}.raw.'               \
                        .format(self.keyword) )
    # split along line and remove empty lines
    lines = value.rstrip().lstrip().split('\n')
    # if only one line left, than split into a list and guess type of each
    # element.
    if len(lines) == 0:
      raise ValueError( 'Expected *non-empty* string as '                      \
                        'input to {0.keyword}.\n'.format(self) )
    elif len(lines) == 1:
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
    self.value = n
  def __get__(self, instance, owner=None):
    """ Returns value as is. """
    return self.value
  def __set__(self, instance, value):
    """ Assigns value as is. """
    self.value = value
  def __repr__(self):
    """ Dumps a representation of self. """
    args = self._addrepr_args()
    return '{0.__class__.__name__}({1})'.format(self, ', '.join(args))

  def _addrepr_args(self):
    from inspect import getargspec
    args = []
    if 'keyword' in self.__dict__:
      args.append('keyword={0.keyword!r}'.format(self))
    iargs = getargspec(self.__class__.__init__)
    doaddval = False
    if iargs.args is None or 'value' not in iargs.args: doaddval = True
    else: 
      i = iargs.args.index('value') - len(iargs.args)
      if iargs.defaults is None or -(i+1) >= len(iargs.defaults): doaddval = True
      else:
        default = iargs.defaults[i]
        if repr(default) != repr(self.value): doaddval = True
    if doaddval: args.append('value={0.value!r}'.format(self))
    return args

  def output_map(self, **kwargs):
    """ Map keyword, value """
    #print "tools/keywords: ValueKeyword.output: keyword: %s  value: %s" \
    #  % (self.keyword, self.value,)
    if self.value is None: return None
    return super(ValueKeyword, self).output_map(**kwargs)
  

class TypedKeyword(ValueKeyword):
  """ Keywords which expect a value of a given type.
  
      This specializes :py:class:`ValueKeyword` to accept only input of a given
      type.
  
      Two types are allowed: 
      
        - A simple type, eg int.
        - A list of types. Yes, a list, not a tuple or any other sequence.

            - only one item: the value can have any length, but the
              type of each element must conform. For example ``[int]`` will map
              "5 6 7" to ``[5, 6, 7]``. 
            - more than one item: the value is a list n items, where n is the
              size of the type. Each element in the value must conform to the
              respective element in the type. For example, ``[int, str]`` will
              map "5 two" to ``[5, "two"]. It will fail when mapping "5 6 7"
              (three elements) or "two 5" (wrong types).

              .. warning:: 

                 Types are checked if the value is set as a whole, not when a
                 single item is set.
     
      The value given will be cast to the given type, unless None, in which
      case the keyword will not be printed to the CRYSTAL input file.

      In practice, we can create the following mapping from python to the
      CRYSTAL_ input::

         functional.keyword = TypedValue([int, float, int])
         functional.keyword = [5, 6, 3]

      will yield:

         | KEYWORD
         | 5 6.0 3

      The following would throw an exception:

        >>> functional.keyword = ['a', 5.3, 5]
        ValueError: ...
        >>> functional.keyword = [0, 2.0]
        pylada.error.ValueError: Expected a sequence of the following type: [int, float, int]

      The last case failed because only two values are given. For a list of
      variable length (but only one type of lement), use::

        functional.keyword = TypedKeyword(type=[int])
        functional.keyword = [0, 1, 2]
        functional.keyword = [0, 1, 2, 3]

      The last two lines will not raise an exception. The CRYSTAL_ input would
      look like this:

        | KEYWORD
        | 0 1 2 3 

      Note that the first value is *not* the length of the list! If you need
      that behavior, see :py:class:`VariableListKeyword`. 

      .. note::

        Formally, the type need not be set in
        :py:method:`~TypedKeyword.__init__`. If it is left out (in which case
        it defaults to None), it is expected that it will be set later by the
        user, prior to use. 
        This is mainly to allow derived classes to define
        :py:attr:`~TypedKeyword.type` as a class attribute, rather than an
        instance attribute.
  """
  def __init__(self, keyword=None, type=None, value=None):
    """ Initializes a keyword with a value. """
    from ...error import ValueError
    super(TypedKeyword, self).__init__(keyword=keyword, value=None)
    #print "tools/keywords: TypedKeyword.const: keyword: %s  value: %s" \
    #  % (keyword, value,)
    if isinstance(type, list) and len(type) == 0:
      raise ValueError('type must be class or a non-empty list of classes')

    if type is not None:
      self.type = type
      """ Type to which the value should be cast if the value is not None. """

    self.value = value

  @property
  def value(self): 
    """ The value to print to input. 

        If None, then this keyword will not appear.
        Otherwise, it is cast to the type given when initializing this instance
        of :py:class:`~pylada.dftcrystal.input.TypedKeyword`
    """
    return self._value
  @value.setter
  def value(self, value):
    from ...error import ValueError
    if value is None: self._value = None; return
    if type(self.type) is list:
      if isinstance(value, str) and not isinstance(self.type, str):
        value = value.replace(',', ' ').replace(';', ' ').split()
      if not hasattr(value, '__iter__'): 
        raise ValueError( '{0} expected a sequence on input, got {1!r}.'       \
                          .format(self.keyword, value) ) 
      if len(self.type) == 1: 
        _type = self.type[0]
        try: self._value = [_type(u) for u in value]
        except Exception as e: raise ValueError(e)
        if len(self._value) == 0: self._value = None
      else: 
        if len(value) != len(self.type):
          raise ValueError( '{0.keyword} expected a sequence of the '          \
                            'following type: {0.type}'.format(self) )
        self._value = [t(v) for t, v in zip(self.type, value)]
    else:
      try: self._value = self.type(value)
      except Exception as e: raise ValueError(e)
  @property
  def raw(self):
    """ Returns raw value for CRYSTAL input. """
    if self._value == None: return '' # otherwise, fails to find attribute.
    if type(self.type) is list:
      return ' '.join(str(v) for v in self.value)
    return str(self._value)
  @raw.setter
  def raw(self, value):
    """ Guesses value from raw input. """
    if type(self.type) is list: value = value.split()
    self.value = value
  def __repr__(self):
    """ Dumps a representation of self. """
    args = self._addrepr_args()
    if 'type' in self.__dict__:
      if isinstance(self.type, list):
        args.append( 'type=[{0}]'                                              \
                     .format(', '.join(u.__name__ for u in self.type)) )
      else: args.append( 'type={0.type.__name__}'.format(self))
    return '{0.__class__.__name__}({1})'.format(self, ', '.join(args))

class VariableListKeyword(TypedKeyword):
  """ Keywords which expect a variable-length list value of a given type.
  
      Expects a  list of values of a given type.
      The length can be anything. However, unlike its base class
      :py:class:`TypedKeyword`, the fortran input
      :py:attr:`~VariableListKeyword.raw` should consist first of an integer
      giving the size of the list of values.

      Furthermore, the type of the element of the list should be given on
      initialization. Eg, ``type=int``, rather than, say, ``type=[int]``
  """
  def __init__(self, keyword=None, type=None, value=None):
    """ Initializes a keyword with a value. """
    super(VariableListKeyword, self).__init__( keyword=keyword, type=type,     \
                                               value=value )
  @property
  def value(self): 
    """ The value to print to input. 

        If None, then this keyword will not appear.
        Otherwise, it is cast to the type given when initializing this instance
        of :py:class:`~pylada.dftcrystal.input.TypedKeyword`
    """
    return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None; return
    if not hasattr(value, '__iter__'): 
      raise ValueError( '{0.keyword} expected a sequence on input.'            \
                        .format(self) )
    self._value = [self.type(u) for u in value]
    if len(self._value) == 0: self._value = None
  @property
  def raw(self):
    """ Returns raw value for CRYSTAL input. """
    if self._value == None: return '' # otherwise, fails to find attribute.
    lstr = ' '.join(str(v) for v in self.value) 
    return '{0}\n{1}'.format(len(self.value), lstr)
  @raw.setter
  def raw(self, value):
    """ Guesses value from raw input. """
    value = value.split()
    self.value = value[1:int(value[0])+1]
  def __getitem__(self, index):
    from ...error import IndexError
    if self.value is None:
      return IndexError('{0} is None.'.format(self.keyword))
    return self.value[index]
  def __setitem__(self, index, value):
    from ...error import IndexError
    if self.value is None:
      return IndexError('{0} is None.'.format(self.keyword))
    self.value[index] = value
  def __len__(self): return len(self.value)
  def __iter__(self): return self.value.__iter__()


class BoolKeyword(ValueKeyword):
  """ Boolean keyword.

      If True, the keyword is present.
      If False, it is not.
      This class uses the get/set mechanism to set whether the keyword should
      appear or not. It is meant to be used in conjunction with other linked keywords.
      Otherwise, it is simpler to use :py:meth:`self.add_keyword('something')
      <pylada.dftcrystal.input.AttrBlock.add_keyword>` directly.
  """
  def __init__(self, keyword=None, value=None):
    """ Initializes FullOptG keyword. """
    super(BoolKeyword, self).__init__(keyword=keyword, value=value)
  @property
  def value(self): return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None; return
    self._value = (value == True)
  def output_map(self, **kwargs):
    """ Map keyword, value """
    #print "tools/keywords: BoolKeyword.output: keyword: %s  value: %s" \
    #  % (self.keyword, self.value,)
    if self.value == False: return None
    return super(BoolKeyword, self).output_map(**kwargs)

class ChoiceKeyword(BaseKeyword):
  """ Keyword value must be chosen from a given set. """
  def __init__(self, values=None, value=None, keyword=None):
    """ Creates keyword which must be chosen from a given set. """ 
    super(ChoiceKeyword, self).__init__(keyword=keyword)
    if values is not None:
      self.values = list(values)
      """ Set of values from which to choose keyword. """
    self.value = value
    """ Current value. """
  @property
  def value(self):
    """ Current value of the keyword. """
    return self._value
  @value.setter
  def value(self, value):
    from ...error import ValueError
    if value is None: self._value = None; return
    if hasattr(value, 'rstrip'): value = value.rstrip().lstrip()
    if hasattr(value, 'lower'): value = value.lower()
    for v in self.values:
      try: o = v.__class__(value)
      except: pass
      if (hasattr(o, 'lower') and o.lower() == v.lower()) or o == v: 
        self._value = o
        return
    raise ValueError( '{0.keyword} accepts only one of the following: {1}'     \
                      .format(self, self.values) )
  def __get__(self, instance, owner=None):
    """ Function called by :py:class:`AttrBlock`. """
    return self._value
  def __set__(self, instance, value):
    """ Function called by :py:class:`AttrBlock`. """
    self.value = value

  @property
  def raw(self):
    if self._value == None: return '' # otherwise, fails to find attribute.
    return str(self.value)
  @raw.setter
  def raw(self, value): self.value = value
  def output_map(self, **kwargs):
    """ Map keyword, value """
    #print "tools/keywords: ChoiceKeyword.output: keyword: %s  value: %s" \
    #  % (self.keyword, self.value,)
    if self._value is None: return None
    if getattr(self, 'keyword', None) is None: return None
    return { self.keyword: str(self.value) }

  def _addrepr_args(self):
    from inspect import getargspec
    args = []
    if 'keyword' in self.__dict__:
      args.append('keyword={0.keyword!r}'.format(self))
    iargs = getargspec(self.__class__.__init__)
    doaddval = False
    if iargs.args is None or 'value' not in iargs.args: doaddval = True
    else: 
      i = iargs.args.index('value') - len(iargs.args)
      if iargs.defaults is None or 1-i < len(iargs.defaults): doaddval = True
      else:
        default = iargs.defaults[i]
        if repr(default) != repr(self.value): doaddval = True
    if doaddval: args.append('value={0.value!r}'.format(self))
    return args

  def __repr__(self): 
    """ Dumps representation to string. """
    args = self._addrepr_args()
    if 'values' in self.__dict__: args.append("values={0.values!r}".format(self))
    return "{0.__class__.__name__}({1})".format(self, ', '.join(args))
    
class QuantityKeyword(ValueKeyword):
  """ Keyword with a value which is signed by a unit. """
  def __init__(self, units=None, shape=None, keyword=None, value=None):
    """ Creates the quantity itself. """

    super(QuantityKeyword, self).__init__(keyword=keyword)
    if units is not None:
      self.units = units
      """ UnitQuantity to which values should be scaled. """
    if shape is not None:
      self.shape = shape
      """ Shape of input/output arrays. """
    self.value = value

  @property
  def value(self): 
    """ Value to which this keyword is set. """
    return self._value
  @value.setter
  def value(self, value):
    if value is None: self._value = None; return
    if hasattr(value, 'rescale'): value = value.rescale(self.units)
    else: value = value * self.units
    shape = getattr(self, 'shape', ())
    if value.shape != shape: value = value.reshape(shape)
    self._value = value
  @property
  def raw(self):
    """ Returns string for CRYSTAL input. """
    if self._value is None: return ''
    shape = getattr(self, 'shape', ())
    if len(shape) > 2: 
      from ...error import NotImplementedError
      raise NotImplementedError( 'Pylada does not know how to print n-d arrays ' \
                                 '(n>2) to CRYSTAL input.')
    if len(shape) == 0: return str(float(self.value))
    else:
      result = str(self.value.magnitude).replace('[', ' ').replace(']', ' ')
      result = result.split('\n')
      return '\n'.join(u.rstrip().lstrip() for u in result)
  @raw.setter
  def raw(self, value):
    """ Creates value from CRYSTAL input. """
    self.value = [float(u) for u in value.rstrip().split()]

  def __repr__(self):
    """ Dumps a representation of self. """
    args = []
    if 'keyword' in self.__dict__:
      args.append('keyword={0.keyword!r}'.format(self))
    if 'shape' in self.__dict__: 
      args.append('shape={0.shape!r}'.format(self))
    if 'units' in self.__dict__: 
      args.append('units={0.units!r}'.format(self))
    if len(getattr(self, 'shape', ())) > 0: 
      args.append('value={0!r}'.format(self.value.magnitude))
    elif self.value is not None: 
      args.append('value={0}'.format(float(self.value)))
    return '{0.__class__.__name__}({1})'.format(self, ', '.join(args))


class AliasKeyword(ValueKeyword):
  """ Accepts a number of aliases for the same output. """
  def __init__(self, aliases=None, value=None, keyword=None):
    """ Initializes a an AliasKeyword """
    super(AliasKeyword, self).__init__(value=value, keyword=keyword)
    #print "tools/keywords: AliasKeyword.const: keyword: %s  value: %s" \
    #  % (keyword, value,)
    if aliases is not None:
      self.aliases = aliases
      """ Mapping from many aliases to same output. """
  @property
  def value(self):
    if self._value is None: return None
    return self.aliases[self._value][0]
  @value.setter
  def value(self, value):
    from itertools import chain
    from ...error import ValueError
    if value is None: 
      self._value = None
      return
    for key, items in self.aliases.iteritems():
      for item in chain(items, [key]):
        try: 
          if item.__class__(value) == item:
            self._value = key
            return
        except: continue
    raise ValueError( 'Incorrect value ({1}) for keyword {0}'                 \
                       .format(self.keyword, value) )
  def output_map(self, **kwargs):
    """ Returns output map. """
    #print "tools/keywords: AliasKeyword.output: keyword: %s  value: %s" \
    #  % (self.keyword, self.value,)
    if self._value is None: return None
    if getattr(self, 'keyword', None) is None: return None
    return {self.keyword: str(self._value)}
  def __repr__(self):
    args = self._addrepr_args()
    if 'aliases' in self.__dict__:
      args.append('aliases={0.aliases!r}'.format(self))
    return '{0.__class__.__name__}({1})'.format(self, ', '.join(args))
