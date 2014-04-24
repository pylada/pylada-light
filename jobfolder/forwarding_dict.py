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

""" Special dictioanry class to forward attributes indefinitely. """
__docformat__ = "restructuredtext en"
__all__ = ['ForwardingDict']
from collections import MutableMapping

class ForwardingDict(MutableMapping): 
  """ An *ordered* dictionary which forwards attributes.
  
      Overloads the ``.`` operator to access items for which the particular
      attribute exists. The return is another instance of
      :py:class:`ForwaringDict`. In this manner, attributes can chained with
      calls to the``.`` operator. The values of forwarded attributes can also
      be changed.
  """
  def __init__( self, ordered=True, readonly=None, naked_end=None,
                only_existing=None, _attr_list=None, dictionary=None):
    """ Initializes a ForwardingDict instance.
    
        :param bool readonly: 
           Whether or not the items in the dictionary can be modified. If None,
           defaults to :py:data:`pylada.jobparams_read_only`.
        :param bool naked_end:
           When only one item exists with the last forwarded attribute, whether
           it should be returned itself, or whether a
           :py:class:`ForwardingDict` should still be returned. Former is
           easier when examining objects interactively, latter is better in
           scripts. If None, defaults to :py:data:`pylada.jobparams_naked_end`.
        :param bool only_existing: 
           When setting attributes, whether to allow creation of new attributes
           for items which do not posses it. If None, defaults to
           :py:data:`pylada.jobparams_only_existing`
        :param _attr_list: 
           A list of strings making up the attributes to unroll. Private.
        :param dict dictionary:
           Initializes items from items in dictionary.
    """
    from .. import jobparams_naked_end, jobparams_only_existing, jobparams_readonly
    super(ForwardingDict, self).__init__()

    self.__dict__['readonly']      = jobparams_readonly if readonly is None else readonly
    """ Whether items can be modified in parallel using attribute syntax. """
    self.__dict__['naked_end']     = jobparams_naked_end if naked_end is None else naked_end
    """ Whether last item is returned as is or wrapped in ForwardingDict. """
    self.__dict__['only_existing'] = jobparams_only_existing if only_existing is None else only_existing
    """ Whether attributes can be added or only modified. """
    self.__dict__['_attr_list']    = [] if _attr_list is None else _attr_list
    """ List of attributes of attributes, from oldest parent to youngest grandkid. """
    self.__dict__['dictionary']    = {} if dictionary is None else dictionary
    """" The dictionary for which to unroll attributes. """

  @property
  def parent(self):
    """ Returns a ForwardingDict with parent items of self, eg unrolled once. """
    return self.copy(_attr_list=self._attr_list[:-1])

  @property
  def root(self):
    """ Returns a ForwardingDict with root grandparent. """
    return self.copy(_attr_list=[])

  @property
  def _attributes(self):
    """ Returns attributes special to this ForwardingDict. """
    from functools import reduce
    from itertools import chain

    result = set()
    attrs = len(self._attr_list) > 0
    for value in self.dictionary.itervalues():
      if attrs: value = reduce(getattr, chain([value], self._attr_list))
      result |= set(dir(value))
      
    return result

  def __getattr__(self, name):
    """ Returns a Forwarding dict with next requested attribute. """
    from functools import reduce
    from itertools import chain

    if name not in self._attributes:
      raise AttributeError( "Attribute {0} not found in {1} instance."\
                            .format(name, self.__class__.__name__) )
    attrs = len(self._attr_list) > 0
    result = self.copy(append=name)
    for key, value in self.dictionary.iteritems():
      if attrs: value = reduce(getattr, chain([value], self._attr_list))
      if not hasattr(value, name): del result[key]
    if self.naked_end and len(result.dictionary) == 1: return result[result.keys()[0]]
    if len(result.dictionary) == 0: 
      raise AttributeError( "Attribute {0} not found in {1} instance."\
                            .format(name, self.__class__.__name__) )
    return result

  def __setattr__(self, name, value):
    """ Forwards attribute setting. """
    from functools import reduce
    from itertools import chain
    # First checks for attribute in ForwardingDict instance.
    try: super(ForwardingDict, self).__getattribute__(name)
    except AttributeError: pass
    else: super(ForwardingDict, self).__setattr__(name, value); return

    # checks this dictionary is writable.
    if self.readonly: raise RuntimeError("ForwardingDict instance is read-only.")

    # Case with no attributes to unroll.
    found = False
    attrs = len(self._attr_list) > 0
    for item in self.dictionary.values():
      if attrs: # unroll attribute list.
        try: item = reduce(getattr, chain([item], self._attr_list))
        except AttributeError: continue
      if hasattr(item, name) or not self.only_existing:
        found = True
        setattr(item, name, value)
    if not found:
      raise AttributeError( "Attribute {0} not found in {1} instance."\
                            .format(name, self.__class__.__name__) )

  def __delattr__(self, name):
    """ Deletes an attribute or forwarded attribute. """
    from functools import reduce
    from itertools import chain

    try: super(ForwardingDict, self).__delattr__(name)
    except AttributeError: pass
    else: return
    
    if self.readonly: raise RuntimeError("ForwardingDict instance is read-only.")

    found = False
    attrs = len(self._attr_list) > 0
    for item in self.dictionary.values():
      if attrs:
        try: item = reduce(getattr, chain([item], self._attr_list))
        except AttributeError: continue
      if hasattr(item, name): 
        delattr(item, name)
        found = True
    if not found:
      raise AttributeError( "Attribute {0} not found in {1} instance."\
                            .format(name, self.__class__.__name__) )

  def __dir__(self):
    from itertools import chain
    results = chain( [u for u in self.__dict__ if u[0] != '_'], \
                     [u for u in dir(self.__class__) if u[0] != '_'], \
                     self._attributes )
    return list(set(results))


  def __getitem__(self, key):
    from functools import reduce
    from itertools import chain
    if len(self._attr_list) == 0: return self.dictionary[key]
    return reduce(getattr, chain([self.dictionary[key]], self._attr_list))
  def __setitem__(self, key, value):
    """ Add/modify item to dictionary.

        Items can be truly added only to root dictionary.
    """
    from functools import reduce
    from itertools import chain
    # root dictioanary.
    if len(self._attr_list) == 0: self.dictionary[key] = value; return
    # checks this is writable.
    if self.readonly: raise RuntimeError("This ForwardingDict is readonly.")
    if key not in self.dictionary:
      raise KeyError( "{0} is not in the ForwaringDict. Items "\
                      "cannot be added to a non-root ForwardingDict.".format(key))
    # non-root dict: must set innermost attribute.
    o = self.dictionary[key]
    if len(self._attr_list) > 1: 
      try: o = reduce(getattr, chain([o], self._attr_list[:-1]))
      except AttributeError:
        raise AttributeError( "Could not unroll list of attributes for object in {0}: {1}."\
                              .format(key, self._attr_list) )  
    if self.only_existing and not hasattr(o, self._attr_list[-1]):
      raise KeyError( "{0} cannot be set with current attribute list.\n{1}\n"\
                      .format(key, self._attr_list) )
    setattr(o, self._attr_list[-1], value)
  def __delitem__(self, key): 
    """ Removes item from dictionary. """
    o = self.dictionary[key]
    del self.dictionary[key]
    return o
  def __len__(self): return len(self.dictionary)
  def __contains__(self, key): return key in self.dictionary
  def __iter__(self): return self.dictionary.__iter__()
  def keys(self): return self.dictionary.keys()


  def __copy__(self):
    """ Returns a shallow copy of this object. """
    result = self.__class__()
    result.__dict__.update(self.__dict__)
    result.dictionary = self.dictionary.copy()
    return result

  def copy(self, append=None, dict=None, **kwargs):
    """ Returns a shallow copy of this object.
     
      :param str append : str or None
         Append value to a deepcopy of a list of attributes. Ignored if None.
      :param kwargs : dict
         Any other attribute to set in the ForwardingDict instance. Note
         that only attributes of the ForwardingDict instance are
         set/modified. This is not propagated to the object the dict holds.
    """
    from copy import copy, deepcopy
    result = copy(self)
    if append is not None and "_attr_list" in kwargs:
      raise ValueError( "Cannot copy attribute _attr_list as "\
                        "a keyword and as ``append`` simultaneously." )
    if 'dictionary' in kwargs: result.dictionary = kwargs.pop('dictionary').copy()
    for key, value in kwargs.iteritems():
      super(ForwardingDict, result).__setattr__(key, value)

    if append is not None:
      result._attr_list = deepcopy(self._attr_list)
      result._attr_list.append(append)

    return result


  def __str__(self): 
    """ Prints dictionary of unrolled values. """
    if len(self) == 0: return '{}'
    if len(self) == 1: return "{{'{0}': {1}}}".format(self.keys()[0], repr(self.values()[0]))
    string = "{\n"
    m = max(len(k) for k in self.keys())
    for k, v in self.iteritems():
      string += "  '{0}': {2}{1},\n".format(k, repr(v), "".join(" " for i in range(m-len(k))))
    return string + "}"
  def __repr__(self): return self.__str__()

  def __setstate__(self, state):
    """ Reloads the state from a pickle. 

        This is defined explicitely since otherwise, the call would go through
        __getattr__ and start an infinite loop.
    """
