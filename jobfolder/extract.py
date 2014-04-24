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

""" Classes to manipulate output from job folders. """
__docformat__ = "restructuredtext en"
__all__ = ['AbstractMassExtract']

from abc import ABCMeta, abstractmethod

class AbstractMassExtract(object): 
  """ Collects extraction methods from different job-folders. 
  
      Wraps around a root job folder and provides means to access it (or
      something related to it). In practice, a derived class will hold a list
      of *somethings* which does something good for a particular folder. This
      is a base class, concerned mostly with providing a rich mapping and
      attribute access interface. It allows the user to focus on a small set of
      executable folders `via` the mapping (``[]``) methods, e.g. a view of the
      folders. The attributes of the wrapped *somethings* of the current view
      are retrieved into a :py:class:`forwarding dict
      <pylada.jobfolder.forwardingdict.ForwardingDict`. 
      
      The :py:meth:`__iter_alljobs__` method should be implemented within
      derived classes. It should yield for each executable folder a tuple
      consisting of the name of that folder and the relevant *something*.
  """
  __metaclass__ = ABCMeta

  def __init__(self, path=None, view=None, excludes=None, dynamic=False, ordered=True, 
               naked_end=None, unix_re=True):
    """ Initializes extraction object. 

        :param str path:
            Root directory for which to investigate all subdirectories.
            If None, uses current working directory.
        :param str view:
            Pattern which the job names must match to be included in the
            extraction. Ignored if None.
        :para excludes:
            List of patterns which the job names must *not* match to be
            included in the extraction. Ignored if None.
        :param bool dynamic:
            If true, chooses a slower but more dynamic caching method. Only
            usefull for ipython shell. 
        :param bool ordered: 
            If true, uses OrderedDict rather than conventional dict.
        :param bool naked_end:
            True if should return value rather than dict when only one item.
        :param bool unix_re: 
            Converts regex patterns from unix-like expression.
    """
    from .. import jobparams_naked_end, unix_re
    from ..misc import RelativePath
    from .ordered_dict import OrderedDict

    super(AbstractMassExtract, self).__init__()

    # this fools the derived classes' __setattr__
    self.__dict__.update({'dicttype': dict, 'view': '/', 'naked_end': naked_end,
                          'unix_re': unix_re, '_excludes': excludes, 
                          '_cached_extractors': None, 'dynamic': dynamic })
    self.naked_end = jobparams_naked_end if naked_end is None else naked_end
    """ If True and dict to return contains only one item, returns value itself. """
    self.unix_re = unix_re
    """ If True, then all regex matching is done using unix-command-line patterns. """
    self.excludes = excludes
    """ Patterns to exclude. """
    self._cached_extractors = None
    """ List of extration objects. """
    self.dynamic = dynamic
    """ If True chooses a slower but more dynamic caching method. """
    self.dicttype = OrderedDict if ordered else dict
    """ Type of dictionary to use. """
    if path is None: self.__dict__['_rootpath'] = None
    else: self.__dict__['_rootpath']= RelativePath(path, hook=self.uncache)

  @property
  def rootpath(self): 
    """ Root of the directory-tree to trawl for OUTCARs. """
    return self._rootpath.path if self._rootpath is not None else None
  @rootpath.setter
  def rootpath(self, value):
    from ..misc import RelativePath
    if self._rootpath is None:
      self._rootpath = RelativePath(path=value, hook=self.uncache)
    else: self._rootpath.path = value

  def uncache(self): 
    """ Uncache values. """
    self._cached_extractors = None

  @property 
  def excludes(self):
    """ Pattern or List of patterns to ignore. or None.

        :py:attr:`unix_re` determines whether these are unix-command-line like
        patterns or true python regex.
    """ 
    try: return self._excludes 
    except AttributeError: return None
  @excludes.setter
  def excludes(self, value):
    if isinstance(value, str): self._excludes = [value]
    else: self._excludes = value

  def avoid(self, excludes):
    """ Returns a new object with further exclusions. 

        :param excludes: Pattern or patterns to exclude from output.
        :type excludes: str or list of str or None 
          
        The goal of this function is to work as an *anti* operator [], i.e. by
        excluding from the output anything that matches the patterns, rather
        including only those which match the pattern.
        This is strickly equivalent to:

        >>> other = massextract.copy(excludes=excludes)
        >>> other.excludes.extend(massextract.excludes)

        and then doing calculations with ``other``. The advantage is that it
        can all be done on one line.

        If the ``excludes`` argument is None or an empty list, then the
        returned object will not exlude anything.
    """ 
    if excludes is None or len(excludes) == 0: return self.shallow_copy(excludes=None)
    result = self.shallow_copy(excludes=excludes)
    if self.excludes is not None: result.excludes.extend(self.excludes)
    return result

  def iteritems(self):
    """ Iterates through all extraction objects and names. """
    for name, job in self._regex_extractors(): yield name, job
  def items(self):
    """ Iterates through all extraction objects and names. """
    return [(name, job) for name, job in self.iteritems()]
    
  def itervalues(self):
    """ Iterates through all extraction objects. """
    for name, job in self._regex_extractors(): yield job
  def values(self):
    """ Iterates through all extraction objects. """
    return [job for job in self.itervalues()]

  def iterkeys(self):
    """ Iterates through all extraction objects. """
    for name, job in self._regex_extractors(): yield name
  def keys(self):
    """ Iterates through all extraction objects. """
    return [name for name in self.iterkeys()]
  
  def __iter__(self):
    """ Iterates through all job names. """
    for name, job in self.iteritems(): yield name
  def __len__(self): 
    """ Returns length of output dictionary. """
    return len(self.keys())

  def __contains__(self, key):
    """ Returns True if key is valid and not empty. """
    from re import compile
    rekey = compile(key)
    for key in self.iterkeys(): 
      if rekey.match(key): return True
    return False

  def _regex_pattern(self, pattern, flags=0):
    """ Returns a regular expression. """
    from re import compile
    from ..misc import translate_to_regex
    if self.unix_re: pattern = translate_to_regex(pattern)
    if len(pattern) == 0: return compile("", flags)
    if pattern[-1] in ('/', '\Z', '$'): return compile(pattern, flags)
    return compile(pattern + r"(?=/|\Z)(?ms)", flags)

  @abstractmethod
  def __iter_alljobs__(self):
    """ Generator to go through all relevant jobs. 
    
        :return: (name, extractor), where name is the name of the job, and
          extractor an extraction object.
    """
    pass

  @property
  def _extractors(self):
    """ Goes through all jobs and collects Extract if available. """
    if self.dynamic:
      if self._cached_extractors is None: self._cached_extractors = self.dicttype()
      result = self.dicttype()
      for name, extract in self.__iter_alljobs__():
        if name not in self._cached_extractors: self._cached_extractors[name] = extract
        result[name] = self._cached_extractors[name]
      return result
    else:
      if self._cached_extractors is not None: return self._cached_extractors
      result = self.dicttype()
      for name, extract in self.__iter_alljobs__(): result[name] = extract
      self._cached_extractors = result
      return result

  def _regex_extractors(self):
    """ Loops through jobs in this view. """
    if self.excludes is not None:
      excludes = [self._regex_pattern(u) for u in self.excludes]
    if self.view == "/": 
      for key, value in self._extractors.iteritems():
        if self.excludes is not None                                           \
           and any(u.match(key) is not None for u in excludes):
          continue
        yield key, value
      return

    regex = self._regex_pattern(self.view)
    for key, value in self._extractors.iteritems():
      if regex.match(key) is None: continue
      if self.excludes is not None                                             \
         and any(u.match(key) is not None for u in excludes):
        continue
      yield key, value

  @property
  def _attributes(self): 
    """ Returns __dir__ special to the extraction itself. """
    results = set([])
    for key, value in self.iteritems():
      results |= set([u for u in dir(value) if u[0] != '_'])
    return results

  def __dir__(self): 
    from itertools import chain
    results = chain( [u for u in self.__dict__ if u[0] != '_'], \
                     [u for u in dir(self.__class__) if u[0] != '_'], \
                     self._attributes )
    return list(set(results))

  def __getattr__(self, name): 
    """ Returns extracted values. """
    from .forwarding_dict import ForwardingDict
    assert name in self._attributes, AttributeError("Unknown attribute {0}.".format(name))

    result = self.dicttype()
    for key, value in self.iteritems():
      try: result[key] = getattr(value, name)
      except: result.pop(key, None)
    if self.naked_end and len(result) == 1: return result[result.keys()[0]]
    return ForwardingDict(dictionary=result, naked_end=self.naked_end)

  def __getitem__(self, name):
    """ Returns a view of the current job-dictionary.
    
        .. note:: normpath_ returns a valid path when descending below
           root, e.g.``normpath('/../../other') == '/other'), so there won't be
           any errors on that account.

           .. _normpath: http://docs.python.org/library/os.path.html#os.path.normpath
    """
    from os.path import normpath, join
    if name[0] != '/': name = join(self.view, name)
    if self.unix_re: name = normpath(name)
    return self.shallow_copy(view=name)

  def __delitem__(self, name):
    """ Removes items from the collection path. 

        This basically adds to the excludes attributes.
    """ 
    if self.excludes is None: self._excludes = [name]
    elif name not in self.excludes: self.excludes.append(name)

  def __getstate__(self): 
    d = self.__dict__.copy()
    return d

  def __setstate__(self, arg):
    self.__dict__.update(arg)
       
  def shallow_copy(self, **kwargs):
    """ Returns a shallow copy. 
    
        :param kwargs:  Any keyword attribute will modify the corresponding
          attribute of the copy.
    """
    from copy import copy
    result = copy(self)
    for key, value in kwargs.iteritems(): setattr(result, key, value)
    return result

  def iterfiles(self, **kwargs):
    """ Iterates over output/input files. 

        This is rerouted to all extraction objects.
    """
    for job in self.itervalues(): 
      if hasattr(job, 'iterfiles'): 
        for file in job.iterfiles(**kwargs): yield file 

  def __getstate__(self):
    d = self.__dict__.copy()
    if d["_rootpath"] is not None: d["_rootpath"].hook = None
    return d

  def __setstate__(self, arg):
    self.__dict__.update(arg)
    if self._rootpath is not None: self._rootpath.hook = self.uncache
