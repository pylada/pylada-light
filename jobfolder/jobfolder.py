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

""" Submodule declaring the folder folders class. """
__docformat__ = "restructuredtext en"

class JobFolder(object):
  """ High-throughput folder class. 
  
      Means to organize any calculations in folders and subfolders. A folder
      is executable is the ``functional`` attribute is not ``None``. The
      attribute should be set to a pickleable calleable. The parameters for the
      calls should be inserted in the ``params`` attribute. Sub-folders can be
      added using the :py:meth:`__div__` and :py:meth:`__setitem__`
      methods. The latter offers the ability to access and set subfolders at
      any point within the tree of folders from any subfolder. The executable
      subfolders can also be iterated in a manner similar to a job-dictionary.
      Finally, a folder can be executed `via` the :py:meth:`compute` method. 
  """
  def __init__(self):
    super(JobFolder, self).__init__()
    # List of subfolders (as in subdirectories). 
    super(JobFolder, self).__setattr__("children", {})
    # This particular folder. 
    super(JobFolder, self).__setattr__("params", {})
    # This particular folder is not set. 
    super(JobFolder, self).__setattr__("_functional", None)
    # Parent folder. 
    super(JobFolder, self).__setattr__("parent", None)

  @property
  def functional(self):
    """ Returns current functional.
    
        The functional is implemented as a property to make sure that it is
        either None or a pickleable callable. The functional is **deepcopied**
        from the input. In other words, this functional stored in the
        folderdictionary is no longuer the one given on input -- it is not a
        reference to the input. This parameter can never be truly deleted.

        >>> del folder.functional 

        is equivalent to:
        
        >>> folder.functional = None

        .. note:: To store a reference to a global functional, one could do
                  ``folder._functional = functional`` instead. However, modifying
                  the input functional will affect the stored functional and
                  vice-versa.
    """
    return self._functional


  @functional.setter
  def functional(self, value):
    from pickle import dumps, loads # ascertains pickle-ability, copies functional
    from pylada.misc import bugLev

    if value is not None and not hasattr(value, "__call__"):
      raise ValueError("folder.functional should be either None(no job) or a callable.")
    # ascertains pickle-ability
    try: string = dumps(value)
    except Exception as e:
      raise ValueError(
        "Could not pickle functional. Caught Error:\n{0}".format(e))

    if bugLev >= 1:
      print 'jobfolder.functional.setter for name: ', self.name
    try: self._functional = loads(string)
    except Exception as e:
      raise ValueError("Could not reload pickled functional. Caught Error:\n{0}".format(e))


  @functional.deleter
  def functional(self): self._functional = None


  @property
  def name(self):
     """ Returns the name of this dictionary as an absolute path. """
     if self.parent is None: return "/"
     string = None
     for key, item in self.parent.children.iteritems():
       if id(item) == id(self):
         string = self.parent.name + key
         break
     if string is None: raise RuntimeError("Could not determine the name of the dictionary.")
     return string + '/'


  @property
  def is_executable(self):
    """ True if functional is not None. """
    return self.functional is not None


  @property
  def untagged_folders(self):
    """ Returns a string with only untagged folders. """
    result = "Folders: \n"
    for name, folder in self.iteritems():
      if not folder.is_tagged: result += "  " + name + "\n"
    return result


  @property
  def is_tagged(self):
    """ True if current folder is tagged. 

        In practice, this is used to turn a folder *on* (untagged) or
        *off* (tagged). The meaning of *tagged* is not enforced, so it could be
        used for other purposes.
    """
    return hasattr(self, "_tagged")


  @property
  def nbfolders(self):
    """ Returns the number of folders in sub-tree. """
    return len([0 for j, o in self.iteritems()])


  @property 
  def root(self): 
    """ Returns root dictionary. """
    result = self
    while result.parent is not None: result = result.parent
    return result


  def __getitem__(self, index): 
    """ Returns folder description from the dictionary.

        If the folder does not exist, will create it.
    """
    from re import split
    from os.path import normpath

    index = normpath(index)
    if index == "" or index is None or index == ".": return self
    if index[0] == "/": return self.root[index[1:]]

    result = self
    names = split(r"(?<!\\)/", index)
    for i, name in enumerate(names):
      if name == "..":
        if result.parent is None: raise KeyError("Cannot go below root level.")
        result = result.parent
      elif name in result.children: result = result.children[name]
      else: raise KeyError("folder " + index + " does not exist.")
    return result
 

  def __delitem__(self, index): 
    """ Returns folder description from the dictionary.

        If the folder does not exist, will create it.
    """
    from os.path import normpath, relpath

    index = normpath(index)

    try: deletee = self.__getitem__(index) # checks if exists.
    except KeyError: raise

    if isinstance(deletee, JobFolder): 
      if id(self) == id(deletee): raise KeyError("Will not commit suicide.")
      parent = self.parent
      while parent is not None: 
        if id(parent) == id(deletee): raise KeyError("Will not go Oedipus on you.")
        parent = parent.parent

    parent = self[index+"/.."]
    name = relpath(index, index+"/..")
    if name in parent.children:
      if id(self) == id(parent.children[name]): raise KeyError("Will not delete self.")
      return parent.children.pop(name)
    raise KeyError("folder " + index + " does not exist.")


  def __setitem__(self, name, value): 
    """ Sets folder/subfolder description in the dictionary.
    
        If the folder does not exist, will create it.  A deepcopy_ of
        value is inserted, rather than a simple shallow ref.

        .. _deepcopy: http://docs.python.org/library/copy.html#copy.deepcopy
    """
    from copy import deepcopy
    from os.path import normpath, dirname, basename

    index = normpath(name)
    parentpath, childpath = dirname(index), basename(index)
    if len(parentpath) != 0: 
      if parentpath not in self:
        raise KeyError('Could not find parent folder {0}.'.format(parentpath))
      mother = self[parentpath]
      parent = self.parent
      while parent is not None:
        if parent is mother: raise KeyError('Will not set parent folder of current folder.')
    if len(childpath) == 0 or childpath == '.': raise KeyError('Will not set current directory.')
    if childpath == '..': raise KeyError('Will not set parent directory.')

    parent = self if len(parentpath) == 0 else self[parentpath]
    parent.children[childpath] = deepcopy(value)
    parent.children[childpath].parent = parent


  def __div__(self, name): 
    """ Adds a folderdictionary to the tree. 

        Any *path* can be given as input. This is akin to doing `mkdir -p`.
        The newly created folder folders is returned.
    """
    from re import split
    from os.path import normpath

    index = normpath(name)
    if index in ["", ".", None]: return self
    if index[0] == "/":  # could create infinit loop.
      result = self
      while result.parent is not None: result = result.parent
      return result / index[1:]

    names = split(r"(?<!\\)/", index) 
    result = self
    for name in names:
      if name == "..":
        if result.parent is None:
          raise RuntimeError('Cannot descend below root.')
        result = result.parent
        continue
      elif name not in result.children:
        result.children[name] = JobFolder()
        result.children[name].parent = result
      result = result.children[name]
    return result


  def subfolders(self):
    """ Sorted keys of the folders directly under this one. """
    return sorted(self.children.iterkeys())
    

  def compute(self, **kwargs):
    """ Executes the functional in this particular folder.
    
        If this particular folder of the folder folders is not executable (e.g.
        ``self.functional is None``), then ``None`` is returned.

        If, on the other hand, this folder contains a real functional, then the
        latter is called taking the parameters stored in the folder as keyword
        arguments. Futhermore, additional keyword arguments passed to this
        method are passed on the functional, possibly overriding those stored
        in the folder. The return from the functional is returned by this
        method: In practice the call is as follows:

        >>> return self.functional(**self.params.copy().update(kwargs))
    """  
    from pylada.misc import bugLev

    if not self.is_executable: return None
    params = self.params.copy()
    params.update(kwargs)
    if bugLev >= 1:
      print 'jobfolder.compute: self: ', self
      print 'jobfolder.compute: kwargs: ', kwargs
      print 'jobfolder.compute: params: ', params
      print 'jobfolder.compute: ===== start self.functional ====='
      print self.functional
      print 'jobfolder.compute: ===== end self.functional ====='
      print 'jobfolder.compute: type(self.functional): ', type(self.functional)
      print 'jobfolder.compute: before call'

    # This calls the dynamically compiled code
    # created by tools/makeclass: create_call_from_iter
    res = self.functional.__call__(**params)
    if bugLev >= 1:
      print 'jobfolder.compute: after call'

    return res


  def update(self, other, merge=False):
    """ Updates folder and tree with other.
    
        :param other:
             :py:class:`JobFolder` dictionary from which to update.
        :param bool merge:
             If false (default), then actual folders in ``other`` completely
             overwrite actual folders in ``self``. If False, then ``params`` in
             ``self`` is updated with ``params`` in ``other`` if either one is
             an executable folder. If ``other`` is an executable folder, then ``functional`` in
             ``self`` is overwritten. If ``other`` is not an executable folder, then
             ``functional`` in ``self`` is not replaced.

        Updates the dictionaries of parameters and sub-folders. Actual folders in
        ``other`` (eg with ``self.is_executable==True``) will completely overwrite those in
        ``self``.  if items in ``other`` are found in ``self``, unless merge is
        set to true. This function is recurrent: subfolders are also updated.
    """
    for key, value in other.children.iteritems():
      if key in self: self[key].update(value)
      else: self[key] = value

    if not merge:
      if not other.is_executable: return
      self.params = other.params
      self.functional = other.functional
    else:
      if not (self.is_executable or other.is_executable): return
      self.params.update(other.params)
      if other.functional is not None: self.functional = other.functional

  def __str__(self):
    result = "Folders: \n"
    for name in self.iterkeys():
      result += "  " + name + "\n"
    return result

  def tag(self):
    """ Tags this folder. """
    if self.is_executable: super(JobFolder, self).__setattr__("_tagged", True)
    
  def untag(self):
    """ Untags this folder. """
    if hasattr(self, "_tagged"): self.__delattr__("_tagged")

  def __delattr__(self, name):
    """ Deletes folder attribute. """
    if name in self.__dict__: return self.__dict__.pop(name)
    if name in self.params: return self.params.pop(name)
    raise AttributeError("Unknown folder attribute " + name + ".")

  def __getattr__(self, name):
    """ Returns folder parameter.
    
        Folder parameters stored in :py:attr:`Jobdict.params` can also be accessed
        via the ``.`` operator.
    """
    if name in self.params: return self.params[name]
    raise AttributeError("Unknown folder attribute " + name + ".")

  def __setattr__(self, name, value):
    """ Sets folder parameter. 

        Folder parameters stored in :py:attr:`Jobdict.params` can also be accessed
        via the ``.`` operator.
    """
    from pickle import dumps
    if name in self.params:
      try: dumps(value)
      except Exception as e:
        raise ValueError("Could not pickle folder-parameter. Caught error:\n{0}".format(e))
      else: self.params[name] = value
    else: super(JobFolder, self).__setattr__(name, value)

  def __dir__(self):
    from itertools import chain
    result = chain([u for u in self.__dict__ if u[0] != '_'], \
                   [u for u in dir(self.__class__) if u[0] != '_'], \
                   [u for u in self.params.iterkeys() if u[0] != '_'])
    return list(set(result))

  def __getstate__(self):
    d = self.__dict__.copy()
    params = d.pop("params")
    return d, params
  def __setstate__(self, args):
    super(JobFolder, self).__setattr__("params", args[1])
    d = self.__dict__.update(args[0])

  def iteritems(self, prefix=''):
    """ Iterates over executable sub-folders.

        Iterates over all executable subfolders. A subfolder is executable if it
        holds a functional to execute.

        :param str prefix:
          Prefix to add to the name of this folder. Convenient when iterating
          over a folder folders with the intention of executing the folders it
          contains.

        :return: yields (directory, folder):

          - name of this folder, prefixed with ``prefix``.
          - folder is an executable :py:class:`Folderdict`.
    """
    from os.path import join
    # Yield this folder if it exists.
    if self.is_executable: yield prefix, self
    # Walk throught children folderdict.
    for name in self.subfolders():
      for u in self[name].iteritems(join(prefix, name)): yield u

  def iterleaves(self):
    """ Iterates over end of sub-trees. """
    # Yield this folder if it exists.
    if len(self.children) == 0: yield self.name
    # Walk throught children folderdict.
    for name in self.children:
      for u in self[name].iterleaves(): yield u

  def itervalues(self): 
    """ Iterates over all executable sub-folders. """
    for name, folder in self.iteritems(): yield folder
  def iterkeys(self): 
    """ Iterates over names of all executable subfolders. """
    for name, folder in self.iteritems(): yield name
  def values(self):
    """ List of all executable sub-folders. """
    return [u for u in self.itervalues()]
  def keys(self):
    """ List of names of all executable sub-folders. """
    return [u for u in self.iterkeys()]
  def items(self):
    """ List of all folders. """
    return [u for u in self.iteritems()]
  __iter__ = iterkeys
  """ Iterator over keys. """


  def __contains__(self, index):
    """ Returns true if index a branch in the folder folders. """
    from re import split
    from os.path import normpath
    index = normpath(index)
    if index == '/': return True
    if index[0] == '/': return index[1:] in self.root
    names = split(r"(?<!\\)/", index) 
    if len(names) == 0: return False
    if len(names) == 1: return names[0] in self.children
    if names[0] not in self.children: return False
    new_index = normpath(index[len(names[0])+1:])
    if len(new_index) == 0: return True
    return new_index in self[names[0]]

  def __copy__(self):
    """ Performs a shallow copy of this folder folders.

        Shallow copies are made of all internal dictionaries children and
        params. However, functional and params values should the same
        object as self. The sub-branches of the returned dictionary are shallow
        copies of the sub-branches of self. In other words, the functional and
        refences in params dictionary are in common between result and self,
        but nothing else.

        The returned dictionary does not have a parent!
    """
    from copy import copy
    result = JobFolder()
    result._functional = self._functional
    result.params   = self.params.copy()
    result.parent     = None
    for name, value in self.children.items():
      result.children[name] = copy(value)
      result.children[name].parent = result
    attrs = self.__dict__.copy()
    attrs.pop('params')
    attrs.pop('parent')
    attrs.pop('children')
    attrs.pop('_functional')
    result.__dict__.update(attrs)
    return result
