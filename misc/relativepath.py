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

""" Defines a path given relative to another. 

    The object is to make it easy to switch from one computer to another, using
    environment variables defined in both.
"""

class RelativePath(object):
  """ Directory property which is relative to the user's home.
  
      The path which is returned (eg __get__) is always absolute. However,
      it is stored relative to the user's home, and hence can be passed from
      one computer system to the next.
      Unless you know what you are doing, it is best to get and set using the
      ``path`` attribute, starting from the current working directory if a
      relative path is given, and from the '/' root if an absolute path is
      given.

      >>> from os import getcwd, environ
      >>> getcwd()
      '/home/me/inhere/'
      >>> relative_directory.path = 'use/this/attribute'
      >>> relative_directory.path
      '/home/me/inhere/use/this/attribute'

      Other descriptors have somewhat more complex behaviors. ``envvar`` is the
      root directory - aka the fixed point. Changing it will simply change the
      root directory.

      >>> environ["SCRATCH"]
      '/scratch/me/'
      >>> relative_directory.envvar = "$SCRATCH"
      >>> relative_directory.envvar
      '/scratch/me/use/this/attribute'

      Modifying ``relative`` will change the second part of the relative
      directory. If a relative path is given, that relative path is used as is,
      without reference to the working directory. It is an error to give an
      absolute directory.

      >>> relative_directory.relative = "now/here"
      '/scratch/me/now/here'
      >>> relative_directory.relative = "/now/here"
      ValueError: Cannot set relative with absolute path. 

  """
  def __init__(self, path=None, envvar=None, hook=None):
    """ Initializes the relative directory. 
    
        :Parameters:
          path : str or None
            path to store here. It can be relative to the current working
            directory, include envirnonment variables or shorthands for user
            homes. If None, will be set to `envvar`.
          envvar : str or None 
            Fixed point wich can be understood from system to system. It should
            be a shorthand to a user homer directory ("~/") or use an
            environment variable ("$SCRATCH"). If None, defaults to user's
            home.
          hook : callable or None
            This function will be called if/when the directory is changed. Note
            that it may be lost during pickling if it is not itself pickelable.
    """
    super(RelativePath, self).__init__()

    self._relative = None
    """ Private path relative to fixed point. """
    self._envvar = None
    """ Private envvar variable. """
    self._hook = None
    """ Private hook variable. """
    self.path = path
    """ Relative path. """
    self.envvar = envvar
    """ Fixed point. """
    self.hook = hook
    """ An object to call when the path is changed.
    
        Callable with at most one argument.
    """

  @property
  def relative(self):
    """ Path relative to fixed point. """
    return self._relative if self._relative is not None else ""
  @relative.setter
  def relative(self, value):
    """ Path relative to fixed point. """
    from os.path import expandvars, expanduser
    if value is None: value = ""
    value = expandvars(expanduser(value.rstrip().lstrip()))
    assert value[0] != '/', ValueError('Cannot set "relative" attribute with absolute path.')
    self._relative = value if len(value) else None
    self.hook(self.path)

  @property 
  def envvar(self):
    """ Fixed point for relative directory. """
    from os import getcwd
    from os.path import expanduser, expandvars, normpath
    from . import Changedir
    from .. import global_root
    if self._envvar is None:
      if global_root is None: return '/'
      if '$' not in global_root and '~' not in global_root:
        return normpath(global_root)
      # Need to figure it out. 
      try:
         with Changedir(expanduser(global_root)) as pwd: return getcwd()
      except OSError as e: 
        raise IOError( 'Could not figure out directory {0}.\n'
                       'Caught error OSError {1.errno}: {1.message}'
                       .format(global_root, e) )
    return normpath(expandvars(expanduser(self._envvar)))
  @envvar.setter
  def envvar(self, value):

    path = self.path if self._relative is not None else None
    if value is None: self._envvar = None
    elif len(value.rstrip().lstrip()) == 0: self._envvar = None
    else: self._envvar = value
    if path is not None: self.path = path
    self.hook(self.path)

  @property 
  def path(self):
    """ Returns absolute path, including fixed-point. """
    from os.path import join, normpath
    if self._relative is None: return self.envvar
    return normpath(join(self.envvar, self._relative))
  @path.setter
  def path(self, value):
    from os.path import relpath, expandvars, expanduser, abspath
    from os import getcwd
    if value is None: value = getcwd()
    if isinstance(value, tuple) and len(value) == 2: 
      self.envvar = value[0]
      self.relative = value[1]
      return
    if len(value.rstrip().lstrip()) == 0: value = getcwd()
    # This is a python bug where things don't work out if the root path is '/'.
    # Seems corrected after 2.7.2
    if self.envvar == '/': 
      self._relative = abspath(expanduser(expandvars(value)))[1:]
    else:
      self._relative = relpath(expanduser(expandvars(value)), self.envvar) 
    self.hook(self.path)

  @property
  def unexpanded(self):
    """ Unexpanded path (eg with envvar as is). """
    from os.path import join
    from .. import global_root
    e = global_root if self._envvar is None else self._envvar
    return e if self._relative is None else join(e, self._relative)

  @property
  def hook(self):
    from inspect import ismethod, getargspec
    if self._hook is None: return lambda x: None
    N = len(getargspec(self._hook)[0])
    if ismethod(self._hook): N -= 1
    if N == 0: return lambda x: self._hook()
    return self._hook
  @hook.setter
  def hook(self, value): 
    from inspect import ismethod, getargspec, isfunction

    if value is None: 
      self._hook = None
      return
    assert ismethod(value) or isfunction(value), \
           TypeError("hook is not a function or bound method.")
    N = len(getargspec(value)[0])
    if ismethod(value):
      assert value.im_self is not None,\
             TypeError("hook callable cannot be an unbound method.")
      N -= 1
    assert N < 2, TypeError("hook callable cannot have more than one argument.")
    self._hook = value
  def __getstate__(self):
    """ Saves state. 

        If hook was not pickleable, then it will not be saved appropriately.
    """
    from pickle import dumps
    try: dumps(self._hook)
    except: return self._relative, self._envvar
    else:   return self._relative, self._envvar, self._hook
  def __setstate__(self, args):
    """ Resets state. 

        If hook was not pickleable, then it will not be reset.
    """
    if len(args) == 3: self._relative, self._envvar, self._hook = args
    else: self._relative, self._envvar = args

  def set(self, path=None, envvar=None):
    """ Sets path and envvar.

        Used by repr.
    """
    hook = self._hook
    self._hook = None
    self.envvar = envvar
    self.path = path
    self._hook = hook
    self.hook(self.path)

  def repr(self):
    """ Makes this instance somewhat representable. 

        Since hook cannot be represented in most cases, and is most-likely set
        on initialization, this method uses ``set`` to get away with
        representability.
    """
    return "{0}, {1}".format(repr(self._envvar), repr(self._relative))


