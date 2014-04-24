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

""" Miscellaneous ressources. """
__all__ = [ 'bugLev', 'testValidProgram', 'copyfile', 'Changedir',
            'read_input', 'exec_input', 'load',
            'RelativePath', 'LockFile', 'open_exclusive', 'translate_to_regex',
            'mkdtemp', 'Redirect' ]

from types import ModuleType

from changedir import Changedir
from relativepath import RelativePath
from lockfile import LockFile, open_exclusive

bugLev = 0
"""
global debug level
"""

def setBugLev( lev):
  global bugLev
  bugLev = lev



testValidProgram = None
"""
Validation test program name
"""

def setTestValidProgram( pgm):
  import os
  global testValidProgram
  if pgm == None: testValidPgm = None
  else: testValidProgram = os.path.expanduser( pgm)



def _copyfile_impl(src, dest):
  """ Copies files by hand. 
      
      Makes sure that files are actually copied to disk, as opposed to
      buffered. Does not check for existence or anything.
  """
  from os import stat, fsync

  stepsize = 2**20
  size = stat(src).st_size
  if size == 0: 
    with open(dest, 'wb') as outfile: pass
    return

  with open(dest, 'wb') as outfile:
    with open(src, 'rb') as infile:
      steps = [stepsize] * (size // stepsize)
      if size % stepsize != 0: steps += [size % stepsize]
      for step in steps:
        buffer = infile.read(step)
        if buffer is None: break
        outfile.write(buffer)
    # makes sure stuff is written to disk prior to returning.
    outfile.flush()
    fsync(outfile.fileno())

def copyfile(src, dest=None, nothrow=None, symlink=False, aslink=False, nocopyempty=False):
  """ Copy ``src`` file onto ``dest`` directory or file.

      :param src:
          Source file.
      :param dest: 
          Destination file or directory.
      :param nothrow:
          Throwing is disable selectively depending on the content of nothrow:

            - exists: will not throw is src does not exist.
            - isfile: will not throw is src is not a file.
            - same: will not throw if src and dest are the same.
            - none: ``src`` can be None.
            - null: ``src`` can be '/dev/null'.
            - never: will never throw.

      :param symlink:
          Creates link rather than actual hard-copy. Symlink are
          created with relative paths given starting from the directory of
          ``dest``.  Defaults to False.
      :param aslink: 
          Creates link rather than actual hard-copy *if* ``src`` is
          itself a link. Links to the file which ``src`` points to, not to
          ``src`` itself. Defaults to False.
      :parma nocopyempty:
          Does not perform copy if file is empty. Defaults to False.

      This function fails selectively, depending on what is in ``nothrow`` list.
  """
  try:
    from os import getcwd, symlink as ln, remove
    from os.path import isdir, isfile, samefile, exists, basename, dirname,\
                        join, islink, realpath, relpath, getsize, abspath
    # sets up nothrow options.
    if nothrow is None: nothrow = []
    if isinstance(nothrow, str): nothrow = nothrow.split()
    if nothrow == 'all': nothrow = 'exists', 'same', 'isfile', 'none', 'null'
    nothrow = [u.lower() for u in nothrow]
    # checks and normalizes input.
    if src is None: 
      if 'none' in nothrow: return False
      raise IOError("Source is None.")
    if dest is None: dest = getcwd()
    if dest == '/dev/null': return True
    if src  == '/dev/null':
      if 'null' in nothrow: return False
      raise IOError("Source is '/dev/null' but Destination is {0}.".format(dest))

    # checks that input source file exists.
    if not exists(src): 
      if 'exists' in nothrow: return False
      raise IOError("{0} does not exist.".format(src))
    src = abspath(realpath(src))
    if not isfile(src):
      if 'isfile' in nothrow: return False
      raise IOError("{0} is not a file.".format(src))
    # makes destination a file.
    if exists(dest) and isdir(dest): dest = join(dest, basename(src))
    # checks if destination file and source file are the same.
    if exists(dest) and samefile(src, dest): 
      if 'same' in nothrow: return False
      raise IOError("{0} and {1} are the same file.".format(src, dest))
    if nocopyempty and isfile(src):
      if getsize(src) == 0: return
    if aslink and islink(src): symlink, src = True, realpath(src)
    if symlink:
      if exists(dest): remove(dest)
      src = realpath(abspath(src))
      dest = realpath(abspath(dest))
      if relpath(src, dirname(dest)).count("../") == relpath(src, '/').count("../"):
        ln(src, realpath(dest))
      else:
        with Changedir(dirname(dest)) as cwd:
           ln(relpath(src, dirname(dest)), basename(dest))
    else: _copyfile_impl(src, dest)
  except:
    if 'never' in nothrow: return False
    raise
  else: return True

class Input(ModuleType):
  """ Fake class which will be updated with the local dictionary. """
  def __init__(self, name = "pylada_input"): 
    """ Initializes input module. """
    super(Input, self).__init__(name, "Input module for pylada scripts.")
  def __getattr__(self, name):
    raise AttributeError( "All out of cheese!\n"
                          "Required input parameter '{0}' not found in {1}." \
                          .format(name, self.__name__) )
  def __delattr__(self, name):
    raise RuntimeError("Cannot delete object from input namespace.")
  def __setattr__(self, name, value):
    raise RuntimeError("Cannot set/change object in input namespace.")
  def update(self, other):
    if hasattr(other, '__dict__'): other = other.__dict__
    for key, value in other.items():
      if key[0] == '_': continue
      super(Input, self).__setattr__(key, value)
  @property
  def __all__(self):
    return list([u for u in self.__dict__.iterkeys() if u[0] != '_'])
  def __contains__(self, name):
    return name in self.__dict__

def read_input(filename='input.py', global_dict=None, local_dict = None, paths=None, comm = None):
  """ Reads and executes input script and returns local dictionary (as namespace instance). """
  from os.path import exists, basename
  assert exists(filename), IOError('File {0} does not exist.'.format(filename))
  with open(filename, 'r') as file: string = file.read()
  return exec_input(string, global_dict, local_dict, paths, basename(filename))

def exec_input( script, global_dict=None, local_dict=None,
                paths=None, name=None ):
  """ Executes input script and returns local dictionary (as namespace instance). """
  # stuff to import into script.
  from os import environ
  from os.path import abspath, expanduser
  from math import pi 
  from numpy import array, matrix, dot, sqrt, abs, ceil
  from numpy.linalg import norm, det
  from .. import crystal
  from . import Input
  import quantities
  
  if bugLev >= 5: print "misc/init: exec_input: entry"
  # Add some names to execution environment.
  if global_dict is None: global_dict = {}
  global_dict.update( { "environ": environ, "pi": pi, "array": array, "matrix": matrix, "dot": dot,
                        "norm": norm, "sqrt": sqrt, "ceil": ceil, "abs": abs,  "det": det,
                        "expanduser": expanduser, "load": load })
  for key, value in quantities.__dict__.iteritems():
    if key[0] != '_' and key not in global_dict:
      global_dict[key] = value
  for key in crystal.__all__: global_dict[key] = getattr(crystal, key)
  if local_dict is None: local_dict = {}
  # Executes input script.
  if bugLev >= 5:
    print 'misc/init: exec_input: ========== start script =========='
    print script
    print 'misc/init: exec_input: ========== end script =========='
  exec(script, global_dict, local_dict)

  # Makes sure expected paths are absolute.
  if paths is not None:
    for path in paths:
      if path not in local_dict: continue
      local_dict[path] = abspath(expanduser(local_dict[path]))
    
  if name is None: name = 'None'
  result = Input(name)
  result.update(local_dict)
  return result

def load(data, *args, **kwargs):
  """ Loads data from the data files. """
  from os import environ
  from os.path import dirname, exists, join
  if "directory" in kwargs: 
    raise KeyError("directory is a reserved keyword of load")

  # find all possible data directories
  directories = []
  if "data_directory" in globals():
    directory = globals()["data_directory"]
    if hasattr(directory, "__iter__"): directories.extend(directory)
    else: directories.append(directory)
  if "PYLADA_DATA_DIRECTORY" in environ:
    directories.extend(environ["PYLADA_DATA_DIRECTORY"].split(":"))

  # then looks for data file.
  if data.rfind(".py") == -1: data += ".py"
  for directory in directories:
    if exists(join(directory, data)):
      kwargs["directory"] = dirname(join(directory, data))
      result = {}
      execfile(join(directory, data), {}, result)
      return result["init"](*args, **kwargs)
  raise IOError("Could not find data ({0}).".format(data))

def add_setter(method, docstring = None): 
  """ Adds an input-like setter property. """
  def _not_available(self): raise RuntimeError("Error: No cheese available.")
  if docstring is None and hasattr(method, "__doc__"): docstring = method.__doc__
  return property(fget=_not_available, fset=method,  doc=docstring)

def import_dictionary(self, modules=None):
  """ Creates a dictionary of import modules. """
  if modules is None: modules = {}
  avoids = ['__builtin__', 'quantities.quantity']
  if self.__class__.__module__ not in avoids:
    if self.__class__.__module__ not in modules:
      modules[self.__class__.__module__] = set([self.__class__.__name__])
    else:
      modules[self.__class__.__module__].add(self.__class__.__name__)
  if not hasattr(self, '__dict__'): return modules
  for value in self.__dict__.itervalues():
    class_, module_ = value.__class__.__name__, value.__class__.__module__
    if module_ in avoids: continue
    if module_ in modules: modules[module_].add(class_)
    else: modules[module_] = set([class_])
  return modules

def import_header_string(modules):
  """ Creates string from dictionary of import modules. """
  result = ''
  for key, values in modules.iteritems():
    result += "from {0} import {1}\n".format(key, ", ".join(values))
  return result

def translate_to_regex(pat):
  """ Translates a pattern from unix to re. 

      Compared to fnmatch.translate, doesn't use '.', but rather '[^/]'.
      And doesn't add the tail that fnmatch.translate does.
      Otherwise, code is taked from fnmatch.translate.
  """
  from re import escape
  i, n = 0, len(pat)
  res = ''
  while i < n:
      c = pat[i]
      i = i+1
      if c == '*':
          res = res + '[^/]*'
      elif c == '?':
          res = res + '[^/]'
      elif c == '[':
          j = i
          if j < n and pat[j] == '!':
              j = j+1
          if j < n and pat[j] == ']':
              j = j+1
          while j < n and pat[j] != ']':
              j = j+1
          if j >= n:
              res = res + '\\['
          else:
              stuff = pat[i:j].replace('\\','\\\\')
              i = j+1
              if stuff[0] == '!':
                  stuff = '^' + stuff[1:]
              elif stuff[0] == '^':
                  stuff = '\\' + stuff
              res = '{0}[{0}]'.format(res, stuff)
      else:
          res = res + escape(c)
  return res 

def latest_file(*args):
  """ Path of latest file.

      Check each argument if it exists and is non-empty and is a file. If there
      are more than one, returns the latest. If there are none, returns None.


      :param *args:
         Path to files for which to perform comparison.

      :returns: path to the latest file or None
  """
  from os.path import exists, getsize, isfile
  from os import stat
  from operator import itemgetter
  if len(args) == 0: return None
  dummy = []
  for filename in args: 
    path = RelativePath(filename).path
    if not exists(path): continue
    if not isfile(path): continue
    if getsize(path) == 0: continue
    dummy.append((path, stat(path).st_mtime))
  if len(dummy) == 0: return None
  dummy = sorted(dummy, key=itemgetter(1))
  return dummy[-1][0]

def mkdtemp(suffix='', prefix='', dir=None):
    """ Creates and returns temporary directory. 
        
	Makes it easier to get all Pylada tmp directories in the same place,
	while retaining a certain amount of flexibility when on a
	supercomputer.  It first checks for a PBS_TMPDIR ernvironment variable.
	If that does not exist, then it checks for a PYLADA_TMPDIR environment
	variable. If that does not exist, it checks wether
	:py:data:`~pylada.global_tmpdir` is not None. If that does not exist,
        then it uses the directory provided in the input. 

	Once ``dir`` has been determined, it calls python's mkdtemp.

	:param suffix: A suffix to the temporary directory
	:param prefix: A prefix to the temporary directory. 
	:param dir: Last alternative for root of tmp directories.
    """
    from os import environ
    from tempfile import mkdtemp as pymkdtemp
    from datetime import datetime
    from .. import global_tmpdir
    rootdir = environ.get( 'PBS_TMPDIR',
                           environ.get('PYLADA_TMPDIR', global_tmpdir) )
    if rootdir is None: rootdir = dir
    rootdir = RelativePath(rootdir).path
    if len(prefix) == 0: prefix = str(datetime.today())
    else: prefix = '{0}_{1}'.format( str(datetime.today()).replace(' ', '-'),
                                     prefix )
    return pymkdtemp(prefix=prefix, suffix=suffix, dir=rootdir)

class Redirect:
  """ Redirects python input, output, error. 
  
  
      Usage is as follows:

      :code-block: python

        with Redirect('something.out', ['out', 'err']):
          print 'something'

      The above will redirect the python output and error to 'something.out'
      until the close of the ``with`` statement.
  """
  def __init__(self, filename, units='out', append=False):
    """ Creates a redirection context. """
    from collections import Sequence
    from ..error import input as InputError
    units = set(units) if isinstance(units, Sequence) else set([units])
    if len(units - set(['in', 'out', 'err'])) != 0:
      raise InputError('Redirect: input should be one of "in", "out", "err".')
    self.units = units
    self.filename = filename
    self.append = append

  def __enter__(self):
    from os.path import abspath
    import sys
    self.old = {}
    if 'in' in self.units: self.old['in'] = sys.stdin
    if 'out' in self.units: self.old['out'] = sys.stdout
    if 'err' in self.units: self.old['err'] = sys.stderr
    self.file = open( self.filename if len(self.filename) else "/dev/null",
                      "a" if self.append else "w" )
    if 'in' in self.units: sys.stdin = self.file
    if 'out' in self.units: sys.stdout = self.file
    if 'err' in self.units: sys.stderr = self.file
    return abspath(self.file.name)

  def __exit__(self, *wargs):
    import sys 
    if 'in' in self.units and 'in' in self.old: sys.stdin = self.old.pop('in')
    if 'err' in self.units and 'err' in self.old: sys.stderr = self.old.pop('err')
    if 'out' in self.units and 'out' in self.old: sys.stdout = self.old.pop('out')
    self.file.close()
    del self.old
    del self.file
