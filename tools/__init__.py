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

""" Miscellaneous ressources to create functionals. """
__docformat__ = "restructuredtext en"
__all__ = [ 'stateless', 'assign_attributes', 'check_success',
            'check_success_generator', 'make_cached', 'uncache', 'SuperCall',
            'create_directory', 'prep_symlink', 'add_pyladarunning_marker',
            'remove_pyladarunning_marker', 'remove_workdir_link',
            'add_section_to_file', 'get_section_from_file', 'OnFinish' ]
def stateless(function):
  """ Decorator to make a function stateless.
  
      Deepcopies structure and self parameters.
      Also sets outdir to getcwd if it is None.
  """
  from functools import wraps

  @wraps(function)
  def wrapper(self, structure, outdir=None, **kwargs ):
    from copy import deepcopy
    from os import getcwd
    from ..misc import RelativePath
    structure = deepcopy(structure)
    self      = deepcopy(self)
    outdir    = getcwd() if outdir is None else RelativePath(outdir).path
    return function(self, structure, outdir, **kwargs)
  return wrapper

def assign_attributes(setothers=None, ignore=None):
  """ Decorator to assign keywords to attributes. """
  from functools import wraps
  if setothers == None: setothers = []
  if ignore == None: ignore = set()

  @wraps(assign_attributes)
  def decorator(function):
    """ Decorator to assign keywords to attributes. """
    @wraps(function)
    def wrapper(self, structure, outdir=None, comm=None, **kwargs):
      # if other keyword arguments are present, then they are assumed to be
      # attributes of self, with value to be changed before launch. 
      for key in kwargs.keys():
        if key in ignore: continue
        # direct attributes.
        if hasattr(self, key): setattr(self, key, kwargs.pop(key))
        else:
          found = False
          for other in setothers:
            if hasattr(self, other) and hasattr(getattr(self, other), key):
              setattr(getattr(self, other), key, kwargs.pop(key))
              found = True
              break
          if found == False:
            raise ValueError( "Unkwown keyword argument to {0.__class__.__name__}: {1}"\
                              .format(self, key) )
      return function(self, structure, outdir=outdir, comm=comm, **kwargs)
    return wrapper
  return decorator

def check_success(function):
  """ Decorator to check for success prior to running functional. """
  from functools import wraps
  @wraps(function)
  def wrapper(self, *args, **kwargs):
    # Checks for previous run, or deletes previous run if requested.
    if not kwargs.get('overwrite', False):
      extract = self.Extract(outcar = kwargs['outdir'])
      if extract.success: return extract # in which case, returns extraction object.
    return function(self, *args, **kwargs)
  return wrapper

def check_success_generator(function):
  """ Decorator to check for success prior to running functional. 
  
      Generator version. Yields stuff.
  """
  from functools import wraps
  @wraps(function)
  def wrapper(self, *args, **kwargs):
    # Checks for previous run, or deletes previous run if requested.
    if not kwargs.get('overwrite', False):
      extract = self.Extract(outcar = kwargs['outdir'])
      if extract.success: yield extract # in which case, returns extraction object.
      return 
    for n in function(self, *args, **kwargs): yield n
  return wrapper

def make_cached(method):
  """ Caches the result of a method for futur calls. """
  from functools import wraps

  @wraps(method)
  def wrapped(*args, **kwargs):
    from pylada.misc import bugLev
    if bugLev >= 5:
      print 'tools/init make_cached entry: method: %s' % (method.__name__,)
    if not hasattr(args[0], '_properties_cache'): 
      setattr(args[0], '_properties_cache', {}) 
    cache = getattr(args[0], '_properties_cache')
    if method.__name__ not in cache:
      cache[method.__name__] = method(*args, **kwargs)
      if bugLev >= 5:
        print 'tools/init make_cached: set method: %s' % (method.__name__,)
    else:
      if bugLev >= 5:
        print 'tools/init make_cached: use method: %s' % (method.__name__,)
    return cache[method.__name__]
  return wrapped

def uncache(ob):
  """ Uncaches results cached by @make_cached. """ 
  if hasattr(ob, '_properties_cache'): del ob._properties_cache


class SuperCall(object):
  """ Obviates issues when using a "super" functional.

      Since functionals of a job-folder are deepcopied, the following line
      will not result in calling the next class in the __mro__.
  
      >>> jobfolder.functional = super(Functional, functional)

      Indeed, this line will first call the __getitem__, __setitem__ (or
      __deepcopy__) of the super object. In general, this means we end-up with
      ``jobfolder.function == functional``.

      This class obviates this difficulty.

      >>> jobfolder.functional = SuperCall(Functional, functional)
  """
  def __init__(self, class_, object_):
    object.__init__(self)
    self.__dict__['_class'] = class_
    self.__dict__['_object'] = object_
  def __call__(self, *args, **kwargs):
    return super(self._class, self._object).__call__(*args, **kwargs)
  def __getattr__(self, name):
    try: return getattr(super(self._class, self._object), name)
    except: return getattr(self._object, name)
  def __setattr__(self, name, value): setattr(self._object, name, value)
  def __getstate__(self): return self._class, self._object
  def __setstate__(self, args):
    self.__dict__['_class'] = args[0]
    self.__dict__['_object'] = args[1]
  def __dir__(self): return dir(super(self._class, self._object))
  def __repr__(self): return repr(super(self._class, self._object))
  def copy(self):
    """ Performs deepcopy of self. """
    from copy import deepcopy
    class_ = deepcopy(self.__dict__['_class'])
    object_= deepcopy(self.__dict__['_object'])
    return self.__class__(class_, object_)


def create_directory(directory):
  """ If directory does not exist, creates it. """
  from os import makedirs
  from os.path import exists
  if not exists(directory): makedirs(directory)

def prep_symlink(outdir, workdir, filename=None):
  """ Creates a symlink between outdir and workdir.

      If outdir and workdir are the same directory, then bails out.
      Both directories should exist prior to call.
      If filename is None, then creates a symlink to workdir in outdir called
      ``workdir``. Otherwise, creates a symlink in workdir called filename.
      If a link ``filename`` already exists, deletes it first.
  """
  from os import remove, symlink
  from os.path import samefile, lexists, abspath, join
  from ..misc import Changedir
  if samefile(outdir, workdir): return
  if filename is None:
    with Changedir(workdir) as cwd: 
      if lexists('workdir'):
        try: remove('workdir')
        except OSError: pass
      try: symlink(abspath(workdir), abspath(join(outdir, 'workdir')))
      except OSError: pass
    return

  with Changedir(workdir) as cwd: 
    if lexists(filename):
      try: remove(filename)
      except OSError: pass
    try: symlink( abspath(join(outdir, filename)),
                  abspath(join(workdir, filename)) )
    except OSError: pass

def remove_workdir_link(outdir):
  """ Removes link from output to working directory. """
  from os.path import exists, join
  from os import remove
  path = join(outdir, 'workdir')
  if exists(path): 
    try: remove(path)
    except OSError: pass

def add_pyladarunning_marker(outdir): 
  """ Creates a marker file in output directory. """
  from os.path import join
  from pylada.misc import bugLev
  file = open(join(outdir, '.pylada_is_running'), 'w')
  file.close()
  if bugLev >= 5:
    print 'tools/init: add_run_mark: is_run outdir: %s' % (outdir,)
def remove_pyladarunning_marker(outdir): 
  """ Creates a marker file in output directory. """
  from os.path import exists, join
  from os import remove
  path = join(outdir, '.pylada_is_running')
  if exists(path): 
    try: remove(path)
    except OSError: pass
  if bugLev >= 5:
    print 'tools/init: rem_run_mark: is_run outdir: %s' % (outdir,)

def add_section_to_file(outdir, filename, marker, string, append=True):
  """ Appends a string to an output file. 

      The string will be added with some simple marking. 

        | #################### MARKER ####################
        | string
        | #################### END MARKER ####################

      The output directory should exist on call. The string is stripped first.
      If the resulting string is empty, it is not added to the file.
  """
  from os.path import join

  string = string.rstrip().lstrip()
  if len(string) == 0: return
  header = ''.join(['#']*20)
  with open(join(outdir, filename), 'a' if append else 'w') as file:
    file.write('{0} {1} {0}\n'.format(header, marker.upper()))
    file.write(string)
    file.write('\n{0} END {1} {0}\n'.format(header, marker.upper()))

def get_section_from_file(stream, marker):
  """ Returns the content of a section. 
      
      A section in a file is defined as:

        | #################### MARKER ####################
        | string
        | #################### END MARKER ####################

      This function returns string.
  """
  header = ''.join(['#']*20)
  startmarker = '{0} {1} {0}\n'.format(header, marker.upper())
  endmarker = '{0} END {1} {0}\n'.format(header, marker.upper())
  found = False
  for line in stream:
    if line == startmarker: found = True; break
  if not found: return ""
  result = ""
  for line in stream: 
    if line == endmarker: break
    result += line
  return result

class OnFinish(object):
  """ Called when a run finishes. 
     
      Makes sure bringdown is called at the end of a call to a functional.
      This object should be used as input to a
      :py:class:`~lada.process.program.ProgramProcess` instance's onfinish
      option.
  """
  def __init__(self, this, *args):
    self.this = this
    self.args = args
  def __call__(self, *args, **kwargs):
    self.this.bringdown(*self.args)
