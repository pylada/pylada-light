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

from pylada.error import IOError, ValueError
class NodeFileExists(IOError):
  """ Thrown when nodefile already exists. """
  pass
class MPISizeError(ValueError):
  """ Thrown when too few/many processes are requested. """
  pass
class Communicator(dict):
  """ Communicator to create MPI processes. """
  __slots__ = ['_nodefile', 'machines', 'parent', '__weakref__']
  """ Mostly to limit the possibility of circular references. """
  def __init__(self, *args, **kwargs):
    from pylada import default_comm
    keywords = default_comm.copy()
    keywords['n'] = 0
    keywords.update(kwargs)
    super(Communicator, self).__init__(*args)
    super(Communicator, self).update(keywords)
    self._nodefile = None
    """ Path to temporary node file.
        
        None if no temporary node file yet.
    """
    self.machines = {}
    """ Holds map of machines this communicator can access. """
    self.parent = None
    """ Reference to parent communicator from which machines are acquired. 
    
        If None, then this should be :py:data:`pylada.default_comm`, eg the very
        first communicator setup at the start of the application.
    """

  def lend(self, nprocs):
    """ Lend n processes from this communicator. 
    
        After the call, this communicator will not have acccess to the machines
        lent to the returned communicator. They should be given back when
        cleanup is called on the result.
    """
    from pylada import do_multiple_mpi_programs
    from weakref import ref
    try: 
      if nprocs == 'all': nprocs = self['n']
    except: pass
    if self._nodefile is not None:
      raise NodeFileExists("Comm already used in other process.")
    if nprocs > self['n']: raise MPISizeError((nprocs, self['n']))
    if nprocs <= 0: raise MPISizeError(nprocs)
    if do_multiple_mpi_programs == False and nprocs != self['n']:
      raise MPISizeError( 'No mpi placement is allowed '                       \
                          'by current Pylada configuration')
    result = self.__class__()
    result.machines = {}
    result.parent = ref(self)
    if len(self.machines) != 0:
      while result['n'] != nprocs:
        key, value = self.machines.iteritems().next()
        if result['n'] + value > nprocs:
          result.machines[key] = nprocs - result['n']
          self.machines[key] = value - result.machines[key]
          result['n'] = sum(result.machines.itervalues())
          self['n'] = sum(self.machines.itervalues())
        else:
          result.machines[key] = value
          result['n'] += value
          self.machines.pop(key)
          self['n'] -= value
    else: 
      result['n'] = nprocs
      self['n'] -= nprocs
    return result

  def split(self, n=2):
    """ Creates list of splitted Communicator. 

        List of machines is also splitted. 
    """ 
    from pylada.error import ValueError
    if self._nodefile is not None:
      raise NodeFileExists("Comm already used in other process.")
    if n < 1: raise ValueError("Cannot split communicator in less than two. ")
    if n > self['n']:
      raise MPISizeError("Cannot split {0} processes "\
                               "into {0} communicators.".format(self['n'], n))
    N = self['n']  
    return [self.lend(N // n + (1 if i < N % n else 0)) for i in xrange(n)]

  def acquire(self, other, n=None):
    """ Acquires the processes from another communicator. 

        Use at your risk... Family lines could become tangled, eg to which
        communicator will the processes return when self is destroyed? In
        practice, it will return to the parent of self. Is that what you want?

        The processes in other are acquired by self. If n is not None, only n
        processes are acquired, up to the maximum number of processes in other.
        If n is None, then all processes are acquired.

        As always, processes are exclusive owned exclusively. Processes
        acquired by self are no longuer in other.
    """ 
    if self._nodefile is not None or other._nodefile is not None:
      raise NodeFileExists("Comm already used in other process.")
    if n is not None: 
      comm = other.lend(n)
      self.acquire(comm)
      return
    for key, value in other.machines.iteritems():
      if key in self.machines: self.machines[key] += value
      else: self.machines[key] = value
    self['n'] += other['n']
    other.machines = {}
    other['n'] = 0
    other.parent = None
    other.cleanup()

  def nodefile(self, dir=None):
    """ Returns name of temporary nodefile. 

        This file should be cleaned up once the mpi process is finished by
        calling :py:meth:`~Communicator.cleanup`. It is an error to call this
        function without first cleaning up.  Since only one mpi process should
        be attached to a communicator, this makes sense.
    """
    from tempfile import NamedTemporaryFile
    from ..misc import RelativePath
    from pylada.misc import bugLev

    if self._nodefile is not None:
      if bugLev >= 5:
        print 'process/mpi.py: old nodefile: ', self._nodefile
      raise NodeFileExists("Please call cleanup first.  nodefile: \"%s\""
        % (self._nodefile,))
    if self['n'] == 0: raise MPISizeError("No processes in this communicator.")

    with NamedTemporaryFile(dir=RelativePath(dir).path, delete=False, prefix='pylada_comm') as file:
      if bugLev >= 5:
        print 'process/mpi.py: new nodefile: ', file.name
      for machine, slots in self.machines.iteritems():
        if slots == 0: continue
        ##file.write('{0} slots={1}\n'.format(machine, slots))
        file.write( machine)
        file.write('\n')
        if bugLev >= 5:
          print '  machine: %s  slots: %s' % (machine, slots,)

      self._nodefile = file.name
    return self._nodefile
 
  def __del__(self): 
    """ Cleans up communicator. """
    try: self.cleanup()
    except: pass
 
  def cleanup(self):
    """ Removes temporary file. """
    from os.path import exists
    from os import remove
    # return nodes to parent.
    parent = None if self.parent is None else self.parent()
    if parent is not None:
      for key, value in self.machines.iteritems(): 
        if key in parent.machines: parent.machines[key] += value
        else: parent.machines[key] = value
      parent['n'] += self['n']
      self.parent = None
      self.machines = {}
      self['n'] = 0
    # remove nodefile, if it exists.
    nodefile, self._nodefile = self._nodefile, None
    if nodefile is not None and exists(nodefile):
      try: remove(nodefile)
      except: pass

  def __getstate__(self):
    """ Pickles a communicator. 
    
        Does not save parent.
    """
    return self.copy(), self.machines, self._nodefile
  def __setstate__(self, value):
    """ Reset state from input.
       
        Sets parent to None.
    """
    self.update(value[0])
    self.machines, self._nodefile = value[1:]
    self.parent = None

def create_global_comm(nprocs, dir=None):
  """ Figures out global communicator through external mpi call. """
  from sys import executable
  from tempfile import NamedTemporaryFile
  from subprocess import PIPE
  from os.path import exists
  from os import environ
  from os import remove, getcwd
  from re import finditer
  from .. import mpirun_exe, modify_global_comm, do_multiple_mpi_programs,     \
                 figure_out_machines as script, launch_program as launch
  from ..misc import Changedir
  from ..error import ConfigError
  import pylada
  from pylada.misc import bugLev
  
  if bugLev >= 1:
    print 'process/mpi: create_global_comm: entry'
    print 'process/mpi: create_global_comm: nprocs: %s' % (nprocs,)
    print 'process/mpi: create_global_comm: dir: \"%s\"' % (dir,)
    print 'process/mpi: create_global_comm: script: \"%s\"' % (script,)

  if not do_multiple_mpi_programs: return
  if nprocs <= 0: raise MPISizeError(nprocs)
  if dir is None: dir = getcwd()
  
  # each proc prints its name to the standard output.
  with Changedir(dir) as pwd: pass
  try: 
    with NamedTemporaryFile(delete=False, dir=dir) as file:
      file.write(script)
      filename = file.name

    formatter = Communicator(n=nprocs).copy()
    formatter['program'] = executable + ' ' + filename

    if bugLev >= 5:
      print "process.mpi: create_global_comm: formatter: ", formatter
      print "process.mpi: filename: \"%s\"" % (filename,)
      print "===content:===\n%s===end===" % ( open(filename).read(),)
      print "process.mpi: create_global_comm: mpirun_exe: ", mpirun_exe
      print "process.mpi: *** launch ***"

    process = launch( mpirun_exe, stdout=PIPE, formatter=formatter,
                      stderr=PIPE, env=environ )
    if bugLev >= 5:
      print "process.mpi: create_global_comm: process: ", process
      print "process.mpi: *** start process.communicate ***"

    stdout, stderr = process.communicate()
    if bugLev >= 5:
      print "process.mpi: === start stdout ===\n%s\n=== end ===" % (stdout,)
      print "process.mpi: === start stderr ===\n%s\n=== end ===" % (stderr,)
      print "process.mpi: *** start process.communicate ***"
  finally:
    if exists(filename):
      try: remove(filename)
      except: pass
  # we use that to deduce the number of machines and processes per machine.
  processes = [ line.group(1) for line                                         \
                in finditer('PYLADA MACHINE HOSTNAME:\s*(\S*)', stdout) ]
  if bugLev >= 5:
    for p in processes:
      print "  process.mpi: create_global_comm: process: ", p
    print "process.mpi: nprocs: create_global_comm: ", nprocs
  # sanity check.
  if nprocs != len(processes):
    envstring = ''
    for key, value in environ.iteritems():
      envstring += '  {0} = {1!r}\n'.format(key, value)
    raise ConfigError( 'Pylada could not determine host machines.\n'             \
                       'Standard output reads as follows:\n{0}\n'              \
                       'Standard error reads as follows:\n{1}\n'               \
                       'environment variables were set as follows:\n{2}\n'     \
                       'following script was run:\n{3}\n'                      \
                       .format(stdout, stderr, envstring, script) )
  # now set up the default communicator.
  machines = set(processes)
  pylada.default_comm = Communicator(pylada.default_comm)
  pylada.default_comm['n'] = nprocs
  pylada.default_comm.machines = {}
  for machine in machines:
    pylada.default_comm.machines[machine] = processes.count(machine)
                        
  # called when we need to change the machine names on some supercomputers.
  modify_global_comm(pylada.default_comm)
