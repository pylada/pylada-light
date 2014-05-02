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

""" Provides a quick launch of jmol. """
jmol_processes = []
""" List of jmol processes. """
def atexit_jmol(X):
  x = X()
  if x is not None: x.__del__()
  
class Jmol(object):
  """ Holds and manages jmol process. """
  def __init__(self, structure):
    """ Creates the jmol process. """
    from tempfile import NamedTemporaryFile
    from subprocess import Popen
    from os import remove 
    from atexit import register
    from weakref import ref
    from ..crystal import write
    from .. import jmol_program
    try: 
      with NamedTemporaryFile(delete=False) as file:
        name = file.name
        file.write(write.castep(structure))
    except:
      try: remove(name)
      except: pass
    else:
      self.name = name
      self.stdout = open('/dev/null', 'w')
      self.stderr = open('/dev/null', 'w')
      self.process = Popen( jmol_program.split() + [name], stdout=self.stdout,
                            stderr=self.stderr) #, shell=True )
      jmol_processes.append(self)
      register(atexit_jmol, ref(self))
  def __del__(self):
    """ Cleans up after process. """
    if 'process' in self.__dict__:
      try: self.process.terminate()
      except: pass
      finally: del self.process
    if 'stdout' in self.__dict__:
      try: self.stdout.close()
      except: pass
      finally: del self.stdout
    if 'stderr' in self.__dict__:
      try: self.stderr.close()
      except: pass
      finally: del self.stderr
    if 'name' in self.__dict__:
      try:
        from os import remove
        remove(self.name)
      except: pass
      finally: del self.name
    try: 
      procs = [id(u) for u in jmol_processes]
      if id(self) in procs:
        jmol_processes.pop(procs.index(id(self)))
    except: pass
    
def jmol(self, event):
  """ Launches JMOL on a structure. """
  from inspect import ismethod
  from numpy import all, abs, max, min

  if '-h' in event.split() or '--help' in event.split(): 
    print  "usage: %jmol [-h] structure\n"                                     \
           "\n"                                                                \
           "Launches jmol for a given structure.\n"                            \
           "\n"                                                                \
           "positional arguments:\n"                                           \
           "  structure   Variable/expression referencing a structure.\n"      \
           "\n"                                                                \
           "optional arguments:\n"                                             \
           "  -h, --help  show this help message and exit"
    return 
  if len(event.rstrip().lstrip()) == 0:
    print '%jmol requires at least one argument.'
    return
  shell = get_ipython()
  if event.rstrip().lstrip() in shell.user_ns:
    structure = shell.user_ns[event.rstrip().lstrip()]
  else: 
    structure = eval(event.rstrip().lstrip(), shell.user_ns.copy())
  if ismethod(getattr(structure, 'eval', None)):
    structure = structure.eval()

  if all(abs(structure.cell[:, 2] - [0, 0, 500.0]) < 1e-8):
    mini = abs(min([a.pos[2] for a in structure]))
    maxi = abs(max([a.pos[2] for a in structure]))
    structure.cell[2, 2] = mini + maxi
    structure.cell[2, 2] *= 1.1

  if all(abs(structure.cell[:, 1] - [0, 500.0, 0.0]) < 1e-8):
    mini = abs(min([a.pos[1] for a in structure]))
    maxi = abs(max([a.pos[1] for a in structure]))
    structure.cell[1, 1] = mini + maxi
    structure.cell[1, 1] *= 1.1

  if all(abs(structure.cell[:, 0] - [500.0, 0.0,  0.0]) < 1e-8):
    mini = abs(min([a.pos[0] for a in structure]))
    maxi = abs(max([a.pos[0] for a in structure]))
    structure.cell[0, 0] = mini + maxi
    structure.cell[0, 0] *= 1.1

  Jmol(structure)
