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

from .process import Process
class DummyProcess(Process):
  """ Used to debug process allocation.
  
      Each dummy process has :py:attr:`~DummyProcess.chance` of completing at
      each polling. This is useful to check whether scheduling works on more
      complicated process managements scenarios.
  """
  def __init__(self, maxtrials=1, chance=0.5, **kwargs):
    """ Initializes a process. """
    super(DummyProcess, self).__init__(maxtrials, **kwargs)
    self.chance = chance

  def poll(self): 
    """ Polls current job. """
    from random import random
    if super(DummyProcess, self).poll(): return True
    if random() < self.chance: 
      self._cleanup()
      return True
    return False

  def start(self, comm):
    """ Starts current job. """
    if super(DummyProcess, self).start(comm): return True
    self._next()
    return False
 
  def _next(self):
    """ Starts an actual process. """
    self.process = True

  def wait(self):
    """ Waits for process to end, then cleanup. """
    super(DummyProcess, self).wait()
    while not self.poll(): continue

class DummyFunctional(object):
  """ A dummy functional which uses a DummyProces. """
  def __init__(self, chance=0.5, **kwargs):
    """ Initializes a DummyFunctional. """
    super(DummyFunctional, self).__init__()
    self.chance = chance
    self.started = False
  def iter(self, **kwargs):
    """ Yields a dummy process. """
    from collections import namedtuple
    if self.started: 
      Extract = namedtuple('Extract', 'success')
      yield Extract(True)
      return
    self.started = True
    chance = kwargs.get('chance', self.chance)
    yield DummyProcess(chance=chance)
  def __call__(self, **kwargs):
    for process in self.iter(**kwargs):
      process.wait()
    return 

