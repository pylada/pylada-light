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

""" Check memory deallocation.

    Creates cyclic references through a structure's dictionary.
    A number of instances are created and gets difference in memory hogged by python.
    These instances are then deleted and gc.collect is called.
    We then go through a loop where the instantiation/creation steps are
    perfomed again, with gc.collect to make sure garbage is deleted. 
    Finally, we check that memory is 10 times less than after first creation above.

    Note that python does not release memory it allocates. Hence, deletion and
    gc.collect must be inside the loop, or test could give false negative.

    The same is checked for a subclass of atom.
""" 
def mklist(Class, N):
  from numpy import identity
  structure = Class(identity(3)*0.25, scale=5.45, m=5)\
               .add_atom(0,0,0, "Au")\
               .add_atom(0.5,0.5,0.5, "Au")
  result = [structure for u in range(10*N)]
  for r in result: r[0].parent = [r]
  b = [u.cell for u in result]
  return result, b

def get_mem(id):
  from subprocess import Popen, PIPE
  output = Popen(["ps","-p", "{0}".format(id), '-o', 'rss'], stdout=PIPE).communicate()[0].splitlines()[-1]
  return int(output)

def mem_per_structure(N):
  import gc
  from os import getpid
  from pylada.crystal.cppwrappers import Structure
  id = getpid()
  gc.set_debug(gc.DEBUG_OBJECTS | gc.DEBUG_UNCOLLECTABLE)
  startmem = get_mem(id)
  a = []
  for i in range(N):
    a.append(mklist(Structure, N))
  mem = float(get_mem(id) - startmem)
  del a
  gc.collect()
  assert mem > 0 # otherwise, test would be invalid.
  return mem

def test(Class, N, mem_per_structure): 
  import gc
  from os import getpid
  gc.set_debug(gc.DEBUG_OBJECTS | gc.DEBUG_UNCOLLECTABLE)

  id = getpid()

  startmem = get_mem(id)
  for i in range(N*5): 
    a, b = mklist(Class, N)
    # do deletion here, otherwise python might allocate extra memory to store our
    # objects, and the test would fail for reasons other than garbage collection.
    del a
    del b
    gc.collect()
  mem2 = float(get_mem(id) - startmem)
  assert mem2 < mem_per_structure / 10.0
  assert len(gc.garbage) == 0

if __name__ == "__main__": 
  from pylada.crystal.cppwrappers import Structure
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  # gets mem_per_structure
  mem = mem_per_structure(10)
  # tries to run test with normal class.
  test(Structure, 10, mem) 
 
  # tries to run test with other class. 
  # check passage through init.
  check_passage = False
  class Subclass(Structure):
    def __init__(self, *args, **kwargs):
      global check_passage
      check_passage = True
      super(Subclass, self).__init__(*args, **kwargs)

  test(Subclass, 10, mem)
  assert check_passage
