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

""" Checks c++ to python and back.

    A cpp-created atom can be passed to python.
    A python-create atom can be passed to cpp and back.
    A python-subclassed atom can be passed to a cpp object and back. 
"""
def test(Class):
  import gc
  from numpy import all, abs
  from pylada.crystal.cppwrappers import Atom
  from _atom_self import get_static_object, set_static_object, _atom
  
  # checks the static object is an Atom at start.
  a = get_static_object();
  assert a.__class__ is Atom
  assert a is _atom
  # checks the static object is always itself.
  b = get_static_object();
  assert a is b
  assert b is _atom
  # checks it can be changed and subclassed.
  a = Class(0.4,0.1,0.2, 'Au')
  set_static_object(a)
  c = get_static_object();
  assert c is not b
  assert c is a
  assert c.__class__ is Class
  assert all(abs(a.pos - [0.4, 0.1, 0.2]) < 1e-8) and a.type == 'Au'
  # checks same as above but with deletion.
  a = Class(0.4,0.1,0.2, 'Au', 'Pd', m=5)
  set_static_object(a)
  del a; del b; del c; gc.collect()
  c = get_static_object();
  assert c.__class__ is Class
  assert all(abs(c.pos - [0.4, 0.1, 0.2]) < 1e-8) and c.type == ['Au', 'Pd'] \
         and len(c.__dict__) == 1 and getattr(c, 'm', 0) == 5

if __name__ == "__main__": 
  from pylada.crystal.cppwrappers import Atom
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  # tries to run test with normal class.
  test(Atom) 
  
  # tries to run test with other class. 
  # subclases an exception and makes sure it is called.
  # also checl passage through init.
  check_passage = False
  class MineErr(Exception): pass
  class Subclass(Atom):
    def __init__(self, *args, **kwargs):
      global check_passage
      check_passage = True
      super(Subclass, self).__init__(*args, **kwargs)
    def __call__(self): raise MineErr()

  test(Subclass)
  assert check_passage
 
  # checks that exception is called in __call__
  import gc
  from _atom_self import get_static_object, set_static_object
  gc.collect()
  try: get_static_object()()
  except MineErr: pass
  else: raise Exception()

  # tries to pass the wrong type object to the cpp wrapper.
  from pylada.error import TypeError as PyladaTypeError
  class B(object): pass
  b = B()
  try: set_static_object(b)
  except PyladaTypeError: pass
  else: raise Exception()
