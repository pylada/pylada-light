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

""" Checks atom methods and attributes. """
def test():
  """ Test atom initialization. """
  from _pyobject import represent, add_attribute, callme, equality

  assert represent("a") == repr("a");
  assert represent(1) == repr(1);

  class A(object):
    def __init__(self): 
      self.been_here = False
    def __call__(self): 
      self.been_here = True

  a = A()
  add_attribute(a, 'hello', 5)
  assert getattr(a, 'hello', 6) == 5
  assert not a.been_here
  callme(a)
  assert a.been_here
 
  assert equality(0, 0)
  assert not equality(1, 0)
  assert equality('1', '1')
  assert not equality('2', '1')

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  test()
