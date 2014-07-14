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
from nose_parameterized import parameterized
from pylada.crystal.cppwrappers import Atom

def test_c_object_to_python():
    from nose.tools import assert_is
    from pylada.crystal.tests.atom_self import get_static_object, _atom
    
    # checks the static object is an Atom at start.
    a = get_static_object();
    assert_is(a.__class__, Atom)
    assert_is(a, _atom)
    # checks the static object is always itself.
    b = get_static_object();
    assert_is(a, b)
    assert_is(b, _atom)

class Subclass(Atom):
  def __init__(self, *args, **kwargs):
    super(Subclass, self).__init__(*args, **kwargs)
classes = [(Atom,), (Subclass,)]

@parameterized(classes)
def test_c_to_python_and_back(Class):
    """ checks module object can  be changed and subclassed from C """
    import gc
    from nose.tools import assert_is, assert_is_not, assert_equal
    from numpy.testing import assert_allclose, assert_equal as np_assert_equal
    from numpy import all, abs
    from pylada.crystal.cppwrappers import Atom
    from pylada.crystal.tests.atom_self import get_static_object, set_static_object, _atom
    
    a = Class(0.3,0.1,0.2, 'Au')
    set_static_object(a)
    c = get_static_object();
    assert_is(c, a)
    assert_is(c.__class__, Class)
    assert_allclose(a.pos, [0.3, 0.1, 0.2])
    assert_equal(a.type, 'Au')
    # checks same as above but with deletion.
    a = Class(0.4,0.1,0.2, 'Au', 'Pd', m=5)
    set_static_object(a)
    del a; del c; gc.collect()
    c = get_static_object();
    assert_is(c.__class__, Class)
    assert_allclose(c.pos, [0.4, 0.1, 0.2])
    np_assert_equal(c.type, ['Au', 'Pd'])
    assert_equal(c.__dict__, {'m': 5})
