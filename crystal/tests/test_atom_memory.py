###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to
#  make it easier to submit large numbers of jobs on supercomputers. It
#  provides a python interface to physical input, such as crystal structures,
#  as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs.
#  It is able to organise and launch computational jobs on PBS and SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License along with
#  PyLaDa.  If not, see <http://www.gnu.org/licenses/>.
###############################

from nose_parameterized import parameterized
from pylada.crystal.cppwrappers import Atom
class Subclass(Atom):
  def __init__(self, *args, **kwargs):
    super(Subclass, self).__init__(*args, **kwargs)
classes = [(Atom,), (Subclass,)]

@parameterized(classes)
def test_referents_nonetype(Class):
    from nose.tools import assert_equal
    from gc import get_referents

    atom = Class(0.1, 0.2, 0.5)
    actual = set([id(u) for u in get_referents(atom)])
    expected = set([id(atom.__dict__), id(None)])
    if Class is not Atom: expected.add(id(Class))
    assert_equal(actual, expected)


@parameterized(classes)
def test_referents_stringtype(Class):
    from nose.tools import assert_equal
    from gc import get_referents

    atom = Class(0.1, 0.2, 0.5, 'Au')
    actual = set([id(u) for u in get_referents(atom)])
    expected = set([id(atom.__dict__), id(atom.type)])
    if Class is not Atom: expected.add(id(Class))
    assert_equal(actual, expected)


@parameterized(classes)
def test_referents_circular(Class):
    from nose.tools import assert_equal
    from gc import get_referents

    atom0, atom1 = Class(0.1, 0.2, 0.5), Class(0.2, 0.2, 0.5)
    atom0.type = atom1
    atom1.type = atom1

    actual0 = set([id(u) for u in get_referents(atom0)])
    expected0 = set([id(atom0.__dict__), id(atom1)])
    if Class is not Atom: expected0.add(id(Class))
    assert_equal(actual0, expected0)

    actual1 = set([id(u) for u in get_referents(atom1)])
    expected1 = set([id(atom1.__dict__), id(atom1)])
    if Class is not Atom: expected1.add(id(Class))
    assert_equal(actual1, expected1)


@parameterized(classes)
def test_garbage_collect_nocycle(Class):
    from nose.tools import assert_true, assert_not_in
    import gc

    atom = Class(0, 1, 2, ['Au', 'B'])
    atom.pos += [0.1, -0.1, 0.2]
    id_of_a = id(atom)
    assert_true(id(atom), [id(u) for u in gc.get_objects()])

    # Deletes atom and collect garbage
    # atom should then be truly destroyed, e.g. neither tracked nor in
    # unreachables.
    del atom
    gc.collect()
    assert_not_in(id_of_a, [id(u) for u in gc.get_objects()])
    assert_not_in(id_of_a, [id(u) for u in gc.garbage])


@parameterized(classes)
def test_garbage_collect_withcycle(Class):
    from nose.tools import assert_not_in, assert_in
    import gc

    atom0, atom1 = Class(0, 1, 2), Class(0, 0.1, 0.2)
    atom0.type = atom1
    atom1.other = atom0
    id_of_0, id_of_1 = id(atom0), id(atom1)
    assert_in(id(atom0), [id(u) for u in gc.get_objects()])
    assert_in(id(atom1), [id(u) for u in gc.get_objects()])

    # Deletes atom0 and collect garbage
    # atom should not be destroyed, nor be unreachable
    del atom0
    gc.collect()
    assert_in(id_of_0, [id(u) for u in gc.get_objects()])
    assert_in(id_of_1, [id(u) for u in gc.get_objects()])

    # Deletes atom1 and collect garbage
    # both atoms should be destroyed and not unreachable
    del atom1
    gc.collect()
    assert_not_in(id_of_0, [id(u) for u in gc.get_objects()])
    assert_not_in(id_of_1, [id(u) for u in gc.get_objects()])
    assert_not_in(id_of_0, [id(u) for u in gc.garbage])
    assert_not_in(id_of_1, [id(u) for u in gc.garbage])


@parameterized(classes)
def test_garbage_collect_hold_ref_to_position(Class):
    from nose.tools import assert_not_in, assert_in, assert_equal
    import gc

    atom = Class(0, 1, 2, 'Au')
    pos = atom.pos
    id_of_atom = id(atom)
    assert_in(id(atom), [id(u) for u in gc.get_objects()])
    assert_not_in(id(pos), [id(u) for u in gc.get_objects()])

    assert_equal(id(pos.base), id(atom))

    # Deletes atom0 and collect garbage
    # atom should not be destroyed since it is held by pos
    del atom
    gc.collect()
    assert_in(id_of_atom, [id(u) for u in gc.get_objects()])

    # Now deletes pos and collects
    # This should also delete atom
    del pos
    gc.collect()
    assert_not_in(id_of_atom, [id(u) for u in gc.get_objects()])
    assert_not_in(id_of_atom, [id(u) for u in gc.garbage])
