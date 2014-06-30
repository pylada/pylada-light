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
from pylada.crystal.cppwrappers import Structure
class StructureSubclass(Structure):
  def __init__(self, *args, **kwargs):
    super(StructureSubclass, self).__init__(*args, **kwargs)
structure_classes = [(Structure, ), (StructureSubclass, )]

def get_a_structure(Class):
    from numpy import identity
    return Class(identity(3)*0.25, scale=5.45, m=5)\
                 .add_atom(0,0,0, "Au")\
                 .add_atom(0.5,0.5,0.5, "Au")\
                 .add_atom(0.25,0.5,0.5, "Pd")

@parameterized(structure_classes)
def test_structure_referents_include_atoms(Class):
    from nose.tools import assert_equal
    from gc import get_referents

    structure = get_a_structure(Class)
    actual = set([id(u) for u in get_referents(structure)])
    expected = set(
        [id(structure.__dict__), id(structure.scale)]
        + [id(atom) for atom in structure]
    )
    if Class is not Structure: expected.add(id(Class))
    assert_equal(actual, expected)


@parameterized(structure_classes)
def test_garbage_collect_nocycle(Class):
    from nose.tools import assert_in, assert_not_in
    import gc

    structure = get_a_structure(Class)
    atom_ids = [id(u) for u in structure]
    structure_id = id(structure)
    scale_id = id(structure.scale)

    for this_id in atom_ids + [structure_id, scale_id]:
        assert_in(this_id, [id(u) for u in gc.get_objects()])

    # Deletes atom and collect garbage
    # atom should then be truly destroyed, e.g. neither tracked nor in
    # unreachables.
    del structure
    gc.collect()
    for this_id in atom_ids + [structure_id, scale_id]:
        assert_not_in(this_id, [id(u) for u in gc.get_objects()])
        assert_not_in(this_id, [id(u) for u in gc.garbage])


@parameterized(structure_classes)
def test_garbage_collect_cycle(Class):
    from nose.tools import assert_not_in, assert_in
    import gc

    structure = get_a_structure(Class)
    atom_ids = [id(u) for u in structure]
    structure_id = id(structure)
    scale_id = id(structure.scale)
    # add a cycle
    structure.parent_structure = structure

    for this_id in atom_ids + [structure_id, scale_id]:
        assert_in(this_id, [id(u) for u in gc.get_objects()])
    assert_in(
        structure_id,
        [id(u) for u in gc.get_referents(structure.__dict__)]
    )

    # Deletes atom and collect garbage
    # structure should then be truly destroyed, e.g. neither tracked nor in
    # unreachables.
    del structure
    gc.collect()
    for this_id in atom_ids + [structure_id, scale_id]:
        assert_not_in(this_id, [id(u) for u in gc.get_objects()])
        assert_not_in(this_id, [id(u) for u in gc.garbage])
