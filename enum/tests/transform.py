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

def get_cell(n=5):
  from numpy.random import randint
  from numpy.linalg import det
  cell = randint(2*n, size=(3,3)) - n
  while abs(det(cell)) < 1e-8:
    cell = randint(2*n, size=(3,3)) - n
  return cell

def test_translations():
  from numpy import abs, all
  from pylada.crystal import binary, supercell, HFTransform
  from pylada.enum import Transforms

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge', 'C']

  for u in xrange(10):
    # create random structure
    structure = supercell(lattice, get_cell(5))
    hft = HFTransform(lattice, structure)
 
    # these are all the translations
    translations = Transforms(lattice).translations(hft)
    assert translations.shape == (len(structure)//len(lattice) - 1, len(structure))
    # compute each translation and gets decorations
    for atom in structure:
      if atom.site != 0: continue
      # create translation
      trans = atom.pos - lattice[0].pos
      if all(abs(trans) < 1e-8): continue
      # figure out its index
      index = hft.index(trans) - 1
      for site in structure:
        pos = site.pos-lattice[site.site].pos 
        i = hft.index(pos, site.site)
        j = hft.index(pos + trans, site.site)
        assert translations[index, i] == j

def test_firstisid():
  """ Assumption is made in Transforms.transformations """
  from numpy import abs, all, identity
  from pylada.crystal import binary, supercell, space_group

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge', 'C']

  for u in xrange(10):
    # create random structure
    structure = supercell(lattice, get_cell(5))
    while len(structure) > 1: structure.pop(-1)
    assert len(structure) == 1
    sg = space_group(structure)[0]
    assert all(abs(sg[:3]- identity(3, dtype='float64'))<1e-8)
    assert all(abs(sg[3])<1e-8)

def test_rotations():
  from numpy import all, dot, zeros
  from numpy.linalg import inv
  from pylada.crystal import binary, supercell, HFTransform, space_group,        \
                           which_site
  from pylada.enum import Transforms

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge', 'C']
  sg = space_group(lattice)
  invcell = inv(lattice.cell)

  def get_cells(n):
    for i in xrange(1, n):
      yield [[i, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]]
    for i in xrange(1, n):
      yield [[i, 0, 0], [0, i, 0], [0, 0, 1]]
    for i in xrange(1, n):
      yield [[i, 0, 0], [0, i, 0], [0, 0, i]]
    yield dot(lattice.cell, [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
  for cell in get_cells(8):
    # create random structure
    structure = supercell(lattice, cell)
    hft = HFTransform(lattice, structure)
 
    # these are all the translations
    transforms = Transforms(lattice)
    permutations = transforms.transformations(hft)
    assert permutations.shape == (len(sg) - 1, len(structure))
    operations = transforms.invariant_ops(structure)
    assert any(operations) 

    # compute each translation and gets decorations
    for index, (op, isgood) in enumerate(zip(sg[1:], operations)):
      if not isgood: continue
      # Create rotation and figure out its index
      permutation = zeros(len(structure), dtype='int') - 1
      for atom in structure:
        pos = dot(op[:3], atom.pos) + op[3]
        newsite = which_site(pos, lattice, invcell)
        i = hft.index(atom.pos - lattice[atom.site].pos, atom.site)
        j = hft.index(pos - lattice[newsite].pos, newsite)
        permutation[i] = j
      assert all(permutation == permutations[index])

def test_multilattice():
  from numpy import all, dot, zeros
  from numpy.linalg import inv
  from pylada.crystal import binary, supercell, HFTransform, space_group,        \
                           which_site
  from pylada.enum import Transforms

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge']
  sg = space_group(lattice)
  invcell = inv(lattice.cell)

  def get_cells(n):
    for i in xrange(1, n):
      yield [[i, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]]
    for i in xrange(1, n):
      yield [[i, 0, 0], [0, i, 0], [0, 0, 1]]
    for i in xrange(1, n):
      yield [[i, 0, 0], [0, i, 0], [0, 0, i]]
    yield dot(lattice.cell, [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
  for cell in get_cells(8):
    # create random structure
    structure = supercell(lattice, cell)
    hft = HFTransform(lattice, structure)
 
    # these are all the translations
    transforms = Transforms(lattice)
    permutations = transforms.transformations(hft)
    assert permutations.shape == (len(sg) - 1, len(structure))
    operations = transforms.invariant_ops(structure)
    assert any(operations) 

    # compute each translation and gets decorations
    for index, (op, isgood) in enumerate(zip(sg[1:], operations)):
      if not isgood: continue
      # Create rotation and figure out its index
      permutation = zeros(len(structure), dtype='int') - 1
      for atom in structure:
        pos = dot(op[:3], atom.pos) + op[3]
        newsite = which_site(pos, lattice, invcell)
        i = hft.index(atom.pos - lattice[atom.site].pos, atom.site)
        j = hft.index(pos - lattice[newsite].pos, newsite)
        permutation[i] = j
      assert all(permutation == permutations[index])

def test_lattice():
  """ Tests lattice enhancements. """
  from numpy import all
  from pylada.crystal import binary, A2BX4
  from pylada.enum import Transforms

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge']
  transforms = Transforms(lattice)
  assert len([u for u in transforms.lattice if u.asymmetric]) == 1
  assert transforms.lattice[0].asymmetric
  assert transforms.lattice[0].equivto == 0
  assert transforms.lattice[0].nbflavors == 2
  assert transforms.lattice[0].index == 0
  assert not transforms.lattice[1].asymmetric
  assert transforms.lattice[1].equivto == 0
  assert transforms.lattice[1].nbflavors == 2
  assert transforms.lattice[1].index == 1
  assert all(all(a == b) for a, b in zip(transforms.flavors, (range(1), range(1))))
  assert all(not hasattr(atom, 'nbflavors') for atom in lattice)

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge', 'C']
  transforms = Transforms(lattice)
  assert len([u for u in transforms.lattice if u.asymmetric]) == 2
  assert transforms.lattice[0].asymmetric
  assert transforms.lattice[0].equivto == 0
  assert transforms.lattice[0].nbflavors == 2
  assert transforms.lattice[0].index == 0
  assert transforms.lattice[0].asymmetric
  assert transforms.lattice[1].equivto == 1
  assert transforms.lattice[1].nbflavors == 3
  assert transforms.lattice[1].index == 1
  assert all(all(a == b) for a, b in zip(transforms.flavors, (range(1), range(2))))

  lattice = A2BX4.b5() 
  for atom in lattice: 
    if atom.type in ['A', 'B']: atom.type = 'A', 'B'
  transforms = Transforms(lattice)
  assert len([u for u in transforms.lattice if u.asymmetric]) == 3
  assert all([transforms.lattice[i].asymmetric for i in [0, 4, 6]])
  assert all([not transforms.lattice[i].asymmetric for i in range(1, 4) + [5] + range(7, 14)])
  assert all([transforms.lattice[i].equivto == 0 for i in range(4)])
  assert all([transforms.lattice[i].equivto == 4 for i in range(4, 6)])
  assert all([transforms.lattice[i].equivto == 6 for i in range(6, 14)])
  assert all([transforms.lattice[i].nbflavors == 2 for i in range(4)])
  assert all([transforms.lattice[i].nbflavors == 2 for i in range(4, 6)])
  assert all([transforms.lattice[i].nbflavors == 1 for i in range(6, 14)])
  assert all([transforms.lattice[i].index == i for i in range(6)])
  assert all([not hasattr(transforms.lattice[i], 'index') for i in range(6, 14)])

  lattice[0], lattice[-1] = lattice[-1], lattice[0]
  transforms = Transforms(lattice)
  assert len([u for u in transforms.lattice if u.asymmetric]) == 3
  assert all([transforms.lattice[i].asymmetric for i in [0, 1, 4]])
  assert all([not transforms.lattice[i].asymmetric for i in range(2, 4) + range(5, 14)])
  assert all([transforms.lattice[i].equivto == 0 for i in [0] + range(6, 13)])
  assert all([transforms.lattice[i].equivto == 1 for i in range(1, 4) + [13]])
  assert all([transforms.lattice[i].equivto == 4 for i in [4, 5]])
  assert all([transforms.lattice[i].nbflavors == 1 for i in [0] + range(6, 13)])
  assert all([transforms.lattice[i].nbflavors == 2 for i in range(1, 4) + [13]])
  assert all([transforms.lattice[i].nbflavors == 2 for i in [4, 5]])
  index = 0
  for i, atom in enumerate(transforms.lattice):
    if atom.nbflavors == 1: assert not hasattr(atom, 'index')
    else: 
      assert atom.index == index
      index += 1

def test_toarray():
  """ Tests label exchange """
  from random import choice
  from numpy import all, zeros
  from pylada.crystal import binary, supercell, HFTransform
  from pylada.enum import Transforms

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge', 'C']
  transforms = Transforms(lattice)
  lattice = transforms.lattice

  for u in xrange(11):
    structure = supercell(lattice, get_cell())
    for atom in structure: atom.type = choice(atom.type)
    hft = HFTransform(lattice, structure) 
    a = transforms.toarray(hft, structure)
    b = zeros(len(structure), dtype='int')
    for atom in structure:
      site = lattice[atom.site]
      b[hft.index(atom.pos-site.pos, atom.site)]                               \
        = site.type.index(atom.type) + 1
    assert all(a == b)

def test_labelexchange():
  """ Tests label exchange """
  from pylada.crystal import binary, supercell, HFTransform
  from pylada.enum import Transforms

  lattice = binary.zinc_blende()
  lattice[0].type = ['Si', 'Ge']
  lattice[1].type = ['Si', 'Ge', 'C']
  transforms = Transforms(lattice)
  lattice = transforms.lattice

  structure = supercell(lattice, [[8, 0, 0], [0, 0.5, 0.5], [0, -0.5, 0.5]])
  species = [ 'Ge', 'C', 'Si', 'C', 'Si', 'C', 'Si', 'Si', 'Ge', 'Si', 'Ge',
              'Si', 'Ge', 'Si', 'Ge', 'Ge', 'Ge', 'C', 'Ge', 'Si', 'Si', 'Si',
              'Si', 'Ge', 'Si', 'Ge', 'Si', 'Si', 'Si', 'C', 'Ge', 'Si']
  for atom, s in zip(structure, species): atom.type = s
  hft = HFTransform(lattice, structure) 
  x = transforms.toarray(hft, structure)
  results = [ 21112222221111123331111231122131L, # <- this is x
              21112222221111122221111321133121L,
              21112222221111123332222132211232L,
              21112222221111121112222312233212L,
              21112222221111122223333123311323L,
              21112222221111121113333213322313L,
              12221111112222213331111231122131L,
              12221111112222212221111321133121L,
              12221111112222213332222132211232L,
              12221111112222211112222312233212L,
              12221111112222212223333123311323L,
              12221111112222211113333213322313L ]
  permutations = transforms.label_exchange(hft)
  for a, b in zip(permutations(x), results[1:]): 
    assert int(str(a)[1:-1].replace(' ', '')) == b
  
if __name__ == '__main__':
  test_translations()
  test_firstisid()
  test_lattice()
  test_toarray()
  test_rotations()
  test_multilattice()
  test_labelexchange()

