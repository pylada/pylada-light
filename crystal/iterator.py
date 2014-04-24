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

def layer(structure, direction, tolerance=1e-12):
  """ Iterates over layers and atoms in a layer. 

      :param structure: 
          :class:`Structure` for which to iterator over atoms.
      :param direction:
          3d-vector defining the growth direction, e.g. vector perpendicular to
          the layers.  Defaults to the first column vector of the structure.
          It is important that two cell-vectors of the structure are (or can be
          transformed to be) perpendicular to the growth direction. Otherwise
          it cannot be assured that layers are well defined, i.e. that each
          atom belongs to a single (as in periodic) layer. This condition is
          *not* enforced (no assertion) since it is not necessary, only
          sufficient.  Note that the third vector need not be parallel to the
          growth direction.
      :param tolerance: 
          Maximum difference between two atoms in the same layer.

      :returns: Yields iterators over atoms in a single epitaxial layer.
  """
  from operator import itemgetter
  from numpy import array, dot
  from . import into_cell, into_voronoi

  direction = array(direction)
  if len(structure) <= 1: yield list(structure); return

  # orders position with respect to direction.
  positions = array([into_cell(atom.pos, structure.cell) for atom in structure])
  projs = [(i, dot(pos, direction)) for i, pos in enumerate(positions)]
  projs = sorted(projs, key=itemgetter(1))

  # creates classes of positions.
  result = [[projs[0]]]
  for i, proj in projs[1:]:
    if abs(proj - result[-1][-1][-1]) < tolerance: result[-1].append((i,proj))
    else: result.append([(i,proj)])

  # only one layer.
  if len(result) == 1: yield structure; return
  # Finds if first and last have atoms in common through periodicity
  first, last = result[0], result[-1]
  centered                                                                     \
    = into_voronoi( positions[[i for i, d in last]] - positions[first[0][0]],
                    structure.cell )
  for j, pos in enumerate(centered[::-1]):
    a0 = dot(pos, direction)
    if any(abs(u[1]-a0) >= tolerance for u in first): continue
    first.append( last.pop(len(centered)-j-1) )

  # last layer got depleted.
  if len(last) == 0: result.pop(-1) 
  # only one layer.
  if len(result) == 1: yield structure; return
  # yield layer iterators.
  for layer in result:
    def inner_layer_iterator():
      """ Iterates over atoms in a single layer. """
      for index, norm in layer: yield structure[index]
    yield inner_layer_iterator ()


def equivalence(structure, operations=None, tolerance=1e-6, splitocc=None):
  """ Yields iterators over atoms equivalent via space group operations.
  
      Only check that the position are equivalent. Does not check that the
      occupations are the same. The sort is stable.

      :param structure:
          :class:`Structure` over which to iterate.
      :param operations: 
          List of symmetry operations.
          A symmetry operation is 4x4 matrix where the upper block is a
          rotation and the lower block a translation. The translation is
          applied *after* the rotation. If None, the operations are obtained
          using:class:`space_group`.
      :param float tolerance:
          Two positions closer than ``tolerance`` are considered equivalent.
      :param callable splitocc:
          Function to split two sites according to something other than
          geometry. Generally, this would be occupation and/or magnetic state.
          It is a callable taking two :py:class:`cppwrappers.Atom` and
          returning True or False depending on whether they are equivalent.
          It should be transitive and symmetric, otherwise results will be
          undetermined.  If None, then splitting occurs only according to
          geometry.
      
      :returns: 
        Yields list of indices in structure of atoms linked by space-group
        operations.
  """
  from numpy import dot
  from numpy.linalg import inv
  from . import space_group, which_site

  # atoms: list of atoms + index. Pop returns the last item. Since we want
  # equivalence iterator to be stable, as much as possible, the list order is
  # inverted.
  atoms = [u for u in enumerate(structure)][::-1]
  if operations == None: operations = space_group(structure)
  invcell = inv(structure.cell)
   
  while len(atoms):
    i, atom = atoms.pop()
    equivs = [i]
    if len(atoms): 
      # check symmetrically equivalent
      others = [u[1] for u in atoms]
      for op in operations:
        newpos = dot(op[:3], atom.pos) + op[3]
        index = which_site(newpos, others, invcell)
        if index != -1: 
          others.pop(index)
          other = atoms.pop(index)[0]
          equivs.append(other)
    # if no further splitting, returns. 
    if splitocc == None: yield equivs
    # otherwise, apply splitting.
    else: 
      # split according to callable.
      # callable should be transitive. 
      # This can be order N^2 in worst case scenario.
      # Using an indexing function would be order N always?
      results = [ [equivs[0]] ]
      for u in equivs[1:]: 
        found = False
        for group in results:
          if splitocc(group[0], u):
            group.append(u)
            found = True
            break
        if not found: results.append([u])
      # now yield each inequivalent group
      for group in results: yield group

def shell(structure, center, direction, thickness=0.05):
  """ Iterates over cylindrical shells of atoms.
  
      It allows to rapidly create core-shell nanowires.
  
      :param structure: 
          :class:`Structure` over which to iterate.
      :param center: 
          3d-vector defining the growth direction of the nanowire.
      :param thickness: 
          Thickness in units of ``structure.scale`` of an individual shell.
      
      :returns: Yields iterators over atoms in a single shell.
  """
  from operator import itemgetter
  from numpy import array, dot
  from numpy.linalg import norm
  from operator import into_voronoi

  direction = array(direction)/norm(array(direction))
  if len(structure) <= 1: yield structure; return

  # orders position with respect to cylindrical coordinates
  positions = array([atom.pos - center for atom in structure])
  positions = into_voronoi(positions, structure.cell)
  projs = [ (i, norm(pos - dot(pos, direction)*direction))                     \
            for i, pos in enumerate(positions)]
  projs = sorted(projs, key=itemgetter(1))

  # creates classes of positions.
  result = {}
  for i, r in projs:
    index = int(r/thickness+1e-12)
    if index in result: result[index].append(i)
    else: result[index] = [i]

  for key, layer in sorted(result.iteritems(), key=itemgetter(0)):
    def inner_layer_iterator():
      """ Iterates over atoms in a single layer. """
      for index in layer: yield structure[index]
    yield inner_layer_iterator()
