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

__docformat__ = "restructuredtext en"
__all__ = ['Transforms']

class Transforms(object):
  """ Lattice transformation object.

      This object can create the correct permutation for any transformation of
      the periodic lattice (except for pure translation).
  """
  def __init__(self, lattice):
    """ Creates Transform object.

        :param op:
          4x3 matrix representing the transformation.
        :param lattice: 
          Lattice which forms the back-bone on which to enumerate structures.
        :type lattice: :py:class:`~pylada.crystal.cppwrappers.Structure`
    """
    from numpy import dot
    from numpy.linalg import inv
    from ..crystal import which_site, space_group
    super(Transforms, self).__init__()
    self.lattice = lattice.copy()
    self.space_group = space_group(self.lattice)
    self.dnt = []
    """ Site permutations and translation vector. """
    self._enhance_lattice()
    self.equivmap = [u.equivto for u in self.lattice]
    """ Site map for label exchange. """
    self.flavors = [ range(1, site.nbflavors+1) for site in self.lattice       \
                     if site.nbflavors != 1 and site.asymmetric ]
    """ List of possible flavors for each asymmetric site. """
    invcell = inv(lattice.cell)
    sites = [site for site in self.lattice if site.nbflavors != 1]
    for op in self.space_group[1:]:
      self.dnt.append([])
      for i, site in enumerate(sites):
        newpos = dot(op[:3], site.pos) + op[3]
        j = which_site(newpos, sites, invcell)
        trans = newpos - sites[j].pos 
        self.dnt[-1].append( (j, trans) )
  
  def _enhance_lattice(self):
    """ Adds stuff to the lattice """
    from numpy import dot
    from ..crystal import which_site
    index = 0
    for i, site in enumerate(self.lattice):
      site.equivto = set([i])
      for op in self.space_group[1:]:
        j = which_site( dot(op[:3], site.pos) + op[3], self.lattice)
        assert j != -1
        site.equivto.add(j)
    for site in self.lattice:
      intermediate = set()
      for j in site.equivto: intermediate |= self.lattice[j].equivto
    for site in self.lattice: site.equivto = min(site.equivto)
    for i, site in enumerate(self.lattice):
      if not hasattr(site.type, '__iter__'): site.nbflavors = 1
      elif len(site.type) < 2:               site.nbflavors = 1
      else:
        site.nbflavors = len(site.type)
        site.index = index
        index += 1
      site.asymmetric = site.equivto == i

  def translations(self, hft):
    """ Array of permutations arising from pure translations """
    from itertools import product
    from numpy import zeros
    nsites = len(self.dnt[0])
    itertrans = [ xrange(hft.quotient[0]), 
                  xrange(hft.quotient[1]), 
                  xrange(hft.quotient[2]) ] 
    size = hft.size
    result = zeros((size-1, nsites * size), dtype='int16')  - 1
    iterable = product(*itertrans)
    a = iterable.next() # avoid null translation
    assert a == (0,0,0) # check that it is null
    for t, (i,j,k) in enumerate(iterable):
      iterpos = [ xrange(hft.quotient[0]), 
                  xrange(hft.quotient[1]), 
                  xrange(hft.quotient[2]) ] 
      for l, m, n in product(*iterpos):
        u = (i+l) % hft.quotient[0]
        v = (j+m) % hft.quotient[1]
        w = (k+n) % hft.quotient[2]
        for s in xrange(nsites):
          result[t, hft.flatten_indices(l, m, n, s)]                           \
              = hft.flatten_indices(u, v, w, s)
    return result

  def transformations(self, hft):
    """ Creates permutations for given Hart-Forcade transform. """
    from itertools import product
    from numpy import zeros, dot
    from numpy.linalg import inv
    result = zeros( (len(self.space_group)-1, hft.size * len(self.dnt[0])),
                    dtype='int') - 1
    invtransform = inv(hft.transform)
    for nop, (op, dnt) in enumerate(zip(self.space_group[1:], self.dnt)):
      rotation = dot(hft.transform, dot(op[:3], invtransform))
      for s, (siteperm, translation) in enumerate(dnt):
        trans = hft.indices(translation)
        iterpos = [ xrange(hft.quotient[0]), 
                    xrange(hft.quotient[1]), 
                    xrange(hft.quotient[2]) ] 
        for i,j,k in product(*iterpos):
          newpos = dot(rotation, [i, j, k])
          l = (int(round(newpos[0])) + trans[0]) % hft.quotient[0]
          m = (int(round(newpos[1])) + trans[1]) % hft.quotient[1]
          n = (int(round(newpos[2])) + trans[2]) % hft.quotient[2]
          result[nop, hft.flatten_indices(i, j, k, s)]                         \
              = hft.flatten_indices(l, m, n, siteperm) 
    return result

  def invariant_ops(self, cell):
    """ boolean indices into invariant operations. 

        :param cell:
           Cell for which the operations should be invariant.
        :type cell:
           :py:attr:`~pylada.crystal.cppwrappers.Structure` or 3x3 matrix
        :returns:
           An boolean array with the same length as the numbers of row returned
           by :py:method:`transformations`.
    """
    from numpy import dot, zeros
    from numpy.linalg import inv
    from .cppwrappers import is_integer
    cell = getattr(cell, 'cell', cell)
    invcell = inv(cell)
    result = zeros(shape=(len(self.space_group)-1), dtype='bool')
    for i, op in enumerate(self.space_group[1:]):
      matrix = dot(invcell, dot(op[:3], cell))
      result[i] = is_integer(matrix)
    return result

  def label_exchange(self, hft):
    """ List of functions to do label exchange """
    from itertools import permutations, product
    from numpy import array
    size = hft.size
    iterables = [permutations(flavors) for flavors in self.flavors] 
    iterables = product(*iterables)
    a = iterables.next()
    assert all(all(array(b) == array(c)) for b, c in zip(a, self.flavors))
    def permutations(x):
      """ Iterator over label exchange """
      for perms in iterables:
        yield array([ perms[self.equivmap[i//size]][x[i]-1]                    \
                      for i in xrange(len(x)) ] )
    return permutations
  
  def toarray(self, hft, structure):        
    """ Transforms structure into an array """
    from numpy import zeros
    result = zeros(len(structure), dtype='int')
    for atom in structure:
      site = atom.site
      if self.lattice[site].nbflavors == 1: continue
      site = self.lattice[atom.site]
      index = hft.index(atom.pos - site.pos, site.index)
      result[index] = site.type.index(atom.type) + 1
    return result
