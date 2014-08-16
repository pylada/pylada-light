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
__all__ = ['Transforms', 'supercells', 'hf_groups']
from .transforms import Transforms

def supercells(lattice, sizerange):
  """ Determines non-equivalent supercells in given size range.
  
      :params lattice: 
         Back-bone lattice
      :type lattice: py:attr:`~pylada.crystal.Structure`
      :param sizerange: 
         List of sizes for which to perform calculations, in number of
         unit-cells per supercell.
      :type sizerange: integer sequence
      :returns:
          dictionary where each key is a size and each element a list of
          inequivalent supercells of that size
  """
  from itertools import product
  from numpy import dot, array
  from numpy.linalg import inv
  from ..crystal import space_group
  from .cppwrappers import is_integer

  sizerange = sorted([k for k in sizerange if k > 0])
  results = {}
  for n in sizerange: results[n] = []
  mink = min(sizerange)
  maxk = max(sizerange)
  cell = lattice.cell
  invcell = inv(cell)
  spacegroup = space_group(lattice)
  for i in xrange(len(spacegroup)):
    spacegroup[i] = dot(invcell, dot(spacegroup[i][:3], cell))

  def isthere(sc, l):
    """ Check if a known supercell """
    isc = inv(sc)
    for op in spacegroup:
      op2 = dot(isc, op)
      if any( is_integer(dot(op2, u)) for u in l): return True
    return False

  supercell = array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype='int')
  for a in range(1, maxk+1):
    maxb = maxk // a 
    if maxb % a != 0: maxb += 1 
    supercell[0,0] = a
    for b in xrange(1, maxb+1): 
      maxc = maxk//(a*b)
      if maxk % (a*b) != 0: maxc += 1
      supercell[1,1] = b
      for c in xrange(max(mink//(a*b), 1), maxc+1):
        n = a * b * c
        if n not in sizerange: continue
        supercell[2, 2] = c
        for d, e, f in product(xrange(b), xrange(c), xrange(c)):
          supercell[1, 0] = d
          supercell[2, 0] = e
          supercell[2, 1] = f
          if not isthere(supercell, results[n]): 
            results[n].append(supercell.copy())

  return results

def hf_groups(lattice, sizerange):
  """ Generator over supercells for given size range.
  
      :params lattice: 
         Back-bone lattice
      :type lattice: py:attr:`~pylada.crystal.Structure`
      :param sizerange: 
         List of sizes for which to perform calculations, in number of
         unit-cells per supercell.
      :type sizerange: integer sequence
      :yields: 
          yields a list of 2-tuples where the first element is an hft and the
          second the hermite cell.
  """
  from numpy import dot
  from ..crystal import HFTransform
  result = {}
  for n, cells in supercells(lattice, sizerange).iteritems():
    result = {}
    for cell in cells:
      hft = HFTransform(lattice, dot(lattice.cell, cell))
      key = repr(hft.quotient.tolist())
      if key in result: result[key].append((hft, cell))
      else: result[key] = [(hft, cell)]
    yield result.values()


def generate_bitstrings(lattice, sizerange):
  """ Generator over bitstrings """
  from numpy import dot, all
  from .cppwrappers import NDimIterator, _lexcompare, Manipulations
  transforms = Transforms(lattice)
  for hfgroups in hf_groups(lattice, sizerange):
    for hfgroup in hfgroups:
      # actual results
      ingroup = []
      # stuff we do not want to see again
      outgroup = set()
  
      # translation operators
      translations = Manipulations(transforms.translations(hfgroup[0][0]))
  
      # Creates argument to ndimensional iterator.
      args = []
      size = hfgroup[0][0].size
      for site in transforms.lattice:
        if site.nbflavors == 1: continue
        args += [site.nbflavors] * size
  
      # loop over possible structures in this hfgroup
      for x in NDimIterator(*args):
        strx = ''.join(str(i) for i in x)
        if strx in outgroup: continue
  
        # check for supercell independent transforms.
        # SHOULD INSERT LABEL EXCHANGE HERE
        # loop over translational symmetries.
        subperiodic = False
        for t in translations(x):
          a = _lexcompare(t, x)
          # if a == t, then smaller exists with this structure.
          # also add it to outgroup
          if a >= 0: outgroup.add(''.join(str(i) for i in t))
          if a == 0: subperiodic = True; continue
          # SHOULD INSERT LABEL EXCHANGE HERE
        if not subperiodic: ingroup.append(x.copy())
  
      # loop over cell specific transformations.
      for hft, hermite in hfgroup:
        # get transformations. Not the best way of doing this.
        invariants = transforms.invariant_ops(dot(lattice.cell, hermite))
        transformations = transforms.transformations(hft)
        for j, (t, i) in enumerate(zip(transformations, invariants)):
          if not i: continue
          if all(t == range(t.shape[0])): invariants[i] = False
        transformations = transformations[invariants]
  
        outgroup = set()
        for x in ingroup:
          strx = ''.join(str(i) for i in x)
          if strx in outgroup: continue
          for transform in transformations: 
            t = x[transform]
            a = _lexcompare(t, x)
            if a == 0: continue
            if a > 0: outgroup.add(''.join(str(i) for i in t))
  
            # SHOULD INSERT LABEL EXCHANGE HERE
            # loop over translational symmetries.
            for tt in translations(t):
              a = _lexcompare(tt, x)
              if a > 0: outgroup.add(''.join(str(i) for i in tt))
              # SHOULD INSERT LABEL EXCHANGE HERE
          yield x, hft, hermite
