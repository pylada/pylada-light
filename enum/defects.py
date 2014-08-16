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

class Iterator(object):

  def __init__(self, size, *args): 
    super(Iterator, self).__init__()
    self.others = args
    self.size = size
    self.reset()
    
  def __iter__(self): return self

  def reset(self):
    from numpy import count_nonzero, ones, logical_and
    from ..error import ValueError
    from .cppwrappers import FCIterator

    # creates list of color iterators, as per Hart, Nelson, Forcade.
    self.iterators = []
    current_allowed = ones(self.size, dtype='bool')
    for n, color, mask in self.others:
      allowed = logical_and(current_allowed, mask)
      length = count_nonzero(allowed)
      if length < n:
        raise ValueError( 'Could not create iterator, '                        \
                          'concentrations do not add up.')
      self.iterators.append(FCIterator(length, n))
      current_allowed[allowed] = [False] * n + [True] * (length-n)
    # necessary cos we bypass calling next for the first time in this
    # instance's next.
    for iter in self.iterators[1:]: iter.next()

    # allocate memory now so we don't have to do it everytime next is called.
    self.x = ones(self.size, dtype='int16')
    self._masks = ones((2, len(self.x)), dtype='bool')

  def next(self):
    from numpy import logical_and
    from ..error import internal
      
    # reset x and mask to default values.
    self.x[:] = 0
    self._masks[:] = True
    mask = self._masks[0]
    change_color = self._masks[1]
    # do next determines which sub-iterators to call next on.
    donext = True
    # we loop over each color in turn
    for iter, (n, color, cmask) in zip(self.iterators, self.others):
      # as long as donext is True, we call next.
      # once it is false (eg StopIteration was not raised) we are done
      # incrementing subiterators.
      if donext: 
        try: bitstring = iter.next()
        except StopIteration:
          iter.reset()
          try: bitstring = iter.next()
          except StopIteration: 
            raise internal('Cannot iterate over type {0}'.format(color))
        else: donext = False
      else: bitstring = iter.yielded
      # change_color is True for those sites the current bitstring can access
      logical_and(cmask, mask, change_color)
      # now only those sites which are on in the relevant bitstring are true
      change_color[change_color] = bitstring
      # we can set x to the relevant color
      self.x[change_color] = color
      # and turn off the bits which have just changed color
      mask[change_color] = False

    # if do next is true, then we have reached the end of the loop
    if donext: raise StopIteration

    # otherwise, return a reference to the current x
    return self.x

def defects(lattice, cellsize, defects):
  """ Generates defects on a lattice """
  from numpy import zeros, dot, all
  from .transforms import Transforms
  from .cppwrappers import _lexcompare, Manipulations
  from . import hf_groups

  # sanity check
  if len(defects) == 0: return

  transforms = Transforms(lattice)
  lattice = transforms.lattice.copy()

  # find the number of active sites.
  nsites = len([0 for u in lattice if u.nbflavors > 1])

  # creates arguments
  args = []
  iterator = enumerate(defects.iteritems())
  # first guy is special cos he will be locked in place to avoid unecessary
  # translations
  i, (specie, n) = iterator.next()
  nsize = len([0 for u in lattice if u.nbflavors > 1])*cellsize
  firstmask = zeros(nsize, dtype='bool' )
  firstcolor = i+1
  for site in lattice:
    if site.nbflavors == 1: continue
    if not site.asymmetric: continue
    if specie not in site.type: continue
    firstmask[site.index*cellsize] = True
  args.append((1, firstcolor, firstmask))
  # now add rest of first guys, if any. They may go anywhere their pal could
  # go, including the same site, as there may be more than one he could occupy,
  # and he is not blessed with  bilocation.
  if n > 1: 
    mask = zeros(nsize, dtype='bool')
    for site in lattice:
      if site.nbflavors == 1: continue
      if specie not in site.type: continue
      mask[cellsize*site.index:cellsize*(site.index+1)] = True
    args.append((n-1, firstcolor, mask))

  for i, (specie, n) in iterator:
    # remove from lattice the reference to this specie
    mask = zeros(nsize, dtype='bool')
    color = i+1
    for site in lattice:
      if site.nbflavors == 1: continue
      if specie not in site.type: continue
      mask[cellsize*site.index:cellsize*(site.index+1)] = True
    args.append((n, color, mask))

  # now we can create the template and the iterator.
  xiterator = Iterator(len(mask), *args)

  # loop over groups directly, not sizes
  for hfgroup in hf_groups(lattice, [cellsize]).next():
    # actual results
    ingroup = []
    # stuff we do not want to see again
    outgroup = set()
  
    # translation operators
    translations = Manipulations(transforms.translations(hfgroup[0][0]))
    # reset iterator
    xiterator.reset()
    for x in xiterator:
      strx = ''.join(str(i) for i in x)
      if strx in outgroup: continue
  
      # check for supercell independent transforms.
      # loop over translational symmetries.
      for t in translations(x):
        # Translation may move the first guy out of position (and not replace
        # with another guy). We can safely ignore those.
        if all(t[firstmask] != firstcolor): continue
        a = _lexcompare(t, x) 
        # if a == t, then smaller exists with this structure.
        # also add it to outgroup.
        if a > 0: outgroup.add(''.join(str(i) for i in t))
      ingroup.append(x.copy())
    print len(ingroup)
  
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
  
          # loop over translational symmetries.
          for tt in translations(t):
            # rotations + translations may move the first guy out of
            # position. We can ignore those translations.
            if all(t[firstmask] != firstcolor): continue
            a = _lexcompare(tt, x)
            if a > 0: outgroup.add(''.join(str(i) for i in tt))
        yield x, hft, hermite
