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

""" Contains basic data type and methods for crystal structures. """
__docformat__ = "restructuredtext en"
__all__ = [ 'Structure', 'Atom', 'HFTransform', 'zero_centered', 'into_voronoi',
            'into_cell', 'supercell', 'primitive', 'is_primitive', 'space_group',
            'transform', 'periodic_dnc', 'neighbors', 'coordination_shells',
            'splitconfigs', 'vasp_ordered', 'specieset', 'map_sites',
            'which_site', 'iterator' ]
from cppwrappers import Structure, Atom, HFTransform, zero_centered, into_voronoi,    \
                        into_cell, supercell, primitive, is_primitive, space_group,   \
                        transform, periodic_dnc, neighbors, coordination_shells,      \
                        splitconfigs, map_sites
import iterator

def specieset(structure):
  """ Returns ordered set of species.
  
      Especially usefull with VASP since we are sure what the list of species
      is always ordered the same way.
  """
  return set([a.type for a in structure])

def vasp_ordered(structure):
  """ Returns a structure with correct VASP order of ions.
  
      :param structure:
          :class:`Structure` for which to reorder atoms.
  """

  from copy import deepcopy
  result = deepcopy(structure)
  def sortme(self): return a.type.lower()
  result[:] = sorted(structure, sortme)
  return result

def which_site(atom, lattice, invcell=None, tolerance=1e-8):
  """ Index of periodically equivalent atom. 


      :param atom: 
        :py:class:`~cppwrappers.Atom` for which to find periodic equivalent.
      :param lattice:
        :py:class:`~cppwrappers.Structure` defining the periodicity.
      :type lattice: :py:class:`~cppwrappers.Structure` or matrix

      :return: index in list of atoms, or -1 if not found.
  """
  from numpy.linalg import inv
  from .cppwrappers import are_periodic_images as api
  if invcell is None: invcell = inv(lattice.cell)
  lattice = [getattr(site, 'pos', site) for site in lattice]
  pos = getattr(atom, 'pos', atom)
  for i, site in enumerate(lattice):
    if api(pos, site, invcell, tolerance): return i
  return -1


def _normalize_freeze_cell(freeze, periodicity=3):
  """ Transforms freeze parameters into a normalized form. 
  
      The normalized form is a list of six boolean where, if True, each of xx,
      yy, zz, yz, xy, xz is *frozen*. The other forms allow strings, list of
      strings, or the same list of booleans as the output.

      If periodicity is 2, then the degrees of freedom are xx, yy.
  """
  from numpy import array
  if isinstance(freeze, str): freeze = freeze.split()
  if periodicity == 3:
    if len(freeze) == 6                                                          \
       and all(isinstance(u, bool) or isinstance(u, int) for u in freeze):
         return [u == True for u in freeze]
    freeze = set([u.lower() for u in freeze])
    return array([ 'xx' in freeze,
                   'yy' in freeze,
                   'zz' in freeze,
                   ('yz' in freeze or 'zy' in freeze),
                   ('xy' in freeze or 'yx' in freeze),
                   ('xz' in freeze or 'zx' in freeze) ])
  elif periodicity == 2:
    if len(freeze) == 2                                                          \
       and all(isinstance(u, bool) or isinstance(u, int) for u in freeze):
         return [u == True for u in freeze]
    freeze = set([u.lower() for u in freeze])
    return array(['xx' in freeze, 'yy' in freeze])

def _normalize_freeze_atom(freeze):
  """ Transforms freeze parameters into a normalized form. 
  
      The normalized form is a list of 3 boolean where, if True, each of x, y,
      z is *frozen*. The other forms allow strings, list of strings, or the
      same list of booleans as the output.
  """
  from numpy import array
  from ..error import TypeError
  if hasattr(freeze, '__iter__') and len(freeze) == 3                          \
     and all(isinstance(u, bool) or isinstance(u, int) for u in freeze):
       return [u == True for u in freeze]
  elif not hasattr(freeze, 'lower'):
    raise TypeError('Could not make sense of freeze parameter.')
  freeze = freeze.lower()
  return array(['x' in freeze, 'y' in freeze, 'z' in freeze])
