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

""" Point-ion charge models.

    This sub-package provides an ewald summation routine and a lennard-johnes
    potential.  The implementations are fairly basic run-of-the-mill fortran
    stuff with a python interface.
"""
__docformat__ = "restructuredtext en"
__all__ = ['ewald']

def ewald(structure, charges=None, cutoff=None, **kwargs):
  """ Ewald summation. 

      Run-of-the-mill Ewald summation. Nothing fancy, so not very fast for
      large structures.

      :param structure:
         The structure to optimize. The charge of each atom can be given as a
         ``charge`` attribute. Otherwise, they should be in the ``charges``
         map.
      :type structure: py:attr:`~pylada.crystal.Structure`
      :param dict charges:
        Map from atomic-types to charges. If not signed by a unit, then should
        be in units of elementary electronic charge. If an atom has a
        ``charge`` attribute, the attribute takes priority ove items in this
        map.
      :param float cutoff:
        Cutoff energy when computing reciprocal space part. Defaults to
        :py:math:`15 Ry`.
  """
  from numpy import array, dot, zeros
  from numpy.linalg import inv
  from quantities import elementary_charge as em, Ry, a0, angstrom
  from .cppwrappers import ewald

  invcell = inv(structure.cell)
  positions = array([dot(invcell, a.pos) for a in structure], dtype='float64')
  cell = structure.cell.copy(order='F') * structure.scale                      \
         * float(angstrom.rescale(a0))
  if cutoff is None: cutoff = 15
  elif hasattr(cutoff, 'rescale'): cutoff = float(cutoff.rescale(Ry))
  else: cutoff = float(cutoff)
  c = []
  for atom in structure:
    if hasattr(atom, 'charge'): 
      if hasattr(atom.charge, 'rescale'): 
        c.append(float(atom.charge.rescale(em)))
        continue
      try: dummy = float(atom.charge)
      except: raise ValueError('charge attribute is not a floating point.')
      else:
        c.append(dummy)
        continue
    if charges is None: 
      raise ValueError( 'Atom has no charge attribute '                        \
                        'and charge dictionary is empty.' )
    if atom.type not in charges:
      raise ValueError( 'Atom has no charge attribute '                        \
                        'and atomic type {0} not in dictionary'                \
                        .format(atom.type) )
    dummy = c[atom.type]
    if hasattr(dummy, 'rescale'): c.append(float(dummy.rescale(em)))
    else:
      try: dummy = float(dummy)
      except: raise ValueError('charge attribute is not a floating point.')
      else: c.append(dummy)

  c = array(c, order='F', dtype='float64')
  energy, force, cforces, stress = ewald(cell, positions, c, cutoff)

  result = structure.copy()
  for atom, force in zip(result, cforces):
    atom.force = force * Ry / a0
  result.energy = energy * Ry
  result.stress = zeros((3, 3), dtype='float64') * Ry
  result.stress[0, 0] = stress[0] * Ry
  result.stress[1, 1] = stress[1] * Ry
  result.stress[2, 2] = stress[2] * Ry
  result.stress[0, 1] = stress[3] * Ry
  result.stress[1, 0] = stress[3] * Ry
  result.stress[1, 2] = stress[4] * Ry
  result.stress[2, 1] = stress[4] * Ry
  result.stress[0, 2] = stress[5] * Ry
  result.stress[2, 0] = stress[5] * Ry
  return result


