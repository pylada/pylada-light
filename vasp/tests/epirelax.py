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

def strain(x, direction, structure):
  """ Creates new structure with given input strain in c direction. """
  from numpy.linalg import inv
  from numpy import outer, dot
  result = structure.copy()
  # define strain matrix
  strain = outer(direction, direction) * x 
  # strain cell matrix and atoms
  result.cell = structure.cell + dot(strain, structure.cell)
  for atom in result: atom.pos += dot(strain, atom.pos)
  # return result.
  return result

def function(vasp, structure, outdir, x, direction, **kwargs):
  from os.path import join
  directory = join(outdir, join("relax_ions", "{0:0<12.10}".format(x)))
  return vasp(strain(x, direction, structure), directory, relaxation='ionic', **kwargs)

def component(extract, direction):
  """ Returns relevant stress component. """
  from numpy import dot
  return dot(dot(direction, extract.stress), direction)

def epitaxial(vasp, structure, outdir=None, direction=[0,0,1], epiconv = 1e-4,
              initstep=0.05, **kwargs):
  """
      Performs epitaxial relaxation in given direction. 
  
      Performs a relaxation for an epitaxial structure on a virtual substrate.
      The external (cell) coordinates of the structure can only relax in the
      growth/epitaxial direction. Internal coordinates (ions), however, are
      allowed to relax in whatever direction. 
      
      Since VASP does not intrinsically allow for such a relaxation, it is
      performed by chaining different vasp calculations together. The
      minimization procedure itself is the secant method, enhanced by the
      knowledge of the stress tensor. The last calculation is static, for
      maximum accuracy.

      :param vasp: 
        :py:class:`Vasp <pylada.vasp.Vasp>` functional with wich to perform the
        relaxation.
      :param structure:
        :py:class:`Structure <pylada.crystal.Structure>` for which to perform the
        relaxation.
      :param str outdir: 
        Directory where to perform calculations. If None, defaults to current
        working directory. The intermediate calculations are stored in the
        relax_ions subdirectory.
      :param direction:
        Epitaxial direction. Defaults to [0, 0, 1].
      :param float epiconv: 
        Convergence criteria of the total energy.
  """
  from os import getcwd
  from copy import deepcopy
  from numpy.linalg import norm
  from numpy import array

  # takes care of input parameters.
  if outdir is None: outdir = getcwd
  vasp = deepcopy(vasp) # keep function stateless!
  direction = array(direction) / norm(direction)
  kwargs.pop('relaxation', None) # we will be saying what to optimize.
  if 'ediff' in kwargs: vasp.ediff = kwargs.pop('ediff')
  if vasp.ediff < epiconv: vasp.ediff = epiconv * 1e-2

  # Compute initial structure.
  xstart = 0.0
  estart = function(vasp, structure, outdir, xstart, direction, **kwargs)
  
  # then checks stress for direction in which to release strain.
  stress_direction = 1.0 if component(estart, direction) > 0e0 else -1.0
  xend = initstep if stress_direction > 0e0 else -initstep
  eend = function(vasp, structure, outdir, xend, direction, **kwargs)

  # make sure xend is on other side of stress tensor sign.
  while stress_direction * component(eend, direction) > 0e0:
    xstart, estart = xend, eend
    xend += initstep if stress_direction > 0e0 else -initstep
    eend = function(vasp, structure, outdir, xend, direction, **kwargs)
  
  # now we have a bracket. We start bisecting it.
  while abs(estart.total_energy - eend.total_energy) > epiconv * float(len(structure)):
    # bisect interval and launch vasp.
    xmid = 0.5 * (xend + xstart)
    emid = function(vasp, structure, outdir, xmid, direction, **kwargs)
    # change interval depending on strain.
    if stress_direction * component(emid, direction) > 0: xstart, estart = xmid, emid
    else: xend, eend = xmid, emid

  # Finally, perform one last static calculation.
  efinal = eend if estart.total_energy > eend.total_energy else estart
  return vasp(efinal.structure, outdir=outdir, restart=efinal, relaxation='static', **kwargs)
