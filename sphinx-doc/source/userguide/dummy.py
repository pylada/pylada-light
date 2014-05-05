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

def Extract(outdir=None):
  """ An extraction function for a dummy functional """
  from os.path import exists
  from os import getcwd
  from collections import namedtuple
  from pickle import load
  from pylada.misc import Changedir

  if outdir == None: outdir = getcwd()
  Extract = namedtuple('Extract', ['success', 'directory', 'energy', 'structure', 'value', 'functional'])
  if not exists(outdir): return Extract(False, outdir, None, None, None, None)
  with Changedir(outdir) as pwd:
    if not exists('OUTCAR'): return Extract(False, outdir, None, None, None, None)
    with open('OUTCAR', 'r') as file:
      structure, energy, value, functional = load(file)
      return Extract(True, outdir, energy, structure, value, functional)
  
def functional(structure, outdir=None, value=False, **kwargs):
  """ A dummy functional """
  from copy import deepcopy
  from pickle import dump
  from random import random
  from pylada.misc import Changedir

  structure = deepcopy(structure)
  structure.value = value
  with Changedir(outdir) as pwd:
    with open('OUTCAR', 'w') as file: dump((random(), structure, value, functional), file)

  return Extract(outdir)
  return structure
functional.Extract = Extract

def create_jobs():
  """ Simple job-folders. """
  from pylada.jobfolder import JobFolder
  from pylada.crystal.binary import zinc_blende

  root = JobFolder()
  
  for name, value, species in zip( ['diamond', 'diamond/alloy', 'GaAs'], 
                                   [0, 1, 2],
                                   [('Si', 'Si'), ('Si', 'Ge'), ('Ga', 'As')] ):
    job = root / name 
    job.functional = functional
    job.params['value'] = value
    job.params['structure'] = zinc_blende()
    for atom, specie in zip(job.structure, species): atom.type = specie

  return root
