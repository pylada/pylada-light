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

""" Defines binary lattices. """
__docformat__ = "restructuredtext en"
__all__ = ['rock_salt', 'zinc_blende', 'wurtzite']

def rock_salt():
  """ rock_salt lattice """
  from pylada.crystal import Structure
  return Structure( 1, 0, 0,\
                    0, 1, 0,\
                    0, 0, 1,\
                    scale=1, name='Rock-Salt' )\
           .add_atom(0, 0, 0, 'A')\
           .add_atom(0.5, 0.5, 0.5, 'B')

def zinc_blende():
  """ zinc_blende lattice """
  from pylada.crystal import Structure
  return Structure( 0, 0.5, 0.5,\
                    0.5, 0, 0.5,\
                    0.5, 0.5, 0,\
                    scale=1, name='Zinc-Blende' )\
           .add_atom(0, 0, 0, 'A')\
           .add_atom(0.25, 0.25, 0.25, 'B')

def wurtzite():
  """ wurtzite lattice """
  from pylada.crystal import Structure
  return Structure( 0.5, 0.5, 0,\
                    -0.866025, 0.866025, 0,\
                    0, 0, 1,\
                    scale=1, name='Wurtzite' )\
           .add_atom(0.5, 0.288675, 0, 'A')\
           .add_atom(0.5, -0.288675, 0.5, 'A')\
           .add_atom(0.5, 0.288675, 0.25, 'B')\
           .add_atom(0.5, -0.288675, 0.75, 'B')

