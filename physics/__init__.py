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

""" Physical quantities. 


    Holds things related to physical quantities, eg atomic units using the
    *quantities* package.
"""
__docformat__ = "restructuredtext en"
__all__ = [ 'a0', 'bohr_radius', 'h', 'planck', 'h_bar', 'reduced_planck',
            'electronic_mass', 'Ry', 'rydberg', 'Kb', 'boltzmann' ]

import quantities as pq

h = pq.UnitQuantity("planck", 4.1356673310e-15 * pq.eV * pq.s)
""" Planck's constant. """
planck = h
""" Planck's constant. """
h_bar = pq.UnitQuantity("h_bar", h / 2e0 / pq.pi, symbol='h_bar')
""" Reduced Planck's constant. """
reduced_plank = h_bar
""" Reduced Planck's constant. """

Ry = pq.UnitQuantity('Rydberg', 0.5 * pq.hartree, symbol='Ry')
""" Rydberg energy units. """
rydberg = Ry
""" Rydberg energy units. """

a0 = pq.UnitQuantity('bohr_radius', 0.529177249 * pq.angstrom, symbol='a0')
""" Bohr radius, unit of length of atomic units. """
bohr_radius = a0
""" Bohr radius, unit of length of atomic units. """

emass = pq.UnitQuantity( "electronic_mass", h_bar**2 / (2e0 * Ry * a0**2),
                         symbol='m_e' )
""" Mass of the electron at rest.

    The value is obtained from a formula. It comes close enough and makes the
    Rydberg units consistent.
"""


Kb = pq.UnitQuantity("boltzmann", 8.617 * pq.eV / pq.K, symbol='Kb')
""" Boltzmann's constant. """
boltzmann = Kb
""" Boltzmann's constant. """

reduced_reciprocal_au = pq.UnitQuantity("reduced_reciprocal_au",
                                        2e0*pq.pi/a0, symbol='2pi/a0')

if 'planck' not in pq.__dict__:      pq.planck          = planck
if 'h_bar' not in pq.__dict__:       pq.h_bar           = h_bar
if 'Ry' not in pq.__dict__:          pq.Ry              = Ry
if 'a0' not in pq.__dict__:          pq.a0              = a0
if 'bohr_radius' not in pq.__dict__: pq.bohr_radius     = bohr_radius
if 'boltzmann' not in pq.__dict__:   pq.boltzmann       = boltzmann  
if 'emass' not in pq.__dict__:       pq.emass           = emass  
