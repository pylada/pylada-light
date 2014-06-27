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

# Structure definition.
from pylada.crystal import Structure
from pylada.crystal.defects import third_order_charge_correction
from quantities import eV

structure = Structure()
structure.name   = 'Ga2CdO4: b5'
structure.scale  = 1.0
structure.energy = -75.497933000000003
structure.weight = 1.0
structure.set_cell = (-0.0001445, 4.3538020, 4.3537935),\
                     (4.3538700, -0.0001445, 4.3538615),\
                     (4.3538020, 4.3537935, -0.0001445)
structure.add_atoms = [(7.61911540668, 7.61923876219, 7.61912846850), 'Cd'],\
                      [(1.08833559332, 1.08834823781, 1.08832253150), 'Cd'],\
                      [(4.35372550000, 4.35379350000, 4.35372550000), 'Ga'],\
                      [(4.35379775000, 2.17685850000, 2.17682450000), 'Ga'],\
                      [(2.17682450000, 4.35386575000, 2.17682875000), 'Ga'],\
                      [(2.17682875000, 2.17686275000, 4.35379775000), 'Ga'],\
                      [(2.32881212361, 2.32884849688, 2.32881647755),  'O'],\
                      [(2.32887187256, 4.20174404476, 4.20169148188),  'O'],\
                      [(4.20168277385, 2.32891695560, 4.20168347161),  'O'],\
                      [(4.20168782554, 4.20174474241, 2.32887622633),  'O'],\
                      [(6.37863887654, 6.37873414925, 6.37863016865),  'O'],\
                      [(6.37857477364, 4.50584295539, 4.50575516433),  'O'],\
                      [(4.50576822615, 6.37867004441, 4.50576752839),  'O'],\
                      [(4.50576317445, 4.50584225759, 6.37857477367),  'O']

# this is converged to less than 1meV
third = third_order_charge_correction(structure, epsilon=10.0, n=20)
print "third =", third
assert abs(third - 0.11708438633232088*eV) < 1e-12
# to compare to Stephan Lany's 3rd0 (-1.299), multiply third by epsilon / (1.0 - 1.0/epsilon)
# tries another cell. Jumping hoops since  can't modify cell component directly.
cell = structure.cell.copy()
cell[0,0] = -30.0
structure.cell = cell
third = third_order_charge_correction(structure, epsilon=10.0, n=20)
print "third =", third
assert abs(third - 0.17412996139970385*eV) < 1e-12
# to compare to Stephan Lany's 3rd0 (-1.928), multiply third by epsilon / (1.0 - 1.0/epsilon)


