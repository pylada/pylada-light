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

""" Physical quantities of elements. 

    Atomic quantities can be used as in:

    .. python::
    
      import pylada.periodic_table
      periodic_table.Au.atomic_weight
      periodic_table.Gold.atomic_weight

    Available quantities can be found in `Element`. Some of
    these are set to None when either meaningless or not available.
"""
__docformat__ = "restructuredtext en"


from _create_data import *
from _element import Element

from numpy import array
from quantities import cm, m, g, J, pm, K as Kelvin, mol, angstrom,\
                         dimensionless, W as Watt, GPa, s, ohm
H = Element(**{'mulliken_jaffe': 2.25, 'single_bond_radius': array(32.0) * pm, 'pauling': 2.2000000000000002, 'molar_volume': array(11.42) * cm**3/mol, 'atomization': array(218000.0) * J/mol, 'sound_velocity': array(1270.0) * m/s, 'sanderson': 2.5899999999999999, 'atomic_weight': 1.0079400000000001, 'critical_temperature': array(33.0) * Kelvin, 'melting_point': array(14.01) * Kelvin, 'allen': 2.2999999999999998, 'thermal_conductivity': array(0.18049999999999999) * Watt/(m*Kelvin), 'ionization_energies': [array(1312000.0) * J/mol], 'vaporization': array(452.0) * J/mol, 'atomic_number': 1, 'symbol': 'H', 'covalent_radius': array(31.0) * pm, 'fusion': array(558.0) * J/mol, 'van_der_waals_radius': array(120.0) * pm, 'electron_affinity': array(72800.0) * J/mol, 'name': 'Hydrogen', 'boiling_point': array(20.280000000000001) * Kelvin, 'allred_rochow': 2.2000000000000002, 'atomic_radius': array(25.0) * pm})
"""
Hydrogen elemental data.

  - mulliken_jaffe: 2.25
  - single_bond_radius: 32.0 pm
  - pauling: 2.2
  - molar_volume: 11.42 cm**3/mol
  - atomization: 218000.0 J/mol
  - sound_velocity: 1270.0 m/s
  - sanderson: 2.59
  - atomic_weight: 1.00794
  - critical_temperature: 33.0 K
  - melting_point: 14.01 K
  - allen: 2.3
  - thermal_conductivity: 0.1805 W/(m*K)
  - ionization_energies: [array(1312000.0) * J/mol]
  - vaporization: 452.0 J/mol
  - atomic_number: 1
  - symbol: H
  - covalent_radius: 31.0 pm
  - fusion: 558.0 J/mol
  - van_der_waals_radius: 120.0 pm
  - electron_affinity: 72800.0 J/mol
  - name: Hydrogen
  - boiling_point: 20.28 K
  - allred_rochow: 2.2
  - atomic_radius: 25.0 pm
"""

He = Element(**{'mulliken_jaffe': 3.4900000000000002, 'electron_affinity': array(0.0) * J/mol, 'single_bond_radius': array(46.0) * pm, 'name': 'Helium', 'thermal_conductivity': array(0.15129999999999999) * Watt/(m*Kelvin), 'atomic_number': 2, 'molar_volume': array(21.0) * cm**3/mol, 'covalent_radius': array(28.0) * pm, 'atomization': array(0.0) * J/mol, 'sound_velocity': array(970.0) * m/s, 'boiling_point': array(4.2199999999999998) * Kelvin, 'fusion': array(20.0) * J/mol, 'atomic_weight': 4.0026020000000004, 'critical_temperature': array(5.1900000000000004) * Kelvin, 'ionization_energies': [array(2372300.0) * J/mol, array(5250500.0) * J/mol], 'melting_point': array(0.94999999999999996) * Kelvin, 'van_der_waals_radius': array(140.0) * pm, 'vaporization': array(83.0) * J/mol, 'symbol': 'He', 'allred_rochow': 5.5})
"""
Helium elemental data.

  - mulliken_jaffe: 3.49
  - single_bond_radius: 46.0 pm
  - molar_volume: 21.0 cm**3/mol
  - atomization: 0.0 J/mol
  - sound_velocity: 970.0 m/s
  - atomic_weight: 4.002602
  - critical_temperature: 5.19 K
  - melting_point: 0.95 K
  - thermal_conductivity: 0.1513 W/(m*K)
  - ionization_energies: [array(2372300.0) * J/mol, array(5250500.0) * J/mol]
  - vaporization: 83.0 J/mol
  - atomic_number: 2
  - symbol: He
  - covalent_radius: 28.0 pm
  - fusion: 20.0 J/mol
  - van_der_waals_radius: 140.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Helium
  - boiling_point: 4.22 K
  - allred_rochow: 5.5
"""

Li = Element(**{'mulliken_jaffe': 0.96999999999999997, 'single_bond_radius': array(133.0) * pm, 'pauling': 0.97999999999999998, 'molar_volume': array(13.02) * cm**3/mol, 'bulk_modulus': array(11.0) * GPa, 'sound_velocity': array(6000.0) * m/s, 'sanderson': 0.89000000000000001, 'atomic_weight': 6.9409999999999998, 'critical_temperature': array(3223.0) * Kelvin, 'melting_point': array(453.69) * Kelvin, 'allen': 0.91200000000000003, 'orbital_radii': (array(0.521239590265) * angstrom, array(0.33073578062499998) * angstrom), 'thermal_conductivity': array(85.0) * Watt/(m*Kelvin), 'ionization_energies': [array(520200.00000000006) * J/mol, array(7298100.0) * J/mol, array(11815000.0) * J/mol], 'vaporization': array(147000.0) * J/mol, 'atomic_number': 3, 'rigidity_modulus': array(4.2000000000000002) * GPa, 'symbol': 'Li', 'covalent_radius': array(128.0) * pm, 'fusion': array(3000.0) * J/mol, 'pettifor': 0.45000000000000001, 'atomization': array(159000.0) * J/mol, 'van_der_waals_radius': array(182.0) * pm, 'electron_affinity': array(59600.0) * J/mol, 'name': 'Lithium', 'boiling_point': array(1615.0) * Kelvin, 'density': array(0.53500000000000003) * g*cm**3, 'double_bond_radius': array(124.0) * pm, 'allred_rochow': 0.96999999999999997, 'young_modulus': array(4.9000000000000004) * GPa, 'thermal_expansion': array(4.6e-05) * 1/Kelvin, 'atomic_radius': array(145.0) * pm})
"""
Lithium elemental data.

  - mulliken_jaffe: 0.97
  - single_bond_radius: 133.0 pm
  - pauling: 0.98
  - molar_volume: 13.02 cm**3/mol
  - atomization: 159000.0 J/mol
  - sound_velocity: 6000.0 m/s
  - sanderson: 0.89
  - atomic_weight: 6.941
  - critical_temperature: 3223.0 K
  - melting_point: 453.69 K
  - allen: 0.912
  - orbital_radii: (array(0.521239590265) * angstrom, array(0.33073578062499998) * angstrom)
  - thermal_conductivity: 85.0 W/(m*K)
  - ionization_energies: [array(520200.00000000006) * J/mol, array(7298100.0) * J/mol, array(11815000.0) * J/mol]
  - vaporization: 147000.0 J/mol
  - atomic_number: 3
  - rigidity_modulus: 4.2 GPa
  - symbol: Li
  - covalent_radius: 128.0 pm
  - fusion: 3000.0 J/mol
  - pettifor: 0.45
  - bulk_modulus: 11.0 GPa
  - van_der_waals_radius: 182.0 pm
  - electron_affinity: 59600.0 J/mol
  - name: Lithium
  - boiling_point: 1615.0 K
  - density: 0.535 g*cm**3
  - double_bond_radius: 124.0 pm
  - allred_rochow: 0.97
  - young_modulus: 4.9 GPa
  - thermal_expansion: 4.6e-05 1/K
  - atomic_radius: 145.0 pm
"""

Be = Element(**{'mulliken_jaffe': 1.54, 'single_bond_radius': array(102.0) * pm, 'pauling': 1.5700000000000001, 'molar_volume': array(4.8499999999999996) * cm**3/mol, 'bulk_modulus': array(130.0) * GPa, 'sound_velocity': array(13000.0) * m/s, 'sanderson': 1.8100000000000001, 'atomic_weight': 9.0121819999999992, 'triple_bond_radius': array(85.0) * pm, 'melting_point': array(1560.0) * Kelvin, 'allen': 1.5760000000000001, 'orbital_radii': (array(0.33867343935999999) * angstrom, array(0.23283798955999999) * angstrom), 'thermal_conductivity': array(190.0) * Watt/(m*Kelvin), 'ionization_energies': [array(899500.0) * J/mol, array(1757100.0) * J/mol, array(14848700.0) * J/mol, array(21006600.0) * J/mol], 'vaporization': array(297000.0) * J/mol, 'atomic_number': 4, 'rigidity_modulus': array(132.0) * GPa, 'symbol': 'Be', 'covalent_radius': array(96.0) * pm, 'fusion': array(7950.0) * J/mol, 'pettifor': 1.5, 'atomization': array(324000.0) * J/mol, 'poisson_ratio': array(0.032000000000000001) * dimensionless, 'electron_affinity': array(0.0) * J/mol, 'name': 'Beryllium', 'boiling_point': array(2742.0) * Kelvin, 'density': array(1.8480000000000001) * g*cm**3, 'double_bond_radius': array(90.0) * pm, 'allred_rochow': 1.47, 'young_modulus': array(287.0) * GPa, 'thermal_expansion': array(1.13e-05) * 1/Kelvin, 'atomic_radius': array(105.0) * pm})
"""
Beryllium elemental data.

  - mulliken_jaffe: 1.54
  - single_bond_radius: 102.0 pm
  - pauling: 1.57
  - molar_volume: 4.85 cm**3/mol
  - atomization: 324000.0 J/mol
  - sound_velocity: 13000.0 m/s
  - sanderson: 1.81
  - atomic_weight: 9.012182
  - triple_bond_radius: 85.0 pm
  - melting_point: 1560.0 K
  - allen: 1.576
  - orbital_radii: (array(0.33867343935999999) * angstrom, array(0.23283798955999999) * angstrom)
  - thermal_conductivity: 190.0 W/(m*K)
  - ionization_energies: [array(899500.0) * J/mol, array(1757100.0) * J/mol, array(14848700.0) * J/mol, array(21006600.0) * J/mol]
  - vaporization: 297000.0 J/mol
  - atomic_number: 4
  - rigidity_modulus: 132.0 GPa
  - symbol: Be
  - covalent_radius: 96.0 pm
  - fusion: 7950.0 J/mol
  - pettifor: 1.5
  - bulk_modulus: 130.0 GPa
  - poisson_ratio: 0.032 dimensionless
  - electron_affinity: 0.0 J/mol
  - name: Beryllium
  - boiling_point: 2742.0 K
  - density: 1.848 g*cm**3
  - double_bond_radius: 90.0 pm
  - allred_rochow: 1.47
  - young_modulus: 287.0 GPa
  - thermal_expansion: 1.13e-05 1/K
  - atomic_radius: 105.0 pm
"""

B = Element(**{'mulliken_jaffe': 2.04, 'single_bond_radius': array(85.0) * pm, 'pauling': 2.04, 'molar_volume': array(4.3899999999999997) * cm**3/mol, 'bulk_modulus': array(320.0) * GPa, 'sound_velocity': array(16200.0) * m/s, 'sanderson': 2.2799999999999998, 'atomic_weight': 10.811, 'triple_bond_radius': array(73.0) * pm, 'melting_point': array(2349.0) * Kelvin, 'allen': 2.0510000000000002, 'orbital_radii': (array(0.25400507951999995) * angstrom, array(0.16669083343499999) * angstrom), 'thermal_conductivity': array(27.0) * Watt/(m*Kelvin), 'ionization_energies': [array(800600.0) * J/mol, array(2427100.0) * J/mol, array(3659700.0) * J/mol, array(25025800.0) * J/mol, array(32826699.999999996) * J/mol], 'vaporization': array(507000.0) * J/mol, 'atomic_number': 5, 'symbol': 'B', 'covalent_radius': array(84.0) * pm, 'fusion': array(50000.0) * J/mol, 'pettifor': 2.0, 'atomization': array(563000.0) * J/mol, 'electron_affinity': array(26700.0) * J/mol, 'name': 'Boron', 'boiling_point': array(4200.0) * Kelvin, 'density': array(2.46) * g*cm**3, 'double_bond_radius': array(78.0) * pm, 'allred_rochow': 2.0099999999999998, 'thermal_expansion': array(6.0000000000000002e-06) * 1/Kelvin, 'atomic_radius': array(85.0) * pm})
"""
Boron elemental data.

  - mulliken_jaffe: 2.04
  - single_bond_radius: 85.0 pm
  - pauling: 2.04
  - molar_volume: 4.39 cm**3/mol
  - atomization: 563000.0 J/mol
  - sound_velocity: 16200.0 m/s
  - sanderson: 2.28
  - atomic_weight: 10.811
  - triple_bond_radius: 73.0 pm
  - melting_point: 2349.0 K
  - allen: 2.051
  - orbital_radii: (array(0.25400507951999995) * angstrom, array(0.16669083343499999) * angstrom)
  - thermal_conductivity: 27.0 W/(m*K)
  - ionization_energies: [array(800600.0) * J/mol, array(2427100.0) * J/mol, array(3659700.0) * J/mol, array(25025800.0) * J/mol, array(32826699.999999996) * J/mol]
  - vaporization: 507000.0 J/mol
  - atomic_number: 5
  - symbol: B
  - covalent_radius: 84.0 pm
  - fusion: 50000.0 J/mol
  - pettifor: 2.0
  - bulk_modulus: 320.0 GPa
  - electron_affinity: 26700.0 J/mol
  - name: Boron
  - boiling_point: 4200.0 K
  - density: 2.46 g*cm**3
  - double_bond_radius: 78.0 pm
  - allred_rochow: 2.01
  - thermal_expansion: 6e-06 1/K
  - atomic_radius: 85.0 pm
"""

C = Element(**{'mulliken_jaffe': 2.48, 'single_bond_radius': array(75.0) * pm, 'pauling': 2.5499999999999998, 'molar_volume': array(5.29) * cm**3/mol, 'bulk_modulus': array(33.0) * GPa, 'sound_velocity': array(18350.0) * m/s, 'sanderson': 2.75, 'atomic_weight': 12.0107, 'triple_bond_radius': array(60.0) * pm, 'melting_point': array(3800.0) * Kelvin, 'allen': 2.544, 'orbital_radii': (array(0.20637912711) * angstrom, array(0.13229431224999999) * angstrom), 'thermal_conductivity': array(140.0) * Watt/(m*Kelvin), 'ionization_energies': [array(1086500.0) * J/mol, array(2352600.0) * J/mol, array(4620500.0) * J/mol, array(6222700.0) * J/mol, array(37831000.0) * J/mol, array(47277000.0) * J/mol], 'vaporization': array(715000.0) * J/mol, 'atomic_number': 6, 'symbol': 'C', 'covalent_radius': array(69.0) * pm, 'pettifor': 2.5, 'atomization': array(717000.0) * J/mol, 'van_der_waals_radius': array(170.0) * pm, 'electron_affinity': array(153900.0) * J/mol, 'name': 'Carbon', 'boiling_point': array(4300.0) * Kelvin, 'density': array(2.2669999999999999) * g*cm**3, 'double_bond_radius': array(67.0) * pm, 'allred_rochow': 2.5, 'thermal_expansion': array(7.0999999999999989e-06) * 1/Kelvin, 'atomic_radius': array(70.0) * pm})
"""
Carbon elemental data.

  - mulliken_jaffe: 2.48
  - single_bond_radius: 75.0 pm
  - pauling: 2.55
  - molar_volume: 5.29 cm**3/mol
  - atomization: 717000.0 J/mol
  - sound_velocity: 18350.0 m/s
  - sanderson: 2.75
  - atomic_weight: 12.0107
  - triple_bond_radius: 60.0 pm
  - melting_point: 3800.0 K
  - allen: 2.544
  - orbital_radii: (array(0.20637912711) * angstrom, array(0.13229431224999999) * angstrom)
  - thermal_conductivity: 140.0 W/(m*K)
  - ionization_energies: [array(1086500.0) * J/mol, array(2352600.0) * J/mol, array(4620500.0) * J/mol, array(6222700.0) * J/mol, array(37831000.0) * J/mol, array(47277000.0) * J/mol]
  - vaporization: 715000.0 J/mol
  - atomic_number: 6
  - symbol: C
  - covalent_radius: 69.0 pm
  - pettifor: 2.5
  - bulk_modulus: 33.0 GPa
  - van_der_waals_radius: 170.0 pm
  - electron_affinity: 153900.0 J/mol
  - name: Carbon
  - boiling_point: 4300.0 K
  - density: 2.267 g*cm**3
  - double_bond_radius: 67.0 pm
  - allred_rochow: 2.5
  - thermal_expansion: 7.1e-06 1/K
  - atomic_radius: 70.0 pm
"""

N = Element(**{'mulliken_jaffe': 2.8999999999999999, 'single_bond_radius': array(71.0) * pm, 'pauling': 3.04, 'molar_volume': array(13.539999999999999) * cm**3/mol, 'atomization': array(473000.0) * J/mol, 'sound_velocity': array(333.60000000000002) * m/s, 'sanderson': 3.1899999999999999, 'atomic_weight': 14.0067, 'critical_temperature': array(126.2) * Kelvin, 'triple_bond_radius': array(54.0) * pm, 'melting_point': array(63.049999999999997) * Kelvin, 'allen': 3.0659999999999998, 'orbital_radii': (array(0.17462849216999998) * angstrom, array(0.11112722228999999) * angstrom), 'thermal_conductivity': array(0.025829999999999999) * Watt/(m*Kelvin), 'ionization_energies': [array(1402300.0) * J/mol, array(2856000.0) * J/mol, array(4578100.0) * J/mol, array(7475000.0) * J/mol, array(9444900.0) * J/mol, array(53266600.0) * J/mol, array(64360000.0) * J/mol], 'vaporization': array(2790.0) * J/mol, 'atomic_number': 7, 'symbol': 'N', 'covalent_radius': array(71.0) * pm, 'fusion': array(360.0) * J/mol, 'pettifor': 3.0, 'van_der_waals_radius': array(155.0) * pm, 'electron_affinity': array(7000.0) * J/mol, 'name': 'Nitrogen', 'boiling_point': array(77.359999999999999) * Kelvin, 'double_bond_radius': array(60.0) * pm, 'allred_rochow': 3.0699999999999998, 'atomic_radius': array(65.0) * pm})
"""
Nitrogen elemental data.

  - mulliken_jaffe: 2.9
  - single_bond_radius: 71.0 pm
  - pauling: 3.04
  - molar_volume: 13.54 cm**3/mol
  - atomization: 473000.0 J/mol
  - sound_velocity: 333.6 m/s
  - sanderson: 3.19
  - atomic_weight: 14.0067
  - critical_temperature: 126.2 K
  - triple_bond_radius: 54.0 pm
  - melting_point: 63.05 K
  - allen: 3.066
  - orbital_radii: (array(0.17462849216999998) * angstrom, array(0.11112722228999999) * angstrom)
  - thermal_conductivity: 0.02583 W/(m*K)
  - ionization_energies: [array(1402300.0) * J/mol, array(2856000.0) * J/mol, array(4578100.0) * J/mol, array(7475000.0) * J/mol, array(9444900.0) * J/mol, array(53266600.0) * J/mol, array(64360000.0) * J/mol]
  - vaporization: 2790.0 J/mol
  - atomic_number: 7
  - symbol: N
  - covalent_radius: 71.0 pm
  - fusion: 360.0 J/mol
  - pettifor: 3.0
  - van_der_waals_radius: 155.0 pm
  - electron_affinity: 7000.0 J/mol
  - name: Nitrogen
  - boiling_point: 77.36 K
  - double_bond_radius: 60.0 pm
  - allred_rochow: 3.07
  - atomic_radius: 65.0 pm
"""

O = Element(**{'mulliken_jaffe': 3.4100000000000001, 'single_bond_radius': array(63.0) * pm, 'pauling': 3.4399999999999999, 'molar_volume': array(17.359999999999999) * cm**3/mol, 'atomization': array(249000.0) * J/mol, 'sound_velocity': array(317.5) * m/s, 'sanderson': 3.6499999999999999, 'atomic_weight': 15.9994, 'critical_temperature': array(154.59999999999999) * Kelvin, 'triple_bond_radius': array(53.0) * pm, 'melting_point': array(54.799999999999997) * Kelvin, 'allen': 3.6099999999999999, 'orbital_radii': (array(0.15081551596499998) * angstrom, array(0.095251904819999983) * angstrom), 'thermal_conductivity': array(0.026579999999999999) * Watt/(m*Kelvin), 'ionization_energies': [array(1313900.0) * J/mol, array(3388300.0) * J/mol, array(5300500.0) * J/mol, array(7469200.0) * J/mol, array(10989500.0) * J/mol, array(13326500.0) * J/mol, array(71330000.0) * J/mol, array(84078000.0) * J/mol], 'vaporization': array(3410.0) * J/mol, 'atomic_number': 8, 'symbol': 'O', 'covalent_radius': array(66.0) * pm, 'fusion': array(222.0) * J/mol, 'pettifor': 3.5, 'van_der_waals_radius': array(152.0) * pm, 'electron_affinity': array(141000.0) * J/mol, 'name': 'Oxygen', 'boiling_point': array(90.200000000000003) * Kelvin, 'double_bond_radius': array(57.0) * pm, 'allred_rochow': 3.5, 'atomic_radius': array(60.0) * pm})
"""
Oxygen elemental data.

  - mulliken_jaffe: 3.41
  - single_bond_radius: 63.0 pm
  - pauling: 3.44
  - molar_volume: 17.36 cm**3/mol
  - atomization: 249000.0 J/mol
  - sound_velocity: 317.5 m/s
  - sanderson: 3.65
  - atomic_weight: 15.9994
  - critical_temperature: 154.6 K
  - triple_bond_radius: 53.0 pm
  - melting_point: 54.8 K
  - allen: 3.61
  - orbital_radii: (array(0.15081551596499998) * angstrom, array(0.095251904819999983) * angstrom)
  - thermal_conductivity: 0.02658 W/(m*K)
  - ionization_energies: [array(1313900.0) * J/mol, array(3388300.0) * J/mol, array(5300500.0) * J/mol, array(7469200.0) * J/mol, array(10989500.0) * J/mol, array(13326500.0) * J/mol, array(71330000.0) * J/mol, array(84078000.0) * J/mol]
  - vaporization: 3410.0 J/mol
  - atomic_number: 8
  - symbol: O
  - covalent_radius: 66.0 pm
  - fusion: 222.0 J/mol
  - pettifor: 3.5
  - van_der_waals_radius: 152.0 pm
  - electron_affinity: 141000.0 J/mol
  - name: Oxygen
  - boiling_point: 90.2 K
  - double_bond_radius: 57.0 pm
  - allred_rochow: 3.5
  - atomic_radius: 60.0 pm
"""

F = Element(**{'mulliken_jaffe': 3.9100000000000001, 'single_bond_radius': array(64.0) * pm, 'pauling': 3.98, 'molar_volume': array(11.199999999999999) * cm**3/mol, 'atomization': array(79000.0) * J/mol, 'sanderson': 4.0, 'atomic_weight': 18.998403199999998, 'critical_temperature': array(144.0) * Kelvin, 'triple_bond_radius': array(53.0) * pm, 'melting_point': array(53.530000000000001) * Kelvin, 'allen': 4.1929999999999996, 'orbital_radii': (array(0.13229431224999999) * angstrom, array(0.082022473594999992) * angstrom), 'thermal_conductivity': array(0.027699999999999999) * Watt/(m*Kelvin), 'ionization_energies': [array(1681000.0) * J/mol, array(3374200.0) * J/mol, array(6050400.0) * J/mol, array(8407700.0) * J/mol, array(11022700.0) * J/mol, array(15164100.0) * J/mol, array(17868000.0) * J/mol, array(92038100.0) * J/mol, array(106434300.0) * J/mol], 'vaporization': array(3270.0) * J/mol, 'atomic_number': 9, 'symbol': 'F', 'covalent_radius': array(57.0) * pm, 'fusion': array(260.0) * J/mol, 'pettifor': 4.0, 'van_der_waals_radius': array(147.0) * pm, 'electron_affinity': array(328000.0) * J/mol, 'name': 'Fluorine', 'boiling_point': array(85.030000000000001) * Kelvin, 'double_bond_radius': array(59.0) * pm, 'allred_rochow': 4.0999999999999996, 'atomic_radius': array(50.0) * pm})
"""
Fluorine elemental data.

  - mulliken_jaffe: 3.91
  - single_bond_radius: 64.0 pm
  - pauling: 3.98
  - molar_volume: 11.2 cm**3/mol
  - atomization: 79000.0 J/mol
  - sanderson: 4.0
  - atomic_weight: 18.9984032
  - critical_temperature: 144.0 K
  - triple_bond_radius: 53.0 pm
  - melting_point: 53.53 K
  - allen: 4.193
  - orbital_radii: (array(0.13229431224999999) * angstrom, array(0.082022473594999992) * angstrom)
  - thermal_conductivity: 0.0277 W/(m*K)
  - ionization_energies: [array(1681000.0) * J/mol, array(3374200.0) * J/mol, array(6050400.0) * J/mol, array(8407700.0) * J/mol, array(11022700.0) * J/mol, array(15164100.0) * J/mol, array(17868000.0) * J/mol, array(92038100.0) * J/mol, array(106434300.0) * J/mol]
  - vaporization: 3270.0 J/mol
  - atomic_number: 9
  - symbol: F
  - covalent_radius: 57.0 pm
  - fusion: 260.0 J/mol
  - pettifor: 4.0
  - van_der_waals_radius: 147.0 pm
  - electron_affinity: 328000.0 J/mol
  - name: Fluorine
  - boiling_point: 85.03 K
  - double_bond_radius: 59.0 pm
  - allred_rochow: 4.1
  - atomic_radius: 50.0 pm
"""

Ne = Element(**{'mulliken_jaffe': 3.98, 'single_bond_radius': array(67.0) * pm, 'molar_volume': array(13.23) * cm**3/mol, 'atomization': array(0.0) * J/mol, 'sound_velocity': array(936.0) * m/s, 'sanderson': 4.5, 'atomic_weight': 20.1797, 'critical_temperature': array(44.399999999999999) * Kelvin, 'melting_point': array(24.559999999999999) * Kelvin, 'allen': 4.7869999999999999, 'orbital_radii': (array(0.11641899477999999) * angstrom, array(0.074084814860000001) * angstrom), 'thermal_conductivity': array(0.049099999999999998) * Watt/(m*Kelvin), 'ionization_energies': [array(2080699.9999999998) * J/mol, array(3952300.0) * J/mol, array(6122000.0) * J/mol, array(9371000.0) * J/mol, array(12177000.0) * J/mol, array(15238000.0) * J/mol, array(19999000.0) * J/mol, array(23069500.0) * J/mol, array(115379500.0) * J/mol, array(131432000.0) * J/mol], 'vaporization': array(1750.0) * J/mol, 'atomic_number': 10, 'symbol': 'Ne', 'covalent_radius': array(58.0) * pm, 'fusion': array(340.0) * J/mol, 'van_der_waals_radius': array(154.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Neon', 'boiling_point': array(27.07) * Kelvin, 'double_bond_radius': array(96.0) * pm, 'allred_rochow': 4.8399999999999999})
"""
Neon elemental data.

  - mulliken_jaffe: 3.98
  - single_bond_radius: 67.0 pm
  - molar_volume: 13.23 cm**3/mol
  - atomization: 0.0 J/mol
  - sound_velocity: 936.0 m/s
  - sanderson: 4.5
  - atomic_weight: 20.1797
  - critical_temperature: 44.4 K
  - melting_point: 24.56 K
  - allen: 4.787
  - orbital_radii: (array(0.11641899477999999) * angstrom, array(0.074084814860000001) * angstrom)
  - thermal_conductivity: 0.0491 W/(m*K)
  - ionization_energies: [array(2080699.9999999998) * J/mol, array(3952300.0) * J/mol, array(6122000.0) * J/mol, array(9371000.0) * J/mol, array(12177000.0) * J/mol, array(15238000.0) * J/mol, array(19999000.0) * J/mol, array(23069500.0) * J/mol, array(115379500.0) * J/mol, array(131432000.0) * J/mol]
  - vaporization: 1750.0 J/mol
  - atomic_number: 10
  - symbol: Ne
  - covalent_radius: 58.0 pm
  - fusion: 340.0 J/mol
  - van_der_waals_radius: 154.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Neon
  - boiling_point: 27.07 K
  - double_bond_radius: 96.0 pm
  - allred_rochow: 4.84
"""

Na = Element(**{'mulliken_jaffe': 0.91000000000000003, 'single_bond_radius': array(155.0) * pm, 'pauling': 0.93000000000000005, 'molar_volume': array(23.780000000000001) * cm**3/mol, 'bulk_modulus': array(6.2999999999999998) * GPa, 'sound_velocity': array(3200.0) * m/s, 'sanderson': 0.56000000000000005, 'atomic_weight': 22.989769280000001, 'critical_temperature': array(2573.0) * Kelvin, 'melting_point': array(370.87) * Kelvin, 'allen': 0.86899999999999999, 'orbital_radii': (array(0.58209497389999998) * angstrom, array(0.82022473595000001) * angstrom), 'thermal_conductivity': array(140.0) * Watt/(m*Kelvin), 'ionization_energies': [array(495800.0) * J/mol, array(4562000.0) * J/mol, array(6910300.0) * J/mol, array(9543000.0) * J/mol, array(13354000.0) * J/mol, array(16613000.0) * J/mol, array(20117000.0) * J/mol, array(25496000.0) * J/mol, array(28932000.0) * J/mol, array(141362000.0) * J/mol, array(159075000.0) * J/mol], 'vaporization': array(97700.0) * J/mol, 'atomic_number': 11, 'rigidity_modulus': array(3.2999999999999998) * GPa, 'symbol': 'Na', 'covalent_radius': array(166.0) * pm, 'fusion': array(2600.0) * J/mol, 'pettifor': 0.40000000000000002, 'atomization': array(107000.0) * J/mol, 'van_der_waals_radius': array(227.0) * pm, 'electron_affinity': array(52800.0) * J/mol, 'name': 'Sodium', 'boiling_point': array(1156.0) * Kelvin, 'density': array(0.96799999999999997) * g*cm**3, 'double_bond_radius': array(160.0) * pm, 'allred_rochow': 1.01, 'young_modulus': array(10.0) * GPa, 'thermal_expansion': array(7.0999999999999991e-05) * 1/Kelvin, 'atomic_radius': array(180.0) * pm})
"""
Sodium elemental data.

  - mulliken_jaffe: 0.91
  - single_bond_radius: 155.0 pm
  - pauling: 0.93
  - molar_volume: 23.78 cm**3/mol
  - atomization: 107000.0 J/mol
  - sound_velocity: 3200.0 m/s
  - sanderson: 0.56
  - atomic_weight: 22.98976928
  - critical_temperature: 2573.0 K
  - melting_point: 370.87 K
  - allen: 0.869
  - orbital_radii: (array(0.58209497389999998) * angstrom, array(0.82022473595000001) * angstrom)
  - thermal_conductivity: 140.0 W/(m*K)
  - ionization_energies: [array(495800.0) * J/mol, array(4562000.0) * J/mol, array(6910300.0) * J/mol, array(9543000.0) * J/mol, array(13354000.0) * J/mol, array(16613000.0) * J/mol, array(20117000.0) * J/mol, array(25496000.0) * J/mol, array(28932000.0) * J/mol, array(141362000.0) * J/mol, array(159075000.0) * J/mol]
  - vaporization: 97700.0 J/mol
  - atomic_number: 11
  - rigidity_modulus: 3.3 GPa
  - symbol: Na
  - covalent_radius: 166.0 pm
  - fusion: 2600.0 J/mol
  - pettifor: 0.4
  - bulk_modulus: 6.3 GPa
  - van_der_waals_radius: 227.0 pm
  - electron_affinity: 52800.0 J/mol
  - name: Sodium
  - boiling_point: 1156.0 K
  - density: 0.968 g*cm**3
  - double_bond_radius: 160.0 pm
  - allred_rochow: 1.01
  - young_modulus: 10.0 GPa
  - thermal_expansion: 7.1e-05 1/K
  - atomic_radius: 180.0 pm
"""

Mg = Element(**{'mulliken_jaffe': 1.3700000000000001, 'single_bond_radius': array(139.0) * pm, 'pauling': 1.3100000000000001, 'molar_volume': array(14.0) * cm**3/mol, 'bulk_modulus': array(45.0) * GPa, 'sound_velocity': array(4602.0) * m/s, 'sanderson': 1.3200000000000001, 'atomic_weight': 24.305, 'triple_bond_radius': array(127.0) * pm, 'melting_point': array(923.0) * Kelvin, 'allen': 1.2929999999999999, 'orbital_radii': (array(0.4762595241) * angstrom, array(0.59797029136999991) * angstrom), 'thermal_conductivity': array(160.0) * Watt/(m*Kelvin), 'ionization_energies': [array(737700.0) * J/mol, array(1450700.0) * J/mol, array(7732700.0) * J/mol, array(10542500.0) * J/mol, array(13630000.0) * J/mol, array(18020000.0) * J/mol, array(21711000.0) * J/mol, array(25661000.0) * J/mol, array(31653000.0) * J/mol, array(35458000.0) * J/mol, array(169988000.0) * J/mol, array(189367700.0) * J/mol], 'vaporization': array(128000.0) * J/mol, 'atomic_number': 12, 'rigidity_modulus': array(17.0) * GPa, 'symbol': 'Mg', 'covalent_radius': array(141.0) * pm, 'fusion': array(8700.0) * J/mol, 'pettifor': 1.28, 'atomization': array(146000.0) * J/mol, 'poisson_ratio': array(0.28999999999999998) * dimensionless, 'van_der_waals_radius': array(173.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Magnesium', 'boiling_point': array(1363.0) * Kelvin, 'density': array(1.738) * g*cm**3, 'double_bond_radius': array(132.0) * pm, 'allred_rochow': 1.23, 'young_modulus': array(45.0) * GPa, 'thermal_expansion': array(8.1999999999999994e-06) * 1/Kelvin, 'atomic_radius': array(150.0) * pm})
"""
Magnesium elemental data.

  - mulliken_jaffe: 1.37
  - single_bond_radius: 139.0 pm
  - pauling: 1.31
  - molar_volume: 14.0 cm**3/mol
  - atomization: 146000.0 J/mol
  - sound_velocity: 4602.0 m/s
  - sanderson: 1.32
  - atomic_weight: 24.305
  - triple_bond_radius: 127.0 pm
  - melting_point: 923.0 K
  - allen: 1.293
  - orbital_radii: (array(0.4762595241) * angstrom, array(0.59797029136999991) * angstrom)
  - thermal_conductivity: 160.0 W/(m*K)
  - ionization_energies: [array(737700.0) * J/mol, array(1450700.0) * J/mol, array(7732700.0) * J/mol, array(10542500.0) * J/mol, array(13630000.0) * J/mol, array(18020000.0) * J/mol, array(21711000.0) * J/mol, array(25661000.0) * J/mol, array(31653000.0) * J/mol, array(35458000.0) * J/mol, array(169988000.0) * J/mol, array(189367700.0) * J/mol]
  - vaporization: 128000.0 J/mol
  - atomic_number: 12
  - rigidity_modulus: 17.0 GPa
  - symbol: Mg
  - covalent_radius: 141.0 pm
  - fusion: 8700.0 J/mol
  - pettifor: 1.28
  - bulk_modulus: 45.0 GPa
  - poisson_ratio: 0.29 dimensionless
  - van_der_waals_radius: 173.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Magnesium
  - boiling_point: 1363.0 K
  - density: 1.738 g*cm**3
  - double_bond_radius: 132.0 pm
  - allred_rochow: 1.23
  - young_modulus: 45.0 GPa
  - thermal_expansion: 8.2e-06 1/K
  - atomic_radius: 150.0 pm
"""

Al = Element(**{'mulliken_jaffe': 1.8300000000000001, 'single_bond_radius': array(126.0) * pm, 'pauling': 1.6100000000000001, 'molar_volume': array(10.0) * cm**3/mol, 'bulk_modulus': array(76.0) * GPa, 'sound_velocity': array(5100.0) * m/s, 'sanderson': 1.71, 'atomic_weight': 26.9815386, 'triple_bond_radius': array(111.0) * pm, 'melting_point': array(933.47000000000003) * Kelvin, 'allen': 1.613, 'orbital_radii': (array(0.40746648173) * angstrom, array(0.478905410345) * angstrom), 'thermal_conductivity': array(235.0) * Watt/(m*Kelvin), 'ionization_energies': [array(577500.0) * J/mol, array(1816700.0) * J/mol, array(2744800.0) * J/mol, array(11577000.0) * J/mol, array(14842000.0) * J/mol, array(18379000.0) * J/mol, array(23326000.0) * J/mol, array(27465000.0) * J/mol, array(31853000.0) * J/mol, array(38473000.0) * J/mol, array(42646000.0) * J/mol, array(201266000.0) * J/mol, array(222315000.0) * J/mol], 'vaporization': array(293000.0) * J/mol, 'atomic_number': 13, 'rigidity_modulus': array(26.0) * GPa, 'symbol': 'Al', 'covalent_radius': array(121.0) * pm, 'fusion': array(10700.0) * J/mol, 'pettifor': 1.6599999999999999, 'atomization': array(326000.0) * J/mol, 'poisson_ratio': array(0.34999999999999998) * dimensionless, 'electron_affinity': array(42500.0) * J/mol, 'name': 'Aluminium', 'boiling_point': array(2792.0) * Kelvin, 'density': array(2.7000000000000002) * g*cm**3, 'double_bond_radius': array(113.0) * pm, 'allred_rochow': 1.47, 'young_modulus': array(70.0) * GPa, 'thermal_expansion': array(2.3099999999999999e-05) * 1/Kelvin, 'atomic_radius': array(125.0) * pm})
"""
Aluminium elemental data.

  - mulliken_jaffe: 1.83
  - single_bond_radius: 126.0 pm
  - pauling: 1.61
  - molar_volume: 10.0 cm**3/mol
  - atomization: 326000.0 J/mol
  - sound_velocity: 5100.0 m/s
  - sanderson: 1.71
  - atomic_weight: 26.9815386
  - triple_bond_radius: 111.0 pm
  - melting_point: 933.47 K
  - allen: 1.613
  - orbital_radii: (array(0.40746648173) * angstrom, array(0.478905410345) * angstrom)
  - thermal_conductivity: 235.0 W/(m*K)
  - ionization_energies: [array(577500.0) * J/mol, array(1816700.0) * J/mol, array(2744800.0) * J/mol, array(11577000.0) * J/mol, array(14842000.0) * J/mol, array(18379000.0) * J/mol, array(23326000.0) * J/mol, array(27465000.0) * J/mol, array(31853000.0) * J/mol, array(38473000.0) * J/mol, array(42646000.0) * J/mol, array(201266000.0) * J/mol, array(222315000.0) * J/mol]
  - vaporization: 293000.0 J/mol
  - atomic_number: 13
  - rigidity_modulus: 26.0 GPa
  - symbol: Al
  - covalent_radius: 121.0 pm
  - fusion: 10700.0 J/mol
  - pettifor: 1.66
  - bulk_modulus: 76.0 GPa
  - poisson_ratio: 0.35 dimensionless
  - electron_affinity: 42500.0 J/mol
  - name: Aluminium
  - boiling_point: 2792.0 K
  - density: 2.7 g*cm**3
  - double_bond_radius: 113.0 pm
  - allred_rochow: 1.47
  - young_modulus: 70.0 GPa
  - thermal_expansion: 2.31e-05 1/K
  - atomic_radius: 125.0 pm
"""

Si = Element(**{'mulliken_jaffe': 2.2799999999999998, 'single_bond_radius': array(116.0) * pm, 'pauling': 1.8999999999999999, 'molar_volume': array(12.06) * cm**3/mol, 'bulk_modulus': array(100.0) * GPa, 'sound_velocity': array(2200.0) * m/s, 'sanderson': 2.1400000000000001, 'atomic_weight': 28.0855, 'triple_bond_radius': array(102.0) * pm, 'melting_point': array(1687.0) * Kelvin, 'allen': 1.9159999999999999, 'orbital_radii': (array(0.35984052931999999) * angstrom, array(0.39159116425999996) * angstrom), 'thermal_conductivity': array(150.0) * Watt/(m*Kelvin), 'ionization_energies': [array(786500.0) * J/mol, array(1577100.0) * J/mol, array(3231600.0) * J/mol, array(4355500.0) * J/mol, array(16091000.0) * J/mol, array(19805000.0) * J/mol, array(23780000.0) * J/mol, array(29287000.0) * J/mol, array(33878000.0) * J/mol, array(38726000.0) * J/mol, array(45962000.0) * J/mol, array(50502000.0) * J/mol, array(235195000.0) * J/mol, array(257922000.0) * J/mol], 'vaporization': array(359000.0) * J/mol, 'atomic_number': 14, 'symbol': 'Si', 'covalent_radius': array(111.0) * pm, 'fusion': array(50200.0) * J/mol, 'pettifor': 1.9199999999999999, 'atomization': array(456000.0) * J/mol, 'van_der_waals_radius': array(210.0) * pm, 'electron_affinity': array(133600.0) * J/mol, 'name': 'Silicon', 'boiling_point': array(3173.0) * Kelvin, 'density': array(2.3300000000000001) * g*cm**3, 'double_bond_radius': array(107.0) * pm, 'allred_rochow': 1.74, 'young_modulus': array(47.0) * GPa, 'thermal_expansion': array(2.6000000000000001e-06) * 1/Kelvin, 'atomic_radius': array(110.0) * pm})
"""
Silicon elemental data.

  - mulliken_jaffe: 2.28
  - single_bond_radius: 116.0 pm
  - pauling: 1.9
  - molar_volume: 12.06 cm**3/mol
  - atomization: 456000.0 J/mol
  - sound_velocity: 2200.0 m/s
  - sanderson: 2.14
  - atomic_weight: 28.0855
  - triple_bond_radius: 102.0 pm
  - melting_point: 1687.0 K
  - allen: 1.916
  - orbital_radii: (array(0.35984052931999999) * angstrom, array(0.39159116425999996) * angstrom)
  - thermal_conductivity: 150.0 W/(m*K)
  - ionization_energies: [array(786500.0) * J/mol, array(1577100.0) * J/mol, array(3231600.0) * J/mol, array(4355500.0) * J/mol, array(16091000.0) * J/mol, array(19805000.0) * J/mol, array(23780000.0) * J/mol, array(29287000.0) * J/mol, array(33878000.0) * J/mol, array(38726000.0) * J/mol, array(45962000.0) * J/mol, array(50502000.0) * J/mol, array(235195000.0) * J/mol, array(257922000.0) * J/mol]
  - vaporization: 359000.0 J/mol
  - atomic_number: 14
  - symbol: Si
  - covalent_radius: 111.0 pm
  - fusion: 50200.0 J/mol
  - pettifor: 1.92
  - bulk_modulus: 100.0 GPa
  - van_der_waals_radius: 210.0 pm
  - electron_affinity: 133600.0 J/mol
  - name: Silicon
  - boiling_point: 3173.0 K
  - density: 2.33 g*cm**3
  - double_bond_radius: 107.0 pm
  - allred_rochow: 1.74
  - young_modulus: 47.0 GPa
  - thermal_expansion: 2.6e-06 1/K
  - atomic_radius: 110.0 pm
"""

P = Element(**{'mulliken_jaffe': 2.2999999999999998, 'single_bond_radius': array(111.0) * pm, 'pauling': 2.1899999999999999, 'molar_volume': array(17.02) * cm**3/mol, 'bulk_modulus': array(11.0) * GPa, 'sanderson': 2.52, 'atomic_weight': 30.973762000000001, 'critical_temperature': array(994.0) * Kelvin, 'triple_bond_radius': array(94.0) * pm, 'melting_point': array(317.30000000000001) * Kelvin, 'allen': 2.2530000000000001, 'orbital_radii': (array(0.31750634939999994) * angstrom, array(0.33867343935999999) * angstrom), 'thermal_conductivity': array(0.23599999999999999) * Watt/(m*Kelvin), 'ionization_energies': [array(1011800.0) * J/mol, array(1907000.0) * J/mol, array(2914100.0) * J/mol, array(4963600.0) * J/mol, array(6273900.0) * J/mol, array(21267000.0) * J/mol, array(25431000.0) * J/mol, array(29872000.0) * J/mol, array(35905000.0) * J/mol, array(40950000.0) * J/mol, array(46261000.0) * J/mol, array(54110000.0) * J/mol, array(59024000.0) * J/mol, array(271790000.0) * J/mol, array(296194000.0) * J/mol], 'vaporization': array(12400.0) * J/mol, 'atomic_number': 15, 'symbol': 'P', 'covalent_radius': array(107.0) * pm, 'fusion': array(640.0) * J/mol, 'pettifor': 2.1800000000000002, 'atomization': array(315000.0) * J/mol, 'van_der_waals_radius': array(180.0) * pm, 'electron_affinity': array(72000.0) * J/mol, 'name': 'Phosphorus', 'boiling_point': array(550.0) * Kelvin, 'density': array(1.823) * g*cm**3, 'double_bond_radius': array(102.0) * pm, 'allred_rochow': 2.0600000000000001, 'atomic_radius': array(100.0) * pm})
"""
Phosphorus elemental data.

  - mulliken_jaffe: 2.3
  - single_bond_radius: 111.0 pm
  - pauling: 2.19
  - molar_volume: 17.02 cm**3/mol
  - atomization: 315000.0 J/mol
  - sanderson: 2.52
  - atomic_weight: 30.973762
  - critical_temperature: 994.0 K
  - triple_bond_radius: 94.0 pm
  - melting_point: 317.3 K
  - allen: 2.253
  - orbital_radii: (array(0.31750634939999994) * angstrom, array(0.33867343935999999) * angstrom)
  - thermal_conductivity: 0.236 W/(m*K)
  - ionization_energies: [array(1011800.0) * J/mol, array(1907000.0) * J/mol, array(2914100.0) * J/mol, array(4963600.0) * J/mol, array(6273900.0) * J/mol, array(21267000.0) * J/mol, array(25431000.0) * J/mol, array(29872000.0) * J/mol, array(35905000.0) * J/mol, array(40950000.0) * J/mol, array(46261000.0) * J/mol, array(54110000.0) * J/mol, array(59024000.0) * J/mol, array(271790000.0) * J/mol, array(296194000.0) * J/mol]
  - vaporization: 12400.0 J/mol
  - atomic_number: 15
  - symbol: P
  - covalent_radius: 107.0 pm
  - fusion: 640.0 J/mol
  - pettifor: 2.18
  - bulk_modulus: 11.0 GPa
  - van_der_waals_radius: 180.0 pm
  - electron_affinity: 72000.0 J/mol
  - name: Phosphorus
  - boiling_point: 550.0 K
  - density: 1.823 g*cm**3
  - double_bond_radius: 102.0 pm
  - allred_rochow: 2.06
  - atomic_radius: 100.0 pm
"""

S = Element(**{'mulliken_jaffe': 2.6899999999999999, 'single_bond_radius': array(103.0) * pm, 'pauling': 2.5800000000000001, 'molar_volume': array(15.529999999999999) * cm**3/mol, 'bulk_modulus': array(7.7000000000000002) * GPa, 'sanderson': 2.96, 'atomic_weight': 32.064999999999998, 'critical_temperature': array(1314.0) * Kelvin, 'triple_bond_radius': array(95.0) * pm, 'melting_point': array(388.36000000000001) * Kelvin, 'allen': 2.589, 'orbital_radii': (array(0.28575571445999998) * angstrom, array(0.29633925944) * angstrom), 'thermal_conductivity': array(0.20499999999999999) * Watt/(m*Kelvin), 'ionization_energies': [array(999600.0) * J/mol, array(2252000.0) * J/mol, array(3357000.0) * J/mol, array(4556000.0) * J/mol, array(7004300.0) * J/mol, array(8495800.0) * J/mol, array(27107000.0) * J/mol, array(31719000.0) * J/mol, array(36621000.0) * J/mol, array(43177000.0) * J/mol, array(48710000.0) * J/mol, array(54460000.0) * J/mol, array(62930000.0) * J/mol, array(68216000.0) * J/mol, array(311046000.0) * J/mol, array(337137000.0) * J/mol], 'vaporization': array(9800.0) * J/mol, 'atomic_number': 16, 'symbol': 'S', 'covalent_radius': array(105.0) * pm, 'fusion': array(1730.0) * J/mol, 'pettifor': 2.4399999999999999, 'atomization': array(279000.0) * J/mol, 'van_der_waals_radius': array(180.0) * pm, 'electron_affinity': array(200000.0) * J/mol, 'name': 'Sulfur', 'boiling_point': array(717.87) * Kelvin, 'density': array(1.96) * g*cm**3, 'double_bond_radius': array(94.0) * pm, 'allred_rochow': 2.4399999999999999, 'atomic_radius': array(100.0) * pm})
"""
Sulfur elemental data.

  - mulliken_jaffe: 2.69
  - single_bond_radius: 103.0 pm
  - pauling: 2.58
  - molar_volume: 15.53 cm**3/mol
  - atomization: 279000.0 J/mol
  - sanderson: 2.96
  - atomic_weight: 32.065
  - critical_temperature: 1314.0 K
  - triple_bond_radius: 95.0 pm
  - melting_point: 388.36 K
  - allen: 2.589
  - orbital_radii: (array(0.28575571445999998) * angstrom, array(0.29633925944) * angstrom)
  - thermal_conductivity: 0.205 W/(m*K)
  - ionization_energies: [array(999600.0) * J/mol, array(2252000.0) * J/mol, array(3357000.0) * J/mol, array(4556000.0) * J/mol, array(7004300.0) * J/mol, array(8495800.0) * J/mol, array(27107000.0) * J/mol, array(31719000.0) * J/mol, array(36621000.0) * J/mol, array(43177000.0) * J/mol, array(48710000.0) * J/mol, array(54460000.0) * J/mol, array(62930000.0) * J/mol, array(68216000.0) * J/mol, array(311046000.0) * J/mol, array(337137000.0) * J/mol]
  - vaporization: 9800.0 J/mol
  - atomic_number: 16
  - symbol: S
  - covalent_radius: 105.0 pm
  - fusion: 1730.0 J/mol
  - pettifor: 2.44
  - bulk_modulus: 7.7 GPa
  - van_der_waals_radius: 180.0 pm
  - electron_affinity: 200000.0 J/mol
  - name: Sulfur
  - boiling_point: 717.87 K
  - density: 1.96 g*cm**3
  - double_bond_radius: 94.0 pm
  - allred_rochow: 2.44
  - atomic_radius: 100.0 pm
"""

Cl = Element(**{'mulliken_jaffe': 3.1000000000000001, 'single_bond_radius': array(99.0) * pm, 'pauling': 3.1600000000000001, 'molar_volume': array(17.390000000000001) * cm**3/mol, 'bulk_modulus': array(1.1000000000000001) * GPa, 'sound_velocity': array(206.0) * m/s, 'sanderson': 3.48, 'atomic_weight': 35.453000000000003, 'critical_temperature': array(417.0) * Kelvin, 'triple_bond_radius': array(93.0) * pm, 'melting_point': array(171.59999999999999) * Kelvin, 'allen': 2.8690000000000002, 'orbital_radii': (array(0.26458862449999998) * angstrom, array(0.26988039698999999) * angstrom), 'thermal_conductivity': array(0.0088999999999999999) * Watt/(m*Kelvin), 'ionization_energies': [array(1251200.0) * J/mol, array(2298000.0) * J/mol, array(3822000.0) * J/mol, array(5158600.0) * J/mol, array(6542000.0) * J/mol, array(9362000.0) * J/mol, array(11018000.0) * J/mol, array(33604000.0) * J/mol, array(38600000.0) * J/mol, array(43961000.0) * J/mol, array(51068000.0) * J/mol, array(57118000.0) * J/mol, array(63363000.0) * J/mol, array(72341000.0) * J/mol, array(78095000.0) * J/mol, array(352992000.0) * J/mol, array(380758000.0) * J/mol], 'vaporization': array(10200.0) * J/mol, 'atomic_number': 17, 'symbol': 'Cl', 'covalent_radius': array(102.0) * pm, 'fusion': array(3200.0) * J/mol, 'pettifor': 2.7000000000000002, 'atomization': array(122000.0) * J/mol, 'van_der_waals_radius': array(175.0) * pm, 'electron_affinity': array(349000.0) * J/mol, 'name': 'Chlorine', 'boiling_point': array(239.11000000000001) * Kelvin, 'double_bond_radius': array(95.0) * pm, 'allred_rochow': 2.8300000000000001, 'atomic_radius': array(100.0) * pm})
"""
Chlorine elemental data.

  - mulliken_jaffe: 3.1
  - single_bond_radius: 99.0 pm
  - pauling: 3.16
  - molar_volume: 17.39 cm**3/mol
  - atomization: 122000.0 J/mol
  - sound_velocity: 206.0 m/s
  - sanderson: 3.48
  - atomic_weight: 35.453
  - critical_temperature: 417.0 K
  - triple_bond_radius: 93.0 pm
  - melting_point: 171.6 K
  - allen: 2.869
  - orbital_radii: (array(0.26458862449999998) * angstrom, array(0.26988039698999999) * angstrom)
  - thermal_conductivity: 0.0089 W/(m*K)
  - ionization_energies: [array(1251200.0) * J/mol, array(2298000.0) * J/mol, array(3822000.0) * J/mol, array(5158600.0) * J/mol, array(6542000.0) * J/mol, array(9362000.0) * J/mol, array(11018000.0) * J/mol, array(33604000.0) * J/mol, array(38600000.0) * J/mol, array(43961000.0) * J/mol, array(51068000.0) * J/mol, array(57118000.0) * J/mol, array(63363000.0) * J/mol, array(72341000.0) * J/mol, array(78095000.0) * J/mol, array(352992000.0) * J/mol, array(380758000.0) * J/mol]
  - vaporization: 10200.0 J/mol
  - atomic_number: 17
  - symbol: Cl
  - covalent_radius: 102.0 pm
  - fusion: 3200.0 J/mol
  - pettifor: 2.7
  - bulk_modulus: 1.1 GPa
  - van_der_waals_radius: 175.0 pm
  - electron_affinity: 349000.0 J/mol
  - name: Chlorine
  - boiling_point: 239.11 K
  - double_bond_radius: 95.0 pm
  - allred_rochow: 2.83
  - atomic_radius: 100.0 pm
"""

Ar = Element(**{'mulliken_jaffe': 3.1899999999999999, 'single_bond_radius': array(96.0) * pm, 'molar_volume': array(22.559999999999999) * cm**3/mol, 'atomization': array(0.0) * J/mol, 'sound_velocity': array(319.0) * m/s, 'sanderson': 3.3100000000000001, 'atomic_weight': 39.948, 'critical_temperature': array(150.80000000000001) * Kelvin, 'triple_bond_radius': array(96.0) * pm, 'melting_point': array(83.799999999999997) * Kelvin, 'allen': 3.242, 'orbital_radii': (array(0.24342153453999998) * angstrom, array(0.24342153453999998) * angstrom), 'thermal_conductivity': array(0.01772) * Watt/(m*Kelvin), 'ionization_energies': [array(1520600.0) * J/mol, array(2665800.0) * J/mol, array(3931000.0) * J/mol, array(5771000.0) * J/mol, array(7238000.0) * J/mol, array(8781000.0) * J/mol, array(11995000.0) * J/mol, array(13842000.0) * J/mol, array(40760000.0) * J/mol, array(46186000.0) * J/mol, array(52002000.0) * J/mol, array(59653000.0) * J/mol, array(66198000.0) * J/mol, array(72918000.0) * J/mol, array(82472000.0) * J/mol, array(88576000.0) * J/mol, array(397604000.0) * J/mol, array(427065000.0) * J/mol], 'vaporization': array(6500.0) * J/mol, 'atomic_number': 18, 'symbol': 'Ar', 'covalent_radius': array(106.0) * pm, 'fusion': array(1180.0) * J/mol, 'van_der_waals_radius': array(188.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Argon', 'boiling_point': array(87.299999999999997) * Kelvin, 'double_bond_radius': array(107.0) * pm, 'allred_rochow': 3.2000000000000002})
"""
Argon elemental data.

  - mulliken_jaffe: 3.19
  - single_bond_radius: 96.0 pm
  - molar_volume: 22.56 cm**3/mol
  - atomization: 0.0 J/mol
  - sound_velocity: 319.0 m/s
  - sanderson: 3.31
  - atomic_weight: 39.948
  - critical_temperature: 150.8 K
  - triple_bond_radius: 96.0 pm
  - melting_point: 83.8 K
  - allen: 3.242
  - orbital_radii: (array(0.24342153453999998) * angstrom, array(0.24342153453999998) * angstrom)
  - thermal_conductivity: 0.01772 W/(m*K)
  - ionization_energies: [array(1520600.0) * J/mol, array(2665800.0) * J/mol, array(3931000.0) * J/mol, array(5771000.0) * J/mol, array(7238000.0) * J/mol, array(8781000.0) * J/mol, array(11995000.0) * J/mol, array(13842000.0) * J/mol, array(40760000.0) * J/mol, array(46186000.0) * J/mol, array(52002000.0) * J/mol, array(59653000.0) * J/mol, array(66198000.0) * J/mol, array(72918000.0) * J/mol, array(82472000.0) * J/mol, array(88576000.0) * J/mol, array(397604000.0) * J/mol, array(427065000.0) * J/mol]
  - vaporization: 6500.0 J/mol
  - atomic_number: 18
  - symbol: Ar
  - covalent_radius: 106.0 pm
  - fusion: 1180.0 J/mol
  - van_der_waals_radius: 188.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Argon
  - boiling_point: 87.3 K
  - double_bond_radius: 107.0 pm
  - allred_rochow: 3.2
"""

K = Element(**{'mulliken_jaffe': 0.72999999999999998, 'single_bond_radius': array(196.0) * pm, 'pauling': 0.81999999999999995, 'molar_volume': array(45.939999999999998) * cm**3/mol, 'bulk_modulus': array(3.1000000000000001) * GPa, 'sound_velocity': array(2000.0) * m/s, 'sanderson': 0.45000000000000001, 'atomic_weight': 39.098300000000002, 'critical_temperature': array(2223.0) * Kelvin, 'melting_point': array(336.52999999999997) * Kelvin, 'allen': 0.73399999999999999, 'orbital_radii': (array(0.81493296345999999) * angstrom, array(1.1377310853499998) * angstrom, array(0.19579558212999998) * angstrom), 'thermal_conductivity': array(100.0) * Watt/(m*Kelvin), 'ionization_energies': [array(418800.0) * J/mol, array(3052000.0) * J/mol, array(4420000.0) * J/mol, array(5877000.0) * J/mol, array(7975000.0) * J/mol, array(9590000.0) * J/mol, array(11343000.0) * J/mol, array(14944000.0) * J/mol, array(16963700.0) * J/mol, array(48610000.0) * J/mol, array(54490000.0) * J/mol, array(60730000.0) * J/mol, array(68950000.0) * J/mol, array(75900000.0) * J/mol, array(83080000.0) * J/mol, array(93400000.0) * J/mol, array(99710000.0) * J/mol, array(444870000.0) * J/mol, array(476061000.0) * J/mol], 'vaporization': array(76900.0) * J/mol, 'atomic_number': 19, 'rigidity_modulus': array(1.3) * GPa, 'symbol': 'K', 'covalent_radius': array(203.0) * pm, 'fusion': array(2330.0) * J/mol, 'pettifor': 0.34999999999999998, 'atomization': array(89000.0) * J/mol, 'van_der_waals_radius': array(275.0) * pm, 'electron_affinity': array(48400.0) * J/mol, 'name': 'Potassium', 'boiling_point': array(1032.0) * Kelvin, 'density': array(0.85599999999999998) * g*cm**3, 'double_bond_radius': array(193.0) * pm, 'allred_rochow': 0.91000000000000003, 'atomic_radius': array(220.0) * pm})
"""
Potassium elemental data.

  - mulliken_jaffe: 0.73
  - single_bond_radius: 196.0 pm
  - pauling: 0.82
  - molar_volume: 45.94 cm**3/mol
  - atomization: 89000.0 J/mol
  - sound_velocity: 2000.0 m/s
  - sanderson: 0.45
  - atomic_weight: 39.0983
  - critical_temperature: 2223.0 K
  - melting_point: 336.53 K
  - allen: 0.734
  - orbital_radii: (array(0.81493296345999999) * angstrom, array(1.1377310853499998) * angstrom, array(0.19579558212999998) * angstrom)
  - thermal_conductivity: 100.0 W/(m*K)
  - ionization_energies: [array(418800.0) * J/mol, array(3052000.0) * J/mol, array(4420000.0) * J/mol, array(5877000.0) * J/mol, array(7975000.0) * J/mol, array(9590000.0) * J/mol, array(11343000.0) * J/mol, array(14944000.0) * J/mol, array(16963700.0) * J/mol, array(48610000.0) * J/mol, array(54490000.0) * J/mol, array(60730000.0) * J/mol, array(68950000.0) * J/mol, array(75900000.0) * J/mol, array(83080000.0) * J/mol, array(93400000.0) * J/mol, array(99710000.0) * J/mol, array(444870000.0) * J/mol, array(476061000.0) * J/mol]
  - vaporization: 76900.0 J/mol
  - atomic_number: 19
  - rigidity_modulus: 1.3 GPa
  - symbol: K
  - covalent_radius: 203.0 pm
  - fusion: 2330.0 J/mol
  - pettifor: 0.35
  - bulk_modulus: 3.1 GPa
  - van_der_waals_radius: 275.0 pm
  - electron_affinity: 48400.0 J/mol
  - name: Potassium
  - boiling_point: 1032.0 K
  - density: 0.856 g*cm**3
  - double_bond_radius: 193.0 pm
  - allred_rochow: 0.91
  - atomic_radius: 220.0 pm
"""

Ca = Element(**{'mulliken_jaffe': 1.0800000000000001, 'single_bond_radius': array(171.0) * pm, 'pauling': 1.0, 'molar_volume': array(26.199999999999999) * cm**3/mol, 'bulk_modulus': array(17.0) * GPa, 'sound_velocity': array(3810.0) * m/s, 'sanderson': 0.94999999999999996, 'atomic_weight': 40.078000000000003, 'triple_bond_radius': array(133.0) * pm, 'melting_point': array(1115.0) * Kelvin, 'allen': 1.034, 'orbital_radii': (array(0.69851396867999993) * angstrom, array(0.88901777831999995) * angstrom, array(0.17992026466) * angstrom), 'thermal_conductivity': array(200.0) * Watt/(m*Kelvin), 'ionization_energies': [array(589800.0) * J/mol, array(1145400.0) * J/mol, array(4912400.0) * J/mol, array(6491000.0) * J/mol, array(8153000.0) * J/mol, array(10496000.0) * J/mol, array(12270000.0) * J/mol, array(14206000.0) * J/mol, array(18191000.0) * J/mol, array(20385000.0) * J/mol, array(57110000.0) * J/mol, array(63410000.0) * J/mol, array(70110000.0) * J/mol, array(78890000.0) * J/mol, array(86310000.0) * J/mol, array(94000000.0) * J/mol, array(104900000.0) * J/mol, array(111710000.0) * J/mol, array(494850000.0) * J/mol, array(527760000.0) * J/mol], 'vaporization': array(155000.0) * J/mol, 'atomic_number': 20, 'rigidity_modulus': array(7.4000000000000004) * GPa, 'symbol': 'Ca', 'covalent_radius': array(176.0) * pm, 'fusion': array(8540.0) * J/mol, 'pettifor': 0.59999999999999998, 'atomization': array(178000.0) * J/mol, 'poisson_ratio': array(0.31) * dimensionless, 'electron_affinity': array(2370.0) * J/mol, 'name': 'Calcium', 'boiling_point': array(1757.0) * Kelvin, 'density': array(1.55) * g*cm**3, 'double_bond_radius': array(147.0) * pm, 'allred_rochow': 1.04, 'young_modulus': array(20.0) * GPa, 'thermal_expansion': array(2.23e-05) * 1/Kelvin, 'atomic_radius': array(180.0) * pm})
"""
Calcium elemental data.

  - mulliken_jaffe: 1.08
  - single_bond_radius: 171.0 pm
  - pauling: 1.0
  - molar_volume: 26.2 cm**3/mol
  - atomization: 178000.0 J/mol
  - sound_velocity: 3810.0 m/s
  - sanderson: 0.95
  - atomic_weight: 40.078
  - triple_bond_radius: 133.0 pm
  - melting_point: 1115.0 K
  - allen: 1.034
  - orbital_radii: (array(0.69851396867999993) * angstrom, array(0.88901777831999995) * angstrom, array(0.17992026466) * angstrom)
  - thermal_conductivity: 200.0 W/(m*K)
  - ionization_energies: [array(589800.0) * J/mol, array(1145400.0) * J/mol, array(4912400.0) * J/mol, array(6491000.0) * J/mol, array(8153000.0) * J/mol, array(10496000.0) * J/mol, array(12270000.0) * J/mol, array(14206000.0) * J/mol, array(18191000.0) * J/mol, array(20385000.0) * J/mol, array(57110000.0) * J/mol, array(63410000.0) * J/mol, array(70110000.0) * J/mol, array(78890000.0) * J/mol, array(86310000.0) * J/mol, array(94000000.0) * J/mol, array(104900000.0) * J/mol, array(111710000.0) * J/mol, array(494850000.0) * J/mol, array(527760000.0) * J/mol]
  - vaporization: 155000.0 J/mol
  - atomic_number: 20
  - rigidity_modulus: 7.4 GPa
  - symbol: Ca
  - covalent_radius: 176.0 pm
  - fusion: 8540.0 J/mol
  - pettifor: 0.6
  - bulk_modulus: 17.0 GPa
  - poisson_ratio: 0.31 dimensionless
  - electron_affinity: 2370.0 J/mol
  - name: Calcium
  - boiling_point: 1757.0 K
  - density: 1.55 g*cm**3
  - double_bond_radius: 147.0 pm
  - allred_rochow: 1.04
  - young_modulus: 20.0 GPa
  - thermal_expansion: 2.23e-05 1/K
  - atomic_radius: 180.0 pm
"""

Sc = Element(**{'single_bond_radius': array(148.0) * pm, 'pauling': 1.3600000000000001, 'molar_volume': array(15.0) * cm**3/mol, 'bulk_modulus': array(57.0) * GPa, 'sanderson': 1.02, 'atomic_weight': 44.955911999999998, 'triple_bond_radius': array(114.0) * pm, 'melting_point': array(1814.0) * Kelvin, 'orbital_radii': (array(0.64559624377999991) * angstrom, array(0.80964119096999998) * angstrom, array(0.16404494718999998) * angstrom), 'thermal_conductivity': array(16.0) * Watt/(m*Kelvin), 'ionization_energies': [array(633100.0) * J/mol, array(1235000.0) * J/mol, array(2388600.0) * J/mol, array(7090600.0) * J/mol, array(8843000.0) * J/mol, array(10679000.0) * J/mol, array(13310000.0) * J/mol, array(15250000.0) * J/mol, array(17370000.0) * J/mol, array(21726000.0) * J/mol, array(24102000.0) * J/mol, array(66320000.0) * J/mol, array(73010000.0) * J/mol, array(80160000.0) * J/mol, array(89490000.0) * J/mol, array(97400000.0) * J/mol, array(105600000.0) * J/mol, array(117000000.0) * J/mol, array(124270000.0) * J/mol, array(547530000.0) * J/mol, array(582163000.0) * J/mol], 'vaporization': array(318000.0) * J/mol, 'atomic_number': 21, 'rigidity_modulus': array(29.0) * GPa, 'symbol': 'Sc', 'covalent_radius': array(170.0) * pm, 'fusion': array(16000.0) * J/mol, 'pettifor': 0.73999999999999999, 'atomization': array(378000.0) * J/mol, 'poisson_ratio': array(0.28000000000000003) * dimensionless, 'electron_affinity': array(18100.0) * J/mol, 'name': 'Scandium', 'boiling_point': array(3103.0) * Kelvin, 'density': array(2.9849999999999999) * g*cm**3, 'double_bond_radius': array(116.0) * pm, 'allred_rochow': 1.2, 'young_modulus': array(74.0) * GPa, 'thermal_expansion': array(1.0199999999999999e-05) * 1/Kelvin, 'atomic_radius': array(160.0) * pm})
"""
Scandium elemental data.

  - single_bond_radius: 148.0 pm
  - pauling: 1.36
  - molar_volume: 15.0 cm**3/mol
  - atomization: 378000.0 J/mol
  - sanderson: 1.02
  - atomic_weight: 44.955912
  - triple_bond_radius: 114.0 pm
  - melting_point: 1814.0 K
  - orbital_radii: (array(0.64559624377999991) * angstrom, array(0.80964119096999998) * angstrom, array(0.16404494718999998) * angstrom)
  - thermal_conductivity: 16.0 W/(m*K)
  - ionization_energies: [array(633100.0) * J/mol, array(1235000.0) * J/mol, array(2388600.0) * J/mol, array(7090600.0) * J/mol, array(8843000.0) * J/mol, array(10679000.0) * J/mol, array(13310000.0) * J/mol, array(15250000.0) * J/mol, array(17370000.0) * J/mol, array(21726000.0) * J/mol, array(24102000.0) * J/mol, array(66320000.0) * J/mol, array(73010000.0) * J/mol, array(80160000.0) * J/mol, array(89490000.0) * J/mol, array(97400000.0) * J/mol, array(105600000.0) * J/mol, array(117000000.0) * J/mol, array(124270000.0) * J/mol, array(547530000.0) * J/mol, array(582163000.0) * J/mol]
  - vaporization: 318000.0 J/mol
  - atomic_number: 21
  - rigidity_modulus: 29.0 GPa
  - symbol: Sc
  - covalent_radius: 170.0 pm
  - fusion: 16000.0 J/mol
  - pettifor: 0.74
  - bulk_modulus: 57.0 GPa
  - poisson_ratio: 0.28 dimensionless
  - electron_affinity: 18100.0 J/mol
  - name: Scandium
  - boiling_point: 3103.0 K
  - density: 2.985 g*cm**3
  - double_bond_radius: 116.0 pm
  - allred_rochow: 1.2
  - young_modulus: 74.0 GPa
  - thermal_expansion: 1.02e-05 1/K
  - atomic_radius: 160.0 pm
"""

Ti = Element(**{'single_bond_radius': array(136.0) * pm, 'pauling': 1.54, 'molar_volume': array(10.640000000000001) * cm**3/mol, 'bulk_modulus': array(110.0) * GPa, 'sound_velocity': array(4140.0) * m/s, 'sanderson': 1.0900000000000001, 'atomic_weight': 47.866999999999997, 'triple_bond_radius': array(108.0) * pm, 'melting_point': array(1941.0) * Kelvin, 'orbital_radii': (array(0.60855383634999993) * angstrom, array(0.75672346606999996) * angstrom, array(0.14816962972) * angstrom), 'thermal_conductivity': array(22.0) * Watt/(m*Kelvin), 'ionization_energies': [array(658800.0) * J/mol, array(1309800.0) * J/mol, array(2652500.0) * J/mol, array(4174600.0000000005) * J/mol, array(9581000.0) * J/mol, array(11533000.0) * J/mol, array(13590000.0) * J/mol, array(16440000.0) * J/mol, array(18530000.0) * J/mol, array(20833000.0) * J/mol, array(25575000.0) * J/mol, array(28125000.0) * J/mol, array(76015000.0) * J/mol, array(83280000.0) * J/mol, array(90880000.0) * J/mol, array(100700000.0) * J/mol, array(109100000.0) * J/mol, array(117800000.0) * J/mol, array(129900000.0) * J/mol, array(137530000.0) * J/mol, array(602930000.0) * J/mol], 'vaporization': array(425000.0) * J/mol, 'atomic_number': 22, 'rigidity_modulus': array(44.0) * GPa, 'symbol': 'Ti', 'covalent_radius': array(160.0) * pm, 'fusion': array(18700.0) * J/mol, 'pettifor': 0.79000000000000004, 'atomization': array(471000.0) * J/mol, 'poisson_ratio': array(0.32000000000000001) * dimensionless, 'electron_affinity': array(7600.0) * J/mol, 'name': 'Titanium', 'boiling_point': array(3560.0) * Kelvin, 'density': array(4.5069999999999997) * g*cm**3, 'double_bond_radius': array(117.0) * pm, 'allred_rochow': 1.3200000000000001, 'young_modulus': array(116.0) * GPa, 'thermal_expansion': array(8.599999999999999e-06) * 1/Kelvin, 'atomic_radius': array(140.0) * pm})
"""
Titanium elemental data.

  - single_bond_radius: 136.0 pm
  - pauling: 1.54
  - molar_volume: 10.64 cm**3/mol
  - atomization: 471000.0 J/mol
  - sound_velocity: 4140.0 m/s
  - sanderson: 1.09
  - atomic_weight: 47.867
  - triple_bond_radius: 108.0 pm
  - melting_point: 1941.0 K
  - orbital_radii: (array(0.60855383634999993) * angstrom, array(0.75672346606999996) * angstrom, array(0.14816962972) * angstrom)
  - thermal_conductivity: 22.0 W/(m*K)
  - ionization_energies: [array(658800.0) * J/mol, array(1309800.0) * J/mol, array(2652500.0) * J/mol, array(4174600.0000000005) * J/mol, array(9581000.0) * J/mol, array(11533000.0) * J/mol, array(13590000.0) * J/mol, array(16440000.0) * J/mol, array(18530000.0) * J/mol, array(20833000.0) * J/mol, array(25575000.0) * J/mol, array(28125000.0) * J/mol, array(76015000.0) * J/mol, array(83280000.0) * J/mol, array(90880000.0) * J/mol, array(100700000.0) * J/mol, array(109100000.0) * J/mol, array(117800000.0) * J/mol, array(129900000.0) * J/mol, array(137530000.0) * J/mol, array(602930000.0) * J/mol]
  - vaporization: 425000.0 J/mol
  - atomic_number: 22
  - rigidity_modulus: 44.0 GPa
  - symbol: Ti
  - covalent_radius: 160.0 pm
  - fusion: 18700.0 J/mol
  - pettifor: 0.79
  - bulk_modulus: 110.0 GPa
  - poisson_ratio: 0.32 dimensionless
  - electron_affinity: 7600.0 J/mol
  - name: Titanium
  - boiling_point: 3560.0 K
  - density: 4.507 g*cm**3
  - double_bond_radius: 117.0 pm
  - allred_rochow: 1.32
  - young_modulus: 116.0 GPa
  - thermal_expansion: 8.6e-06 1/K
  - atomic_radius: 140.0 pm
"""

V = Element(**{'single_bond_radius': array(134.0) * pm, 'pauling': 1.6299999999999999, 'molar_volume': array(8.3200000000000003) * cm**3/mol, 'bulk_modulus': array(160.0) * GPa, 'sound_velocity': array(4560.0) * m/s, 'sanderson': 1.3899999999999999, 'atomic_weight': 50.941499999999998, 'triple_bond_radius': array(106.0) * pm, 'melting_point': array(2183.0) * Kelvin, 'orbital_radii': (array(0.57680320140999997) * angstrom, array(0.70909751365999996) * angstrom, array(0.13758608474) * angstrom), 'thermal_conductivity': array(31.0) * Watt/(m*Kelvin), 'ionization_energies': [array(650900.0) * J/mol, array(1414000.0) * J/mol, array(2830000.0) * J/mol, array(4507000.0) * J/mol, array(6298700.0) * J/mol, array(12363000.0) * J/mol, array(14530000.0) * J/mol, array(16730000.0) * J/mol, array(19860000.0) * J/mol, array(22240000.0) * J/mol, array(24670000.0) * J/mol, array(29730000.0) * J/mol, array(32446000.0) * J/mol, array(86450000.0) * J/mol, array(94170000.0) * J/mol, array(102300000.0) * J/mol, array(112700000.0) * J/mol, array(121600000.0) * J/mol, array(130700000.0) * J/mol, array(143400000.0) * J/mol, array(151440000.0) * J/mol], 'vaporization': array(453000.0) * J/mol, 'atomic_number': 23, 'rigidity_modulus': array(47.0) * GPa, 'symbol': 'V', 'covalent_radius': array(153.0) * pm, 'fusion': array(22800.0) * J/mol, 'pettifor': 0.83999999999999997, 'atomization': array(515000.0) * J/mol, 'poisson_ratio': array(0.37) * dimensionless, 'electron_affinity': array(50600.0) * J/mol, 'name': 'Vanadium', 'boiling_point': array(3680.0) * Kelvin, 'density': array(6.1100000000000003) * g*cm**3, 'double_bond_radius': array(112.0) * pm, 'allred_rochow': 1.45, 'young_modulus': array(128.0) * GPa, 'thermal_expansion': array(8.3999999999999992e-06) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Vanadium elemental data.

  - single_bond_radius: 134.0 pm
  - pauling: 1.63
  - molar_volume: 8.32 cm**3/mol
  - atomization: 515000.0 J/mol
  - sound_velocity: 4560.0 m/s
  - sanderson: 1.39
  - atomic_weight: 50.9415
  - triple_bond_radius: 106.0 pm
  - melting_point: 2183.0 K
  - orbital_radii: (array(0.57680320140999997) * angstrom, array(0.70909751365999996) * angstrom, array(0.13758608474) * angstrom)
  - thermal_conductivity: 31.0 W/(m*K)
  - ionization_energies: [array(650900.0) * J/mol, array(1414000.0) * J/mol, array(2830000.0) * J/mol, array(4507000.0) * J/mol, array(6298700.0) * J/mol, array(12363000.0) * J/mol, array(14530000.0) * J/mol, array(16730000.0) * J/mol, array(19860000.0) * J/mol, array(22240000.0) * J/mol, array(24670000.0) * J/mol, array(29730000.0) * J/mol, array(32446000.0) * J/mol, array(86450000.0) * J/mol, array(94170000.0) * J/mol, array(102300000.0) * J/mol, array(112700000.0) * J/mol, array(121600000.0) * J/mol, array(130700000.0) * J/mol, array(143400000.0) * J/mol, array(151440000.0) * J/mol]
  - vaporization: 453000.0 J/mol
  - atomic_number: 23
  - rigidity_modulus: 47.0 GPa
  - symbol: V
  - covalent_radius: 153.0 pm
  - fusion: 22800.0 J/mol
  - pettifor: 0.84
  - bulk_modulus: 160.0 GPa
  - poisson_ratio: 0.37 dimensionless
  - electron_affinity: 50600.0 J/mol
  - name: Vanadium
  - boiling_point: 3680.0 K
  - density: 6.11 g*cm**3
  - double_bond_radius: 112.0 pm
  - allred_rochow: 1.45
  - young_modulus: 128.0 GPa
  - thermal_expansion: 8.4e-06 1/K
  - atomic_radius: 135.0 pm
"""

Cr = Element(**{'single_bond_radius': array(122.0) * pm, 'pauling': 1.6599999999999999, 'molar_volume': array(7.2300000000000004) * cm**3/mol, 'bulk_modulus': array(160.0) * GPa, 'sound_velocity': array(5940.0) * m/s, 'sanderson': 1.6599999999999999, 'atomic_weight': 51.996099999999998, 'triple_bond_radius': array(103.0) * pm, 'melting_point': array(2180.0) * Kelvin, 'orbital_radii': (array(0.56621965642999994) * angstrom, array(0.72497283112999999) * angstrom, array(0.13229431224999999) * angstrom), 'thermal_conductivity': array(94.0) * Watt/(m*Kelvin), 'ionization_energies': [array(652900.0) * J/mol, array(1590600.0) * J/mol, array(2987000.0) * J/mol, array(4743000.0) * J/mol, array(6702000.0) * J/mol, array(8744900.0) * J/mol, array(15455000.0) * J/mol, array(17820000.0) * J/mol, array(20190000.0) * J/mol, array(23580000.0) * J/mol, array(26130000.0) * J/mol, array(28750000.0) * J/mol, array(34230000.0) * J/mol, array(37066000.0) * J/mol, array(97510000.0) * J/mol, array(105800000.0) * J/mol, array(114300000.0) * J/mol, array(125300000.0) * J/mol, array(134700000.0) * J/mol, array(144300000.0) * J/mol, array(157700000.0) * J/mol], 'vaporization': array(339000.0) * J/mol, 'atomic_number': 24, 'rigidity_modulus': array(115.0) * GPa, 'symbol': 'Cr', 'covalent_radius': array(139.0) * pm, 'fusion': array(20500.0) * J/mol, 'pettifor': 0.89000000000000001, 'atomization': array(397000.0) * J/mol, 'poisson_ratio': array(0.20999999999999999) * dimensionless, 'electron_affinity': array(64300.0) * J/mol, 'name': 'Chromium', 'boiling_point': array(2944.0) * Kelvin, 'density': array(7.1399999999999997) * g*cm**3, 'double_bond_radius': array(111.0) * pm, 'allred_rochow': 1.5600000000000001, 'young_modulus': array(279.0) * GPa, 'thermal_expansion': array(4.9000000000000005e-06) * 1/Kelvin, 'atomic_radius': array(140.0) * pm})
"""
Chromium elemental data.

  - single_bond_radius: 122.0 pm
  - pauling: 1.66
  - molar_volume: 7.23 cm**3/mol
  - atomization: 397000.0 J/mol
  - sound_velocity: 5940.0 m/s
  - sanderson: 1.66
  - atomic_weight: 51.9961
  - triple_bond_radius: 103.0 pm
  - melting_point: 2180.0 K
  - orbital_radii: (array(0.56621965642999994) * angstrom, array(0.72497283112999999) * angstrom, array(0.13229431224999999) * angstrom)
  - thermal_conductivity: 94.0 W/(m*K)
  - ionization_energies: [array(652900.0) * J/mol, array(1590600.0) * J/mol, array(2987000.0) * J/mol, array(4743000.0) * J/mol, array(6702000.0) * J/mol, array(8744900.0) * J/mol, array(15455000.0) * J/mol, array(17820000.0) * J/mol, array(20190000.0) * J/mol, array(23580000.0) * J/mol, array(26130000.0) * J/mol, array(28750000.0) * J/mol, array(34230000.0) * J/mol, array(37066000.0) * J/mol, array(97510000.0) * J/mol, array(105800000.0) * J/mol, array(114300000.0) * J/mol, array(125300000.0) * J/mol, array(134700000.0) * J/mol, array(144300000.0) * J/mol, array(157700000.0) * J/mol]
  - vaporization: 339000.0 J/mol
  - atomic_number: 24
  - rigidity_modulus: 115.0 GPa
  - symbol: Cr
  - covalent_radius: 139.0 pm
  - fusion: 20500.0 J/mol
  - pettifor: 0.89
  - bulk_modulus: 160.0 GPa
  - poisson_ratio: 0.21 dimensionless
  - electron_affinity: 64300.0 J/mol
  - name: Chromium
  - boiling_point: 2944.0 K
  - density: 7.14 g*cm**3
  - double_bond_radius: 111.0 pm
  - allred_rochow: 1.56
  - young_modulus: 279.0 GPa
  - thermal_expansion: 4.9e-06 1/K
  - atomic_radius: 140.0 pm
"""

Mn = Element(**{'single_bond_radius': array(119.0) * pm, 'pauling': 1.55, 'molar_volume': array(7.3499999999999996) * cm**3/mol, 'bulk_modulus': array(120.0) * GPa, 'sound_velocity': array(5150.0) * m/s, 'sanderson': 2.2000000000000002, 'atomic_weight': 54.938045000000002, 'triple_bond_radius': array(103.0) * pm, 'melting_point': array(1519.0) * Kelvin, 'orbital_radii': (array(0.52388547650999995) * angstrom, array(0.65088801626999993) * angstrom, array(0.12171076726999999) * angstrom), 'thermal_conductivity': array(7.7999999999999998) * Watt/(m*Kelvin), 'ionization_energies': [array(717300.0) * J/mol, array(1509000.0) * J/mol, array(3248000.0) * J/mol, array(4940000.0) * J/mol, array(6990000.0) * J/mol, array(9220000.0) * J/mol, array(11500000.0) * J/mol, array(18770000.0) * J/mol, array(21400000.0) * J/mol, array(23960000.0) * J/mol, array(27590000.0) * J/mol, array(30330000.0) * J/mol, array(33150000.0) * J/mol, array(38880000.0) * J/mol, array(41987000.0) * J/mol, array(109480000.0) * J/mol, array(118100000.0) * J/mol, array(127100000.0) * J/mol, array(138600000.0) * J/mol, array(148500000.0) * J/mol, array(158600000.0) * J/mol], 'vaporization': array(220000.0) * J/mol, 'atomic_number': 25, 'symbol': 'Mn', 'covalent_radius': array(139.0) * pm, 'fusion': array(13200.0) * J/mol, 'pettifor': 0.93999999999999995, 'atomization': array(281000.0) * J/mol, 'electron_affinity': array(0.0) * J/mol, 'name': 'Manganese', 'boiling_point': array(2334.0) * Kelvin, 'density': array(7.4699999999999998) * g*cm**3, 'double_bond_radius': array(105.0) * pm, 'allred_rochow': 1.6000000000000001, 'young_modulus': array(198.0) * GPa, 'thermal_expansion': array(2.1699999999999999e-05) * 1/Kelvin, 'atomic_radius': array(140.0) * pm})
"""
Manganese elemental data.

  - single_bond_radius: 119.0 pm
  - pauling: 1.55
  - molar_volume: 7.35 cm**3/mol
  - atomization: 281000.0 J/mol
  - sound_velocity: 5150.0 m/s
  - sanderson: 2.2
  - atomic_weight: 54.938045
  - triple_bond_radius: 103.0 pm
  - melting_point: 1519.0 K
  - orbital_radii: (array(0.52388547650999995) * angstrom, array(0.65088801626999993) * angstrom, array(0.12171076726999999) * angstrom)
  - thermal_conductivity: 7.8 W/(m*K)
  - ionization_energies: [array(717300.0) * J/mol, array(1509000.0) * J/mol, array(3248000.0) * J/mol, array(4940000.0) * J/mol, array(6990000.0) * J/mol, array(9220000.0) * J/mol, array(11500000.0) * J/mol, array(18770000.0) * J/mol, array(21400000.0) * J/mol, array(23960000.0) * J/mol, array(27590000.0) * J/mol, array(30330000.0) * J/mol, array(33150000.0) * J/mol, array(38880000.0) * J/mol, array(41987000.0) * J/mol, array(109480000.0) * J/mol, array(118100000.0) * J/mol, array(127100000.0) * J/mol, array(138600000.0) * J/mol, array(148500000.0) * J/mol, array(158600000.0) * J/mol]
  - vaporization: 220000.0 J/mol
  - atomic_number: 25
  - symbol: Mn
  - covalent_radius: 139.0 pm
  - fusion: 13200.0 J/mol
  - pettifor: 0.94
  - bulk_modulus: 120.0 GPa
  - electron_affinity: 0.0 J/mol
  - name: Manganese
  - boiling_point: 2334.0 K
  - density: 7.47 g*cm**3
  - double_bond_radius: 105.0 pm
  - allred_rochow: 1.6
  - young_modulus: 198.0 GPa
  - thermal_expansion: 2.17e-05 1/K
  - atomic_radius: 140.0 pm
"""

Fe = Element(**{'single_bond_radius': array(116.0) * pm, 'pauling': 1.8300000000000001, 'molar_volume': array(7.0899999999999999) * cm**3/mol, 'bulk_modulus': array(170.0) * GPa, 'sound_velocity': array(4910.0) * m/s, 'sanderson': 2.2000000000000002, 'atomic_weight': 55.844999999999999, 'triple_bond_radius': array(102.0) * pm, 'melting_point': array(1811.0) * Kelvin, 'orbital_radii': (array(0.5027183865499999) * angstrom, array(0.61384560883999995) * angstrom, array(0.11641899477999999) * angstrom), 'thermal_conductivity': array(80.0) * Watt/(m*Kelvin), 'ionization_energies': [array(762500.0) * J/mol, array(1561900.0) * J/mol, array(2957000.0) * J/mol, array(5290000.0) * J/mol, array(7240000.0) * J/mol, array(9560000.0) * J/mol, array(12060000.0) * J/mol, array(14580000.0) * J/mol, array(22540000.0) * J/mol, array(25290000.0) * J/mol, array(28000000.0) * J/mol, array(31920000.0) * J/mol, array(34830000.0) * J/mol, array(37840000.0) * J/mol, array(44100000.0) * J/mol, array(47206000.0) * J/mol, array(122200000.0) * J/mol, array(131000000.0) * J/mol, array(140500000.0) * J/mol, array(152600000.0) * J/mol, array(163000000.0) * J/mol], 'vaporization': array(347000.0) * J/mol, 'atomic_number': 26, 'rigidity_modulus': array(82.0) * GPa, 'symbol': 'Fe', 'covalent_radius': array(132.0) * pm, 'fusion': array(13800.0) * J/mol, 'pettifor': 0.98999999999999999, 'atomization': array(415000.0) * J/mol, 'poisson_ratio': array(0.28999999999999998) * dimensionless, 'electron_affinity': array(15700.0) * J/mol, 'name': 'Iron', 'boiling_point': array(3134.0) * Kelvin, 'density': array(7.8739999999999997) * g*cm**3, 'double_bond_radius': array(109.0) * pm, 'allred_rochow': 1.6399999999999999, 'young_modulus': array(211.0) * GPa, 'thermal_expansion': array(1.1800000000000001e-05) * 1/Kelvin, 'atomic_radius': array(140.0) * pm})
"""
Iron elemental data.

  - single_bond_radius: 116.0 pm
  - pauling: 1.83
  - molar_volume: 7.09 cm**3/mol
  - atomization: 415000.0 J/mol
  - sound_velocity: 4910.0 m/s
  - sanderson: 2.2
  - atomic_weight: 55.845
  - triple_bond_radius: 102.0 pm
  - melting_point: 1811.0 K
  - orbital_radii: (array(0.5027183865499999) * angstrom, array(0.61384560883999995) * angstrom, array(0.11641899477999999) * angstrom)
  - thermal_conductivity: 80.0 W/(m*K)
  - ionization_energies: [array(762500.0) * J/mol, array(1561900.0) * J/mol, array(2957000.0) * J/mol, array(5290000.0) * J/mol, array(7240000.0) * J/mol, array(9560000.0) * J/mol, array(12060000.0) * J/mol, array(14580000.0) * J/mol, array(22540000.0) * J/mol, array(25290000.0) * J/mol, array(28000000.0) * J/mol, array(31920000.0) * J/mol, array(34830000.0) * J/mol, array(37840000.0) * J/mol, array(44100000.0) * J/mol, array(47206000.0) * J/mol, array(122200000.0) * J/mol, array(131000000.0) * J/mol, array(140500000.0) * J/mol, array(152600000.0) * J/mol, array(163000000.0) * J/mol]
  - vaporization: 347000.0 J/mol
  - atomic_number: 26
  - rigidity_modulus: 82.0 GPa
  - symbol: Fe
  - covalent_radius: 132.0 pm
  - fusion: 13800.0 J/mol
  - pettifor: 0.99
  - bulk_modulus: 170.0 GPa
  - poisson_ratio: 0.29 dimensionless
  - electron_affinity: 15700.0 J/mol
  - name: Iron
  - boiling_point: 3134.0 K
  - density: 7.874 g*cm**3
  - double_bond_radius: 109.0 pm
  - allred_rochow: 1.64
  - young_modulus: 211.0 GPa
  - thermal_expansion: 1.18e-05 1/K
  - atomic_radius: 140.0 pm
"""

Co = Element(**{'single_bond_radius': array(111.0) * pm, 'pauling': 1.8799999999999999, 'molar_volume': array(6.6699999999999999) * cm**3/mol, 'bulk_modulus': array(180.0) * GPa, 'sound_velocity': array(4720.0) * m/s, 'sanderson': 2.5600000000000001, 'atomic_weight': 58.933194999999998, 'triple_bond_radius': array(96.0) * pm, 'melting_point': array(1768.0) * Kelvin, 'orbital_radii': (array(0.48684306907999997) * angstrom, array(0.58209497389999998) * angstrom, array(0.11112722228999999) * angstrom), 'thermal_conductivity': array(100.0) * Watt/(m*Kelvin), 'ionization_energies': [array(760400.0) * J/mol, array(1648000.0) * J/mol, array(3232000.0) * J/mol, array(4950000.0) * J/mol, array(7670000.0) * J/mol, array(9840000.0) * J/mol, array(12440000.0) * J/mol, array(15230000.0) * J/mol, array(17959000.0) * J/mol, array(26570000.0) * J/mol, array(29400000.0) * J/mol, array(32400000.0) * J/mol, array(36600000.0) * J/mol, array(39700000.0) * J/mol, array(42800000.0) * J/mol, array(49396000.0) * J/mol, array(52737000.0) * J/mol, array(134810000.0) * J/mol, array(145170000.0) * J/mol, array(154700000.0) * J/mol, array(167400000.0) * J/mol], 'vaporization': array(375000.0) * J/mol, 'atomic_number': 27, 'rigidity_modulus': array(75.0) * GPa, 'symbol': 'Co', 'covalent_radius': array(126.0) * pm, 'fusion': array(16200.0) * J/mol, 'pettifor': 1.04, 'atomization': array(426000.0) * J/mol, 'poisson_ratio': array(0.31) * dimensionless, 'electron_affinity': array(63700.0) * J/mol, 'name': 'Cobalt', 'boiling_point': array(3200.0) * Kelvin, 'density': array(8.9000000000000004) * g*cm**3, 'double_bond_radius': array(103.0) * pm, 'allred_rochow': 1.7, 'young_modulus': array(209.0) * GPa, 'thermal_expansion': array(1.2999999999999999e-05) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Cobalt elemental data.

  - single_bond_radius: 111.0 pm
  - pauling: 1.88
  - molar_volume: 6.67 cm**3/mol
  - atomization: 426000.0 J/mol
  - sound_velocity: 4720.0 m/s
  - sanderson: 2.56
  - atomic_weight: 58.933195
  - triple_bond_radius: 96.0 pm
  - melting_point: 1768.0 K
  - orbital_radii: (array(0.48684306907999997) * angstrom, array(0.58209497389999998) * angstrom, array(0.11112722228999999) * angstrom)
  - thermal_conductivity: 100.0 W/(m*K)
  - ionization_energies: [array(760400.0) * J/mol, array(1648000.0) * J/mol, array(3232000.0) * J/mol, array(4950000.0) * J/mol, array(7670000.0) * J/mol, array(9840000.0) * J/mol, array(12440000.0) * J/mol, array(15230000.0) * J/mol, array(17959000.0) * J/mol, array(26570000.0) * J/mol, array(29400000.0) * J/mol, array(32400000.0) * J/mol, array(36600000.0) * J/mol, array(39700000.0) * J/mol, array(42800000.0) * J/mol, array(49396000.0) * J/mol, array(52737000.0) * J/mol, array(134810000.0) * J/mol, array(145170000.0) * J/mol, array(154700000.0) * J/mol, array(167400000.0) * J/mol]
  - vaporization: 375000.0 J/mol
  - atomic_number: 27
  - rigidity_modulus: 75.0 GPa
  - symbol: Co
  - covalent_radius: 126.0 pm
  - fusion: 16200.0 J/mol
  - pettifor: 1.04
  - bulk_modulus: 180.0 GPa
  - poisson_ratio: 0.31 dimensionless
  - electron_affinity: 63700.0 J/mol
  - name: Cobalt
  - boiling_point: 3200.0 K
  - density: 8.9 g*cm**3
  - double_bond_radius: 103.0 pm
  - allred_rochow: 1.7
  - young_modulus: 209.0 GPa
  - thermal_expansion: 1.3e-05 1/K
  - atomic_radius: 135.0 pm
"""

Ni = Element(**{'single_bond_radius': array(110.0) * pm, 'pauling': 1.9099999999999999, 'molar_volume': array(6.5899999999999999) * cm**3/mol, 'bulk_modulus': array(180.0) * GPa, 'sound_velocity': array(4970.0) * m/s, 'sanderson': 1.9399999999999999, 'atomic_weight': 58.693399999999997, 'triple_bond_radius': array(101.0) * pm, 'melting_point': array(1728.0) * Kelvin, 'orbital_radii': (array(0.50801015903999991) * angstrom, array(0.64559624377999991) * angstrom, array(0.103189563555) * angstrom), 'thermal_conductivity': array(91.0) * Watt/(m*Kelvin), 'ionization_energies': [array(737100.0) * J/mol, array(1753000.0) * J/mol, array(3395000.0) * J/mol, array(5300000.0) * J/mol, array(7339000.0) * J/mol, array(10400000.0) * J/mol, array(12800000.0) * J/mol, array(15600000.0) * J/mol, array(18600000.0) * J/mol, array(21670000.0) * J/mol, array(30970000.0) * J/mol, array(34000000.0) * J/mol, array(37100000.0) * J/mol, array(41500000.0) * J/mol, array(44800000.0) * J/mol, array(48100000.0) * J/mol, array(55101000.0) * J/mol, array(58570000.0) * J/mol, array(148700000.0) * J/mol, array(159000000.0) * J/mol, array(169400000.0) * J/mol], 'vaporization': array(378000.0) * J/mol, 'atomic_number': 28, 'rigidity_modulus': array(76.0) * GPa, 'symbol': 'Ni', 'covalent_radius': array(124.0) * pm, 'fusion': array(17200.0) * J/mol, 'pettifor': 1.0900000000000001, 'atomization': array(431000.0) * J/mol, 'poisson_ratio': array(0.31) * dimensionless, 'van_der_waals_radius': array(163.0) * pm, 'electron_affinity': array(112000.0) * J/mol, 'name': 'Nickel', 'boiling_point': array(3186.0) * Kelvin, 'density': array(8.9079999999999995) * g*cm**3, 'double_bond_radius': array(101.0) * pm, 'allred_rochow': 1.75, 'young_modulus': array(200.0) * GPa, 'thermal_expansion': array(1.34e-05) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Nickel elemental data.

  - single_bond_radius: 110.0 pm
  - pauling: 1.91
  - molar_volume: 6.59 cm**3/mol
  - atomization: 431000.0 J/mol
  - sound_velocity: 4970.0 m/s
  - sanderson: 1.94
  - atomic_weight: 58.6934
  - triple_bond_radius: 101.0 pm
  - melting_point: 1728.0 K
  - orbital_radii: (array(0.50801015903999991) * angstrom, array(0.64559624377999991) * angstrom, array(0.103189563555) * angstrom)
  - thermal_conductivity: 91.0 W/(m*K)
  - ionization_energies: [array(737100.0) * J/mol, array(1753000.0) * J/mol, array(3395000.0) * J/mol, array(5300000.0) * J/mol, array(7339000.0) * J/mol, array(10400000.0) * J/mol, array(12800000.0) * J/mol, array(15600000.0) * J/mol, array(18600000.0) * J/mol, array(21670000.0) * J/mol, array(30970000.0) * J/mol, array(34000000.0) * J/mol, array(37100000.0) * J/mol, array(41500000.0) * J/mol, array(44800000.0) * J/mol, array(48100000.0) * J/mol, array(55101000.0) * J/mol, array(58570000.0) * J/mol, array(148700000.0) * J/mol, array(159000000.0) * J/mol, array(169400000.0) * J/mol]
  - vaporization: 378000.0 J/mol
  - atomic_number: 28
  - rigidity_modulus: 76.0 GPa
  - symbol: Ni
  - covalent_radius: 124.0 pm
  - fusion: 17200.0 J/mol
  - pettifor: 1.09
  - bulk_modulus: 180.0 GPa
  - poisson_ratio: 0.31 dimensionless
  - van_der_waals_radius: 163.0 pm
  - electron_affinity: 112000.0 J/mol
  - name: Nickel
  - boiling_point: 3186.0 K
  - density: 8.908 g*cm**3
  - double_bond_radius: 101.0 pm
  - allred_rochow: 1.75
  - young_modulus: 200.0 GPa
  - thermal_expansion: 1.34e-05 1/K
  - atomic_radius: 135.0 pm
"""

Cu = Element(**{'mulliken_jaffe': 1.49, 'single_bond_radius': array(112.0) * pm, 'pauling': 1.8999999999999999, 'molar_volume': array(7.1100000000000003) * cm**3/mol, 'bulk_modulus': array(140.0) * GPa, 'sound_velocity': array(3570.0) * m/s, 'sanderson': 1.98, 'atomic_weight': 63.545999999999999, 'triple_bond_radius': array(120.0) * pm, 'melting_point': array(1357.77) * Kelvin, 'orbital_radii': (array(0.46567597911999997) * angstrom, array(0.61384560883999995) * angstrom, array(0.097897791064999989) * angstrom), 'thermal_conductivity': array(400.0) * Watt/(m*Kelvin), 'ionization_energies': [array(745500.0) * J/mol, array(1957900.0) * J/mol, array(3555000.0) * J/mol, array(5536000.0) * J/mol, array(7700000.0) * J/mol, array(9900000.0) * J/mol, array(13400000.0) * J/mol, array(16000000.0) * J/mol, array(19200000.0) * J/mol, array(22400000.0) * J/mol, array(25600000.0) * J/mol, array(35600000.0) * J/mol, array(38700000.0) * J/mol, array(42000000.0) * J/mol, array(46700000.0) * J/mol, array(50200000.0) * J/mol, array(53700000.0) * J/mol, array(61100000.0) * J/mol, array(64702000.0) * J/mol, array(163700000.0) * J/mol, array(174100000.0) * J/mol], 'vaporization': array(300000.0) * J/mol, 'atomic_number': 29, 'rigidity_modulus': array(48.0) * GPa, 'symbol': 'Cu', 'covalent_radius': array(132.0) * pm, 'fusion': array(13100.0) * J/mol, 'pettifor': 1.2, 'atomization': array(338000.0) * J/mol, 'poisson_ratio': array(0.34000000000000002) * dimensionless, 'van_der_waals_radius': array(140.0) * pm, 'electron_affinity': array(118400.0) * J/mol, 'name': 'Copper', 'boiling_point': array(3200.0) * Kelvin, 'density': array(8.9199999999999999) * g*cm**3, 'double_bond_radius': array(115.0) * pm, 'allred_rochow': 1.75, 'young_modulus': array(130.0) * GPa, 'thermal_expansion': array(1.6499999999999998e-05) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Copper elemental data.

  - mulliken_jaffe: 1.49
  - single_bond_radius: 112.0 pm
  - pauling: 1.9
  - molar_volume: 7.11 cm**3/mol
  - atomization: 338000.0 J/mol
  - sound_velocity: 3570.0 m/s
  - sanderson: 1.98
  - atomic_weight: 63.546
  - triple_bond_radius: 120.0 pm
  - melting_point: 1357.77 K
  - orbital_radii: (array(0.46567597911999997) * angstrom, array(0.61384560883999995) * angstrom, array(0.097897791064999989) * angstrom)
  - thermal_conductivity: 400.0 W/(m*K)
  - ionization_energies: [array(745500.0) * J/mol, array(1957900.0) * J/mol, array(3555000.0) * J/mol, array(5536000.0) * J/mol, array(7700000.0) * J/mol, array(9900000.0) * J/mol, array(13400000.0) * J/mol, array(16000000.0) * J/mol, array(19200000.0) * J/mol, array(22400000.0) * J/mol, array(25600000.0) * J/mol, array(35600000.0) * J/mol, array(38700000.0) * J/mol, array(42000000.0) * J/mol, array(46700000.0) * J/mol, array(50200000.0) * J/mol, array(53700000.0) * J/mol, array(61100000.0) * J/mol, array(64702000.0) * J/mol, array(163700000.0) * J/mol, array(174100000.0) * J/mol]
  - vaporization: 300000.0 J/mol
  - atomic_number: 29
  - rigidity_modulus: 48.0 GPa
  - symbol: Cu
  - covalent_radius: 132.0 pm
  - fusion: 13100.0 J/mol
  - pettifor: 1.2
  - bulk_modulus: 140.0 GPa
  - poisson_ratio: 0.34 dimensionless
  - van_der_waals_radius: 140.0 pm
  - electron_affinity: 118400.0 J/mol
  - name: Copper
  - boiling_point: 3200.0 K
  - density: 8.92 g*cm**3
  - double_bond_radius: 115.0 pm
  - allred_rochow: 1.75
  - young_modulus: 130.0 GPa
  - thermal_expansion: 1.65e-05 1/K
  - atomic_radius: 135.0 pm
"""

Zn = Element(**{'mulliken_jaffe': 1.6499999999999999, 'single_bond_radius': array(118.0) * pm, 'pauling': 1.6499999999999999, 'molar_volume': array(9.1600000000000001) * cm**3/mol, 'bulk_modulus': array(70.0) * GPa, 'sound_velocity': array(3700.0) * m/s, 'sanderson': 2.23, 'atomic_weight': 65.379999999999995, 'melting_point': array(692.67999999999995) * Kelvin, 'orbital_radii': (array(0.43392534417999995) * angstrom, array(0.56092788394000004) * angstrom, array(0.09260601857499999) * angstrom), 'thermal_conductivity': array(120.0) * Watt/(m*Kelvin), 'ionization_energies': [array(906400.0) * J/mol, array(1733300.0) * J/mol, array(3833000.0) * J/mol, array(5731000.0) * J/mol, array(7970000.0) * J/mol, array(10400000.0) * J/mol, array(12900000.0) * J/mol, array(16800000.0) * J/mol, array(19600000.0) * J/mol, array(23000000.0) * J/mol, array(26400000.0) * J/mol, array(29990000.0) * J/mol, array(40490000.0) * J/mol, array(43800000.0) * J/mol, array(47300000.0) * J/mol, array(52300000.0) * J/mol, array(55900000.0) * J/mol, array(59700000.0) * J/mol, array(67300000.0) * J/mol, array(71200000.0) * J/mol, array(179100000.0) * J/mol], 'vaporization': array(119000.0) * J/mol, 'atomic_number': 30, 'rigidity_modulus': array(43.0) * GPa, 'symbol': 'Zn', 'covalent_radius': array(122.0) * pm, 'fusion': array(7350.0) * J/mol, 'pettifor': 1.4399999999999999, 'atomization': array(131000.0) * J/mol, 'poisson_ratio': array(0.25) * dimensionless, 'van_der_waals_radius': array(139.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Zinc', 'boiling_point': array(1180.0) * Kelvin, 'density': array(7.1399999999999997) * g*cm**3, 'double_bond_radius': array(120.0) * pm, 'allred_rochow': 1.6599999999999999, 'young_modulus': array(108.0) * GPa, 'thermal_expansion': array(3.0199999999999999e-05) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Zinc elemental data.

  - mulliken_jaffe: 1.65
  - single_bond_radius: 118.0 pm
  - pauling: 1.65
  - molar_volume: 9.16 cm**3/mol
  - atomization: 131000.0 J/mol
  - sound_velocity: 3700.0 m/s
  - sanderson: 2.23
  - atomic_weight: 65.38
  - melting_point: 692.68 K
  - orbital_radii: (array(0.43392534417999995) * angstrom, array(0.56092788394000004) * angstrom, array(0.09260601857499999) * angstrom)
  - thermal_conductivity: 120.0 W/(m*K)
  - ionization_energies: [array(906400.0) * J/mol, array(1733300.0) * J/mol, array(3833000.0) * J/mol, array(5731000.0) * J/mol, array(7970000.0) * J/mol, array(10400000.0) * J/mol, array(12900000.0) * J/mol, array(16800000.0) * J/mol, array(19600000.0) * J/mol, array(23000000.0) * J/mol, array(26400000.0) * J/mol, array(29990000.0) * J/mol, array(40490000.0) * J/mol, array(43800000.0) * J/mol, array(47300000.0) * J/mol, array(52300000.0) * J/mol, array(55900000.0) * J/mol, array(59700000.0) * J/mol, array(67300000.0) * J/mol, array(71200000.0) * J/mol, array(179100000.0) * J/mol]
  - vaporization: 119000.0 J/mol
  - atomic_number: 30
  - rigidity_modulus: 43.0 GPa
  - symbol: Zn
  - covalent_radius: 122.0 pm
  - fusion: 7350.0 J/mol
  - pettifor: 1.44
  - bulk_modulus: 70.0 GPa
  - poisson_ratio: 0.25 dimensionless
  - van_der_waals_radius: 139.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Zinc
  - boiling_point: 1180.0 K
  - density: 7.14 g*cm**3
  - double_bond_radius: 120.0 pm
  - allred_rochow: 1.66
  - young_modulus: 108.0 GPa
  - thermal_expansion: 3.02e-05 1/K
  - atomic_radius: 135.0 pm
"""

Ga = Element(**{'mulliken_jaffe': 2.0099999999999998, 'single_bond_radius': array(124.0) * pm, 'pauling': 1.8100000000000001, 'molar_volume': array(11.800000000000001) * cm**3/mol, 'atomization': array(277000.0) * J/mol, 'sound_velocity': array(2740.0) * m/s, 'sanderson': 2.4199999999999999, 'atomic_weight': 69.722999999999999, 'triple_bond_radius': array(121.0) * pm, 'melting_point': array(302.91000000000003) * Kelvin, 'allen': 1.756, 'orbital_radii': (array(0.40217470923999998) * angstrom, array(0.49478072781499999) * angstrom, array(0.089960132329999998) * angstrom), 'thermal_conductivity': array(29.0) * Watt/(m*Kelvin), 'ionization_energies': [array(578800.0) * J/mol, array(1979300.0) * J/mol, array(2963000.0) * J/mol, array(6180000.0) * J/mol], 'vaporization': array(256000.0) * J/mol, 'atomic_number': 31, 'symbol': 'Ga', 'covalent_radius': array(122.0) * pm, 'fusion': array(5590.0) * J/mol, 'pettifor': 1.6799999999999999, 'van_der_waals_radius': array(187.0) * pm, 'electron_affinity': array(28900.0) * J/mol, 'name': 'Gallium', 'boiling_point': array(2477.0) * Kelvin, 'density': array(5.9039999999999999) * g*cm**3, 'double_bond_radius': array(117.0) * pm, 'allred_rochow': 1.8200000000000001, 'thermal_expansion': array(0.00011999999999999999) * 1/Kelvin, 'atomic_radius': array(130.0) * pm})
"""
Gallium elemental data.

  - mulliken_jaffe: 2.01
  - single_bond_radius: 124.0 pm
  - pauling: 1.81
  - molar_volume: 11.8 cm**3/mol
  - atomization: 277000.0 J/mol
  - sound_velocity: 2740.0 m/s
  - sanderson: 2.42
  - atomic_weight: 69.723
  - triple_bond_radius: 121.0 pm
  - melting_point: 302.91 K
  - allen: 1.756
  - orbital_radii: (array(0.40217470923999998) * angstrom, array(0.49478072781499999) * angstrom, array(0.089960132329999998) * angstrom)
  - thermal_conductivity: 29.0 W/(m*K)
  - ionization_energies: [array(578800.0) * J/mol, array(1979300.0) * J/mol, array(2963000.0) * J/mol, array(6180000.0) * J/mol]
  - vaporization: 256000.0 J/mol
  - atomic_number: 31
  - symbol: Ga
  - covalent_radius: 122.0 pm
  - fusion: 5590.0 J/mol
  - pettifor: 1.68
  - van_der_waals_radius: 187.0 pm
  - electron_affinity: 28900.0 J/mol
  - name: Gallium
  - boiling_point: 2477.0 K
  - density: 5.904 g*cm**3
  - double_bond_radius: 117.0 pm
  - allred_rochow: 1.82
  - thermal_expansion: 0.00012 1/K
  - atomic_radius: 130.0 pm
"""

Ge = Element(**{'mulliken_jaffe': 2.3300000000000001, 'single_bond_radius': array(121.0) * pm, 'pauling': 2.0099999999999998, 'molar_volume': array(13.630000000000001) * cm**3/mol, 'atomization': array(377000.0) * J/mol, 'sound_velocity': array(5400.0) * m/s, 'sanderson': 2.6200000000000001, 'atomic_weight': 72.640000000000001, 'triple_bond_radius': array(114.0) * pm, 'melting_point': array(1211.4000000000001) * Kelvin, 'allen': 1.994, 'orbital_radii': (array(0.38100761927999993) * angstrom, array(0.44450888915999998) * angstrom, array(0.084668359839999999) * angstrom), 'thermal_conductivity': array(60.0) * Watt/(m*Kelvin), 'ionization_energies': [array(762000.0) * J/mol, array(1537500.0) * J/mol, array(3302100.0) * J/mol, array(4411000.0) * J/mol, array(9020000.0) * J/mol], 'vaporization': array(334000.0) * J/mol, 'atomic_number': 32, 'symbol': 'Ge', 'covalent_radius': array(120.0) * pm, 'fusion': array(31800.0) * J/mol, 'pettifor': 1.9199999999999999, 'electron_affinity': array(119000.0) * J/mol, 'name': 'Germanium', 'boiling_point': array(3093.0) * Kelvin, 'density': array(5.3230000000000004) * g*cm**3, 'double_bond_radius': array(111.0) * pm, 'allred_rochow': 2.02, 'thermal_expansion': array(6.0000000000000002e-06) * 1/Kelvin, 'atomic_radius': array(125.0) * pm})
"""
Germanium elemental data.

  - mulliken_jaffe: 2.33
  - single_bond_radius: 121.0 pm
  - pauling: 2.01
  - molar_volume: 13.63 cm**3/mol
  - atomization: 377000.0 J/mol
  - sound_velocity: 5400.0 m/s
  - sanderson: 2.62
  - atomic_weight: 72.64
  - triple_bond_radius: 114.0 pm
  - melting_point: 1211.4 K
  - allen: 1.994
  - orbital_radii: (array(0.38100761927999993) * angstrom, array(0.44450888915999998) * angstrom, array(0.084668359839999999) * angstrom)
  - thermal_conductivity: 60.0 W/(m*K)
  - ionization_energies: [array(762000.0) * J/mol, array(1537500.0) * J/mol, array(3302100.0) * J/mol, array(4411000.0) * J/mol, array(9020000.0) * J/mol]
  - vaporization: 334000.0 J/mol
  - atomic_number: 32
  - symbol: Ge
  - covalent_radius: 120.0 pm
  - fusion: 31800.0 J/mol
  - pettifor: 1.92
  - electron_affinity: 119000.0 J/mol
  - name: Germanium
  - boiling_point: 3093.0 K
  - density: 5.323 g*cm**3
  - double_bond_radius: 111.0 pm
  - allred_rochow: 2.02
  - thermal_expansion: 6e-06 1/K
  - atomic_radius: 125.0 pm
"""

As = Element(**{'mulliken_jaffe': 2.2599999999999998, 'single_bond_radius': array(121.0) * pm, 'pauling': 2.1800000000000002, 'molar_volume': array(12.949999999999999) * cm**3/mol, 'bulk_modulus': array(22.0) * GPa, 'sanderson': 2.8199999999999998, 'atomic_weight': 74.921599999999998, 'critical_temperature': array(1700.0) * Kelvin, 'triple_bond_radius': array(106.0) * pm, 'melting_point': array(1090.0) * Kelvin, 'allen': 2.2109999999999999, 'orbital_radii': (array(0.35454875682999998) * angstrom, array(0.39423705050499996) * angstrom, array(0.082022473594999992) * angstrom), 'thermal_conductivity': array(50.0) * Watt/(m*Kelvin), 'ionization_energies': [array(947000.0) * J/mol, array(1798000.0) * J/mol, array(2735000.0) * J/mol, array(4837000.0) * J/mol, array(6043000.0) * J/mol, array(12310000.0) * J/mol], 'vaporization': array(32400.0) * J/mol, 'atomic_number': 33, 'symbol': 'As', 'covalent_radius': array(119.0) * pm, 'fusion': array(27700.0) * J/mol, 'pettifor': 2.1600000000000001, 'atomization': array(302000.0) * J/mol, 'van_der_waals_radius': array(185.0) * pm, 'electron_affinity': array(78000.0) * J/mol, 'name': 'Arsenic', 'boiling_point': array(887.0) * Kelvin, 'density': array(5.7270000000000003) * g*cm**3, 'double_bond_radius': array(114.0) * pm, 'allred_rochow': 2.2000000000000002, 'young_modulus': array(8.0) * GPa, 'atomic_radius': array(115.0) * pm})
"""
Arsenic elemental data.

  - mulliken_jaffe: 2.26
  - single_bond_radius: 121.0 pm
  - pauling: 2.18
  - molar_volume: 12.95 cm**3/mol
  - atomization: 302000.0 J/mol
  - sanderson: 2.82
  - atomic_weight: 74.9216
  - critical_temperature: 1700.0 K
  - triple_bond_radius: 106.0 pm
  - melting_point: 1090.0 K
  - allen: 2.211
  - orbital_radii: (array(0.35454875682999998) * angstrom, array(0.39423705050499996) * angstrom, array(0.082022473594999992) * angstrom)
  - thermal_conductivity: 50.0 W/(m*K)
  - ionization_energies: [array(947000.0) * J/mol, array(1798000.0) * J/mol, array(2735000.0) * J/mol, array(4837000.0) * J/mol, array(6043000.0) * J/mol, array(12310000.0) * J/mol]
  - vaporization: 32400.0 J/mol
  - atomic_number: 33
  - symbol: As
  - covalent_radius: 119.0 pm
  - fusion: 27700.0 J/mol
  - pettifor: 2.16
  - bulk_modulus: 22.0 GPa
  - van_der_waals_radius: 185.0 pm
  - electron_affinity: 78000.0 J/mol
  - name: Arsenic
  - boiling_point: 887.0 K
  - density: 5.727 g*cm**3
  - double_bond_radius: 114.0 pm
  - allred_rochow: 2.2
  - young_modulus: 8.0 GPa
  - atomic_radius: 115.0 pm
"""

Se = Element(**{'mulliken_jaffe': 2.6000000000000001, 'single_bond_radius': array(116.0) * pm, 'pauling': 2.5499999999999998, 'molar_volume': array(16.420000000000002) * cm**3/mol, 'atomization': array(227000.0) * J/mol, 'sound_velocity': array(3350.0) * m/s, 'sanderson': 3.0099999999999998, 'atomic_weight': 78.959999999999994, 'critical_temperature': array(1766.0) * Kelvin, 'triple_bond_radius': array(107.0) * pm, 'melting_point': array(494.0) * Kelvin, 'allen': 2.4239999999999999, 'orbital_radii': (array(0.32544400813499996) * angstrom, array(0.35454875682999998) * angstrom, array(0.079376587349999986) * angstrom), 'thermal_conductivity': array(0.52000000000000002) * Watt/(m*Kelvin), 'ionization_energies': [array(941000.0) * J/mol, array(2045000.0) * J/mol, array(2973700.0) * J/mol, array(4144000.0) * J/mol, array(6590000.0) * J/mol, array(7880000.0) * J/mol, array(14990000.0) * J/mol], 'vaporization': array(26000.0) * J/mol, 'atomic_number': 34, 'rigidity_modulus': array(3.7000000000000002) * GPa, 'symbol': 'Se', 'covalent_radius': array(120.0) * pm, 'fusion': array(5400.0) * J/mol, 'pettifor': 2.3999999999999999, 'bulk_modulus': array(8.3000000000000007) * GPa, 'poisson_ratio': array(0.33000000000000002) * dimensionless, 'van_der_waals_radius': array(190.0) * pm, 'electron_affinity': array(195000.0) * J/mol, 'name': 'Selenium', 'boiling_point': array(958.0) * Kelvin, 'density': array(4.819) * g*cm**3, 'double_bond_radius': array(107.0) * pm, 'allred_rochow': 2.48, 'young_modulus': array(10.0) * GPa, 'atomic_radius': array(115.0) * pm})
"""
Selenium elemental data.

  - mulliken_jaffe: 2.6
  - single_bond_radius: 116.0 pm
  - pauling: 2.55
  - molar_volume: 16.42 cm**3/mol
  - atomization: 227000.0 J/mol
  - sound_velocity: 3350.0 m/s
  - sanderson: 3.01
  - atomic_weight: 78.96
  - critical_temperature: 1766.0 K
  - triple_bond_radius: 107.0 pm
  - melting_point: 494.0 K
  - allen: 2.424
  - orbital_radii: (array(0.32544400813499996) * angstrom, array(0.35454875682999998) * angstrom, array(0.079376587349999986) * angstrom)
  - thermal_conductivity: 0.52 W/(m*K)
  - ionization_energies: [array(941000.0) * J/mol, array(2045000.0) * J/mol, array(2973700.0) * J/mol, array(4144000.0) * J/mol, array(6590000.0) * J/mol, array(7880000.0) * J/mol, array(14990000.0) * J/mol]
  - vaporization: 26000.0 J/mol
  - atomic_number: 34
  - rigidity_modulus: 3.7 GPa
  - symbol: Se
  - covalent_radius: 120.0 pm
  - fusion: 5400.0 J/mol
  - pettifor: 2.4
  - bulk_modulus: 8.3 GPa
  - poisson_ratio: 0.33 dimensionless
  - van_der_waals_radius: 190.0 pm
  - electron_affinity: 195000.0 J/mol
  - name: Selenium
  - boiling_point: 958.0 K
  - density: 4.819 g*cm**3
  - double_bond_radius: 107.0 pm
  - allred_rochow: 2.48
  - young_modulus: 10.0 GPa
  - atomic_radius: 115.0 pm
"""

Br = Element(**{'mulliken_jaffe': 2.9500000000000002, 'single_bond_radius': array(114.0) * pm, 'pauling': 2.96, 'molar_volume': array(19.780000000000001) * cm**3/mol, 'bulk_modulus': array(1.8999999999999999) * GPa, 'sanderson': 3.2200000000000002, 'atomic_weight': 79.903999999999996, 'critical_temperature': array(586.0) * Kelvin, 'triple_bond_radius': array(110.0) * pm, 'melting_point': array(265.80000000000001) * Kelvin, 'allen': 2.6850000000000001, 'orbital_radii': (array(0.30692280441999997) * angstrom, array(0.32808989437999997) * angstrom, array(0.075672346606999993) * angstrom), 'thermal_conductivity': array(0.12) * Watt/(m*Kelvin), 'ionization_energies': [array(1139900.0) * J/mol, array(2103000.0) * J/mol, array(3470000.0) * J/mol, array(4560000.0) * J/mol, array(5760000.0) * J/mol, array(8550000.0) * J/mol, array(9940000.0) * J/mol, array(18600000.0) * J/mol], 'vaporization': array(14800.0) * J/mol, 'atomic_number': 35, 'symbol': 'Br', 'covalent_radius': array(120.0) * pm, 'fusion': array(5800.0) * J/mol, 'pettifor': 2.6400000000000001, 'atomization': array(112000.0) * J/mol, 'van_der_waals_radius': array(185.0) * pm, 'electron_affinity': array(324600.0) * J/mol, 'name': 'Bromine', 'boiling_point': array(332.0) * Kelvin, 'double_bond_radius': array(109.0) * pm, 'allred_rochow': 2.7400000000000002, 'atomic_radius': array(115.0) * pm})
"""
Bromine elemental data.

  - mulliken_jaffe: 2.95
  - single_bond_radius: 114.0 pm
  - pauling: 2.96
  - molar_volume: 19.78 cm**3/mol
  - atomization: 112000.0 J/mol
  - sanderson: 3.22
  - atomic_weight: 79.904
  - critical_temperature: 586.0 K
  - triple_bond_radius: 110.0 pm
  - melting_point: 265.8 K
  - allen: 2.685
  - orbital_radii: (array(0.30692280441999997) * angstrom, array(0.32808989437999997) * angstrom, array(0.075672346606999993) * angstrom)
  - thermal_conductivity: 0.12 W/(m*K)
  - ionization_energies: [array(1139900.0) * J/mol, array(2103000.0) * J/mol, array(3470000.0) * J/mol, array(4560000.0) * J/mol, array(5760000.0) * J/mol, array(8550000.0) * J/mol, array(9940000.0) * J/mol, array(18600000.0) * J/mol]
  - vaporization: 14800.0 J/mol
  - atomic_number: 35
  - symbol: Br
  - covalent_radius: 120.0 pm
  - fusion: 5800.0 J/mol
  - pettifor: 2.64
  - bulk_modulus: 1.9 GPa
  - van_der_waals_radius: 185.0 pm
  - electron_affinity: 324600.0 J/mol
  - name: Bromine
  - boiling_point: 332.0 K
  - double_bond_radius: 109.0 pm
  - allred_rochow: 2.74
  - atomic_radius: 115.0 pm
"""

Kr = Element(**{'mulliken_jaffe': 3.0, 'single_bond_radius': array(117.0) * pm, 'pauling': 3.0, 'molar_volume': array(27.989999999999998) * cm**3/mol, 'atomization': array(0.0) * J/mol, 'sound_velocity': array(1120.0) * m/s, 'sanderson': 2.9100000000000001, 'atomic_weight': 83.798000000000002, 'critical_temperature': array(209.40000000000001) * Kelvin, 'triple_bond_radius': array(108.0) * pm, 'melting_point': array(115.79000000000001) * Kelvin, 'allen': 2.9660000000000002, 'orbital_radii': (array(0.29633925944) * angstrom, array(0.31750634939999994) * angstrom, array(0.073026460362000001) * angstrom), 'thermal_conductivity': array(0.0094299999999999991) * Watt/(m*Kelvin), 'ionization_energies': [array(1350800.0) * J/mol, array(2350400.0) * J/mol, array(3565000.0) * J/mol, array(5070000.0) * J/mol, array(6240000.0) * J/mol, array(7570000.0) * J/mol, array(10710000.0) * J/mol, array(12138000.0) * J/mol, array(22274000.0) * J/mol, array(25880000.0) * J/mol, array(29700000.0) * J/mol, array(33800000.0) * J/mol, array(37700000.0) * J/mol, array(43100000.0) * J/mol, array(47500000.0) * J/mol, array(52200000.0) * J/mol, array(57100000.0) * J/mol, array(61800000.0) * J/mol, array(75800000.0) * J/mol, array(80400000.0) * J/mol, array(85300000.0) * J/mol], 'vaporization': array(9020.0) * J/mol, 'atomic_number': 36, 'symbol': 'Kr', 'covalent_radius': array(116.0) * pm, 'fusion': array(1640.0) * J/mol, 'van_der_waals_radius': array(202.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Krypton', 'boiling_point': array(119.93000000000001) * Kelvin, 'double_bond_radius': array(121.0) * pm, 'allred_rochow': 2.9399999999999999})
"""
Krypton elemental data.

  - mulliken_jaffe: 3.0
  - single_bond_radius: 117.0 pm
  - pauling: 3.0
  - molar_volume: 27.99 cm**3/mol
  - atomization: 0.0 J/mol
  - sound_velocity: 1120.0 m/s
  - sanderson: 2.91
  - atomic_weight: 83.798
  - critical_temperature: 209.4 K
  - triple_bond_radius: 108.0 pm
  - melting_point: 115.79 K
  - allen: 2.966
  - orbital_radii: (array(0.29633925944) * angstrom, array(0.31750634939999994) * angstrom, array(0.073026460362000001) * angstrom)
  - thermal_conductivity: 0.00943 W/(m*K)
  - ionization_energies: [array(1350800.0) * J/mol, array(2350400.0) * J/mol, array(3565000.0) * J/mol, array(5070000.0) * J/mol, array(6240000.0) * J/mol, array(7570000.0) * J/mol, array(10710000.0) * J/mol, array(12138000.0) * J/mol, array(22274000.0) * J/mol, array(25880000.0) * J/mol, array(29700000.0) * J/mol, array(33800000.0) * J/mol, array(37700000.0) * J/mol, array(43100000.0) * J/mol, array(47500000.0) * J/mol, array(52200000.0) * J/mol, array(57100000.0) * J/mol, array(61800000.0) * J/mol, array(75800000.0) * J/mol, array(80400000.0) * J/mol, array(85300000.0) * J/mol]
  - vaporization: 9020.0 J/mol
  - atomic_number: 36
  - symbol: Kr
  - covalent_radius: 116.0 pm
  - fusion: 1640.0 J/mol
  - van_der_waals_radius: 202.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Krypton
  - boiling_point: 119.93 K
  - double_bond_radius: 121.0 pm
  - allred_rochow: 2.94
"""

Rb = Element(**{'mulliken_jaffe': 0.68999999999999995, 'single_bond_radius': array(210.0) * pm, 'pauling': 0.81999999999999995, 'molar_volume': array(55.759999999999998) * cm**3/mol, 'bulk_modulus': array(2.5) * GPa, 'sound_velocity': array(1300.0) * m/s, 'sanderson': 0.31, 'atomic_weight': 85.467799999999997, 'critical_temperature': array(2093.0) * Kelvin, 'melting_point': array(312.45999999999998) * Kelvin, 'allen': 0.70599999999999996, 'orbital_radii': (array(0.88372600582999994) * angstrom, array(1.2859007150699999) * angstrom, array(0.37571584678999997) * angstrom), 'thermal_conductivity': array(58.0) * Watt/(m*Kelvin), 'ionization_energies': [array(403000.0) * J/mol, array(2633000.0) * J/mol, array(3860000.0) * J/mol, array(5080000.0) * J/mol, array(6850000.0) * J/mol, array(8140000.0) * J/mol, array(9570000.0) * J/mol, array(13120000.0) * J/mol, array(14500000.0) * J/mol, array(26740000.0) * J/mol], 'vaporization': array(72000.0) * J/mol, 'atomic_number': 37, 'symbol': 'Rb', 'covalent_radius': array(220.0) * pm, 'fusion': array(2190.0) * J/mol, 'pettifor': 0.29999999999999999, 'atomization': array(81000.0) * J/mol, 'electron_affinity': array(46900.0) * J/mol, 'name': 'Rubidium', 'boiling_point': array(961.0) * Kelvin, 'density': array(1.532) * g*cm**3, 'double_bond_radius': array(202.0) * pm, 'allred_rochow': 0.89000000000000001, 'young_modulus': array(2.3999999999999999) * GPa, 'atomic_radius': array(235.0) * pm})
"""
Rubidium elemental data.

  - mulliken_jaffe: 0.69
  - single_bond_radius: 210.0 pm
  - pauling: 0.82
  - molar_volume: 55.76 cm**3/mol
  - atomization: 81000.0 J/mol
  - sound_velocity: 1300.0 m/s
  - sanderson: 0.31
  - atomic_weight: 85.4678
  - critical_temperature: 2093.0 K
  - melting_point: 312.46 K
  - allen: 0.706
  - orbital_radii: (array(0.88372600582999994) * angstrom, array(1.2859007150699999) * angstrom, array(0.37571584678999997) * angstrom)
  - thermal_conductivity: 58.0 W/(m*K)
  - ionization_energies: [array(403000.0) * J/mol, array(2633000.0) * J/mol, array(3860000.0) * J/mol, array(5080000.0) * J/mol, array(6850000.0) * J/mol, array(8140000.0) * J/mol, array(9570000.0) * J/mol, array(13120000.0) * J/mol, array(14500000.0) * J/mol, array(26740000.0) * J/mol]
  - vaporization: 72000.0 J/mol
  - atomic_number: 37
  - symbol: Rb
  - covalent_radius: 220.0 pm
  - fusion: 2190.0 J/mol
  - pettifor: 0.3
  - bulk_modulus: 2.5 GPa
  - electron_affinity: 46900.0 J/mol
  - name: Rubidium
  - boiling_point: 961.0 K
  - density: 1.532 g*cm**3
  - double_bond_radius: 202.0 pm
  - allred_rochow: 0.89
  - young_modulus: 2.4 GPa
  - atomic_radius: 235.0 pm
"""

Sr = Element(**{'mulliken_jaffe': 1.0, 'single_bond_radius': array(185.0) * pm, 'pauling': 0.94999999999999996, 'molar_volume': array(33.939999999999998) * cm**3/mol, 'atomization': array(164000.0) * J/mol, 'sanderson': 0.71999999999999997, 'atomic_weight': 87.620000000000005, 'triple_bond_radius': array(139.0) * pm, 'melting_point': array(1050.0) * Kelvin, 'allen': 0.96299999999999997, 'orbital_radii': (array(0.75143169357999995) * angstrom, array(0.94722727570999998) * angstrom, array(0.33496919861699997) * angstrom), 'thermal_conductivity': array(35.0) * Watt/(m*Kelvin), 'ionization_energies': [array(549500.0) * J/mol, array(1064200.0) * J/mol, array(4138000.0) * J/mol, array(5500000.0) * J/mol, array(6910000.0) * J/mol, array(8760000.0) * J/mol, array(10230000.0) * J/mol, array(11800000.0) * J/mol, array(15600000.0) * J/mol, array(17100000.0) * J/mol, array(31270000.0) * J/mol], 'vaporization': array(137000.0) * J/mol, 'atomic_number': 38, 'rigidity_modulus': array(6.0999999999999996) * GPa, 'symbol': 'Sr', 'covalent_radius': array(195.0) * pm, 'fusion': array(8000.0) * J/mol, 'pettifor': 0.55000000000000004, 'poisson_ratio': array(0.28000000000000003) * dimensionless, 'electron_affinity': array(5030.0) * J/mol, 'name': 'Strontium', 'boiling_point': array(1655.0) * Kelvin, 'density': array(2.6299999999999999) * g*cm**3, 'double_bond_radius': array(157.0) * pm, 'allred_rochow': 0.98999999999999999, 'thermal_expansion': array(2.2499999999999998e-05) * 1/Kelvin, 'atomic_radius': array(200.0) * pm})
"""
Strontium elemental data.

  - mulliken_jaffe: 1.0
  - single_bond_radius: 185.0 pm
  - pauling: 0.95
  - molar_volume: 33.94 cm**3/mol
  - atomization: 164000.0 J/mol
  - sanderson: 0.72
  - atomic_weight: 87.62
  - triple_bond_radius: 139.0 pm
  - melting_point: 1050.0 K
  - allen: 0.963
  - orbital_radii: (array(0.75143169357999995) * angstrom, array(0.94722727570999998) * angstrom, array(0.33496919861699997) * angstrom)
  - thermal_conductivity: 35.0 W/(m*K)
  - ionization_energies: [array(549500.0) * J/mol, array(1064200.0) * J/mol, array(4138000.0) * J/mol, array(5500000.0) * J/mol, array(6910000.0) * J/mol, array(8760000.0) * J/mol, array(10230000.0) * J/mol, array(11800000.0) * J/mol, array(15600000.0) * J/mol, array(17100000.0) * J/mol, array(31270000.0) * J/mol]
  - vaporization: 137000.0 J/mol
  - atomic_number: 38
  - rigidity_modulus: 6.1 GPa
  - symbol: Sr
  - covalent_radius: 195.0 pm
  - fusion: 8000.0 J/mol
  - pettifor: 0.55
  - poisson_ratio: 0.28 dimensionless
  - electron_affinity: 5030.0 J/mol
  - name: Strontium
  - boiling_point: 1655.0 K
  - density: 2.63 g*cm**3
  - double_bond_radius: 157.0 pm
  - allred_rochow: 0.99
  - thermal_expansion: 2.25e-05 1/K
  - atomic_radius: 200.0 pm
"""

Y = Element(**{'single_bond_radius': array(163.0) * pm, 'pauling': 1.22, 'molar_volume': array(19.879999999999999) * cm**3/mol, 'bulk_modulus': array(41.0) * GPa, 'sound_velocity': array(3300.0) * m/s, 'sanderson': 0.65000000000000002, 'atomic_weight': 88.905850000000001, 'triple_bond_radius': array(124.0) * pm, 'melting_point': array(1799.0) * Kelvin, 'orbital_radii': (array(0.69851396867999993) * angstrom, array(0.85726714337999999) * angstrom, array(0.30692280441999997) * angstrom), 'thermal_conductivity': array(17.0) * Watt/(m*Kelvin), 'ionization_energies': [array(600000.0) * J/mol, array(1180000.0) * J/mol, array(1980000.0) * J/mol, array(5847000.0) * J/mol, array(7430000.0) * J/mol, array(8970000.0) * J/mol, array(11190000.0) * J/mol, array(12450000.0) * J/mol, array(14110000.0) * J/mol, array(18400000.0) * J/mol, array(19900000.0) * J/mol, array(36090000.0) * J/mol], 'vaporization': array(380000.0) * J/mol, 'atomic_number': 39, 'rigidity_modulus': array(26.0) * GPa, 'symbol': 'Y', 'covalent_radius': array(190.0) * pm, 'fusion': array(11400.0) * J/mol, 'pettifor': 0.69999999999999996, 'atomization': array(425000.0) * J/mol, 'poisson_ratio': array(0.23999999999999999) * dimensionless, 'electron_affinity': array(29600.0) * J/mol, 'name': 'Yttrium', 'boiling_point': array(3609.0) * Kelvin, 'density': array(4.4720000000000004) * g*cm**3, 'double_bond_radius': array(130.0) * pm, 'allred_rochow': 1.1100000000000001, 'young_modulus': array(64.0) * GPa, 'thermal_expansion': array(1.0599999999999998e-05) * 1/Kelvin, 'atomic_radius': array(180.0) * pm})
"""
Yttrium elemental data.

  - single_bond_radius: 163.0 pm
  - pauling: 1.22
  - molar_volume: 19.88 cm**3/mol
  - atomization: 425000.0 J/mol
  - sound_velocity: 3300.0 m/s
  - sanderson: 0.65
  - atomic_weight: 88.90585
  - triple_bond_radius: 124.0 pm
  - melting_point: 1799.0 K
  - orbital_radii: (array(0.69851396867999993) * angstrom, array(0.85726714337999999) * angstrom, array(0.30692280441999997) * angstrom)
  - thermal_conductivity: 17.0 W/(m*K)
  - ionization_energies: [array(600000.0) * J/mol, array(1180000.0) * J/mol, array(1980000.0) * J/mol, array(5847000.0) * J/mol, array(7430000.0) * J/mol, array(8970000.0) * J/mol, array(11190000.0) * J/mol, array(12450000.0) * J/mol, array(14110000.0) * J/mol, array(18400000.0) * J/mol, array(19900000.0) * J/mol, array(36090000.0) * J/mol]
  - vaporization: 380000.0 J/mol
  - atomic_number: 39
  - rigidity_modulus: 26.0 GPa
  - symbol: Y
  - covalent_radius: 190.0 pm
  - fusion: 11400.0 J/mol
  - pettifor: 0.7
  - bulk_modulus: 41.0 GPa
  - poisson_ratio: 0.24 dimensionless
  - electron_affinity: 29600.0 J/mol
  - name: Yttrium
  - boiling_point: 3609.0 K
  - density: 4.472 g*cm**3
  - double_bond_radius: 130.0 pm
  - allred_rochow: 1.11
  - young_modulus: 64.0 GPa
  - thermal_expansion: 1.06e-05 1/K
  - atomic_radius: 180.0 pm
"""

Zr = Element(**{'single_bond_radius': array(154.0) * pm, 'pauling': 1.3300000000000001, 'molar_volume': array(14.02) * cm**3/mol, 'atomization': array(605000.0) * J/mol, 'sound_velocity': array(3800.0) * m/s, 'sanderson': 0.90000000000000002, 'atomic_weight': 91.224000000000004, 'triple_bond_radius': array(121.0) * pm, 'melting_point': array(2128.0) * Kelvin, 'orbital_radii': (array(0.66940921998499991) * angstrom, array(0.82551650844000002) * angstrom, array(0.28575571445999998) * angstrom), 'thermal_conductivity': array(23.0) * Watt/(m*Kelvin), 'ionization_energies': [array(640100.0) * J/mol, array(1270000.0) * J/mol, array(2218000.0) * J/mol, array(3313000.0) * J/mol, array(7752000.0) * J/mol, array(9500000.0) * J/mol], 'vaporization': array(580000.0) * J/mol, 'atomic_number': 40, 'rigidity_modulus': array(33.0) * GPa, 'symbol': 'Zr', 'covalent_radius': array(175.0) * pm, 'fusion': array(21000.0) * J/mol, 'pettifor': 0.76000000000000001, 'poisson_ratio': array(0.34000000000000002) * dimensionless, 'electron_affinity': array(41100.0) * J/mol, 'name': 'Zirconium', 'boiling_point': array(4682.0) * Kelvin, 'density': array(6.5110000000000001) * g*cm**3, 'double_bond_radius': array(127.0) * pm, 'allred_rochow': 1.22, 'young_modulus': array(68.0) * GPa, 'thermal_expansion': array(5.6999999999999996e-06) * 1/Kelvin, 'atomic_radius': array(155.0) * pm})
"""
Zirconium elemental data.

  - single_bond_radius: 154.0 pm
  - pauling: 1.33
  - molar_volume: 14.02 cm**3/mol
  - atomization: 605000.0 J/mol
  - sound_velocity: 3800.0 m/s
  - sanderson: 0.9
  - atomic_weight: 91.224
  - triple_bond_radius: 121.0 pm
  - melting_point: 2128.0 K
  - orbital_radii: (array(0.66940921998499991) * angstrom, array(0.82551650844000002) * angstrom, array(0.28575571445999998) * angstrom)
  - thermal_conductivity: 23.0 W/(m*K)
  - ionization_energies: [array(640100.0) * J/mol, array(1270000.0) * J/mol, array(2218000.0) * J/mol, array(3313000.0) * J/mol, array(7752000.0) * J/mol, array(9500000.0) * J/mol]
  - vaporization: 580000.0 J/mol
  - atomic_number: 40
  - rigidity_modulus: 33.0 GPa
  - symbol: Zr
  - covalent_radius: 175.0 pm
  - fusion: 21000.0 J/mol
  - pettifor: 0.76
  - poisson_ratio: 0.34 dimensionless
  - electron_affinity: 41100.0 J/mol
  - name: Zirconium
  - boiling_point: 4682.0 K
  - density: 6.511 g*cm**3
  - double_bond_radius: 127.0 pm
  - allred_rochow: 1.22
  - young_modulus: 68.0 GPa
  - thermal_expansion: 5.7e-06 1/K
  - atomic_radius: 155.0 pm
"""

Nb = Element(**{'single_bond_radius': array(147.0) * pm, 'pauling': 1.6000000000000001, 'molar_volume': array(10.83) * cm**3/mol, 'bulk_modulus': array(170.0) * GPa, 'sound_velocity': array(3480.0) * m/s, 'sanderson': 1.4199999999999999, 'atomic_weight': 92.906379999999999, 'triple_bond_radius': array(116.0) * pm, 'melting_point': array(2750.0) * Kelvin, 'orbital_radii': (array(0.65088801626999993) * angstrom, array(0.80964119096999998) * angstrom, array(0.26988039698999999) * angstrom), 'thermal_conductivity': array(54.0) * Watt/(m*Kelvin), 'ionization_energies': [array(652100.0) * J/mol, array(1380000.0) * J/mol, array(2416000.0) * J/mol, array(3700000.0) * J/mol, array(4877000.0) * J/mol, array(9847000.0) * J/mol, array(12100000.0) * J/mol], 'vaporization': array(690000.0) * J/mol, 'atomic_number': 41, 'rigidity_modulus': array(38.0) * GPa, 'symbol': 'Nb', 'covalent_radius': array(164.0) * pm, 'fusion': array(26800.0) * J/mol, 'pettifor': 0.81999999999999995, 'atomization': array(733000.0) * J/mol, 'poisson_ratio': array(0.40000000000000002) * dimensionless, 'electron_affinity': array(86100.0) * J/mol, 'name': 'Niobium', 'boiling_point': array(5017.0) * Kelvin, 'density': array(8.5700000000000003) * g*cm**3, 'double_bond_radius': array(125.0) * pm, 'allred_rochow': 1.23, 'young_modulus': array(105.0) * GPa, 'thermal_expansion': array(7.2999999999999996e-06) * 1/Kelvin, 'atomic_radius': array(145.0) * pm})
"""
Niobium elemental data.

  - single_bond_radius: 147.0 pm
  - pauling: 1.6
  - molar_volume: 10.83 cm**3/mol
  - atomization: 733000.0 J/mol
  - sound_velocity: 3480.0 m/s
  - sanderson: 1.42
  - atomic_weight: 92.90638
  - triple_bond_radius: 116.0 pm
  - melting_point: 2750.0 K
  - orbital_radii: (array(0.65088801626999993) * angstrom, array(0.80964119096999998) * angstrom, array(0.26988039698999999) * angstrom)
  - thermal_conductivity: 54.0 W/(m*K)
  - ionization_energies: [array(652100.0) * J/mol, array(1380000.0) * J/mol, array(2416000.0) * J/mol, array(3700000.0) * J/mol, array(4877000.0) * J/mol, array(9847000.0) * J/mol, array(12100000.0) * J/mol]
  - vaporization: 690000.0 J/mol
  - atomic_number: 41
  - rigidity_modulus: 38.0 GPa
  - symbol: Nb
  - covalent_radius: 164.0 pm
  - fusion: 26800.0 J/mol
  - pettifor: 0.82
  - bulk_modulus: 170.0 GPa
  - poisson_ratio: 0.4 dimensionless
  - electron_affinity: 86100.0 J/mol
  - name: Niobium
  - boiling_point: 5017.0 K
  - density: 8.57 g*cm**3
  - double_bond_radius: 125.0 pm
  - allred_rochow: 1.23
  - young_modulus: 105.0 GPa
  - thermal_expansion: 7.3e-06 1/K
  - atomic_radius: 145.0 pm
"""

Mo = Element(**{'single_bond_radius': array(138.0) * pm, 'pauling': 2.1600000000000001, 'molar_volume': array(9.3800000000000008) * cm**3/mol, 'bulk_modulus': array(230.0) * GPa, 'sound_velocity': array(6190.0) * m/s, 'sanderson': 1.1499999999999999, 'atomic_weight': 95.959999999999994, 'triple_bond_radius': array(113.0) * pm, 'melting_point': array(2896.0) * Kelvin, 'orbital_radii': (array(0.64559624377999991) * angstrom, array(0.79376587349999994) * angstrom, array(0.25929685200999997) * angstrom), 'thermal_conductivity': array(139.0) * Watt/(m*Kelvin), 'ionization_energies': [array(684300.0) * J/mol, array(1560000.0) * J/mol, array(2618000.0) * J/mol, array(4480000.0) * J/mol, array(5257000.0) * J/mol, array(6640800.0) * J/mol, array(12125000.0) * J/mol, array(13860000.0) * J/mol, array(15835000.0) * J/mol, array(17980000.0) * J/mol, array(20190000.0) * J/mol, array(22219000.0) * J/mol, array(26930000.0) * J/mol, array(29196000.0) * J/mol, array(52490000.0) * J/mol, array(55000000.0) * J/mol, array(61400000.0) * J/mol, array(67700000.0) * J/mol, array(74000000.0) * J/mol, array(80400000.0) * J/mol, array(87000000.0) * J/mol], 'vaporization': array(600000.0) * J/mol, 'atomic_number': 42, 'rigidity_modulus': array(20.0) * GPa, 'symbol': 'Mo', 'covalent_radius': array(154.0) * pm, 'fusion': array(36000.0) * J/mol, 'pettifor': 0.88, 'atomization': array(659000.0) * J/mol, 'poisson_ratio': array(0.31) * dimensionless, 'electron_affinity': array(71900.0) * J/mol, 'name': 'Molybdenum', 'boiling_point': array(4912.0) * Kelvin, 'density': array(10.279999999999999) * g*cm**3, 'double_bond_radius': array(121.0) * pm, 'allred_rochow': 1.3, 'young_modulus': array(329.0) * GPa, 'thermal_expansion': array(4.7999999999999998e-06) * 1/Kelvin, 'atomic_radius': array(145.0) * pm})
"""
Molybdenum elemental data.

  - single_bond_radius: 138.0 pm
  - pauling: 2.16
  - molar_volume: 9.38 cm**3/mol
  - atomization: 659000.0 J/mol
  - sound_velocity: 6190.0 m/s
  - sanderson: 1.15
  - atomic_weight: 95.96
  - triple_bond_radius: 113.0 pm
  - melting_point: 2896.0 K
  - orbital_radii: (array(0.64559624377999991) * angstrom, array(0.79376587349999994) * angstrom, array(0.25929685200999997) * angstrom)
  - thermal_conductivity: 139.0 W/(m*K)
  - ionization_energies: [array(684300.0) * J/mol, array(1560000.0) * J/mol, array(2618000.0) * J/mol, array(4480000.0) * J/mol, array(5257000.0) * J/mol, array(6640800.0) * J/mol, array(12125000.0) * J/mol, array(13860000.0) * J/mol, array(15835000.0) * J/mol, array(17980000.0) * J/mol, array(20190000.0) * J/mol, array(22219000.0) * J/mol, array(26930000.0) * J/mol, array(29196000.0) * J/mol, array(52490000.0) * J/mol, array(55000000.0) * J/mol, array(61400000.0) * J/mol, array(67700000.0) * J/mol, array(74000000.0) * J/mol, array(80400000.0) * J/mol, array(87000000.0) * J/mol]
  - vaporization: 600000.0 J/mol
  - atomic_number: 42
  - rigidity_modulus: 20.0 GPa
  - symbol: Mo
  - covalent_radius: 154.0 pm
  - fusion: 36000.0 J/mol
  - pettifor: 0.88
  - bulk_modulus: 230.0 GPa
  - poisson_ratio: 0.31 dimensionless
  - electron_affinity: 71900.0 J/mol
  - name: Molybdenum
  - boiling_point: 4912.0 K
  - density: 10.28 g*cm**3
  - double_bond_radius: 121.0 pm
  - allred_rochow: 1.3
  - young_modulus: 329.0 GPa
  - thermal_expansion: 4.8e-06 1/K
  - atomic_radius: 145.0 pm
"""

Tc = Element(**{'single_bond_radius': array(128.0) * pm, 'pauling': 1.8999999999999999, 'molar_volume': array(8.6300000000000008) * cm**3/mol, 'atomization': array(661000.0) * J/mol, 'atomic_weight': 98.0, 'triple_bond_radius': array(110.0) * pm, 'melting_point': array(2430.0) * Kelvin, 'orbital_radii': (array(0.61384560883999995) * angstrom, array(0.78847410100999993) * angstrom, array(0.24077564829499998) * angstrom), 'thermal_conductivity': array(51.0) * Watt/(m*Kelvin), 'ionization_energies': [array(702000.0) * J/mol, array(1470000.0) * J/mol, array(2850000.0) * J/mol], 'vaporization': array(550000.0) * J/mol, 'atomic_number': 43, 'symbol': 'Tc', 'covalent_radius': array(147.0) * pm, 'fusion': array(23000.0) * J/mol, 'pettifor': 0.93999999999999995, 'electron_affinity': array(53000.0) * J/mol, 'name': 'Technetium', 'boiling_point': array(4538.0) * Kelvin, 'density': array(11.5) * g*cm**3, 'double_bond_radius': array(120.0) * pm, 'allred_rochow': 1.3600000000000001, 'atomic_radius': array(135.0) * pm})
"""
Technetium elemental data.

  - single_bond_radius: 128.0 pm
  - pauling: 1.9
  - molar_volume: 8.63 cm**3/mol
  - atomization: 661000.0 J/mol
  - atomic_weight: 98.0
  - triple_bond_radius: 110.0 pm
  - melting_point: 2430.0 K
  - orbital_radii: (array(0.61384560883999995) * angstrom, array(0.78847410100999993) * angstrom, array(0.24077564829499998) * angstrom)
  - thermal_conductivity: 51.0 W/(m*K)
  - ionization_energies: [array(702000.0) * J/mol, array(1470000.0) * J/mol, array(2850000.0) * J/mol]
  - vaporization: 550000.0 J/mol
  - atomic_number: 43
  - symbol: Tc
  - covalent_radius: 147.0 pm
  - fusion: 23000.0 J/mol
  - pettifor: 0.94
  - electron_affinity: 53000.0 J/mol
  - name: Technetium
  - boiling_point: 4538.0 K
  - density: 11.5 g*cm**3
  - double_bond_radius: 120.0 pm
  - allred_rochow: 1.36
  - atomic_radius: 135.0 pm
"""

Ru = Element(**{'single_bond_radius': array(125.0) * pm, 'pauling': 2.2000000000000002, 'molar_volume': array(8.1699999999999999) * cm**3/mol, 'bulk_modulus': array(220.0) * GPa, 'sound_velocity': array(5970.0) * m/s, 'atomic_weight': 101.06999999999999, 'triple_bond_radius': array(103.0) * pm, 'melting_point': array(2607.0) * Kelvin, 'orbital_radii': (array(0.60590795010499998) * angstrom, array(0.77259878353999989) * angstrom, array(0.23812976205) * angstrom), 'thermal_conductivity': array(120.0) * Watt/(m*Kelvin), 'ionization_energies': [array(710200.0) * J/mol, array(1620000.0) * J/mol, array(2747000.0) * J/mol], 'vaporization': array(580000.0) * J/mol, 'atomic_number': 44, 'rigidity_modulus': array(173.0) * GPa, 'symbol': 'Ru', 'covalent_radius': array(146.0) * pm, 'fusion': array(25700.0) * J/mol, 'pettifor': 1.0, 'atomization': array(652000.0) * J/mol, 'poisson_ratio': array(0.29999999999999999) * dimensionless, 'electron_affinity': array(101300.0) * J/mol, 'name': 'Ruthenium', 'boiling_point': array(4423.0) * Kelvin, 'density': array(12.369999999999999) * g*cm**3, 'double_bond_radius': array(114.0) * pm, 'allred_rochow': 1.4199999999999999, 'young_modulus': array(447.0) * GPa, 'thermal_expansion': array(6.3999999999999997e-06) * 1/Kelvin, 'atomic_radius': array(130.0) * pm})
"""
Ruthenium elemental data.

  - single_bond_radius: 125.0 pm
  - pauling: 2.2
  - molar_volume: 8.17 cm**3/mol
  - atomization: 652000.0 J/mol
  - sound_velocity: 5970.0 m/s
  - atomic_weight: 101.07
  - triple_bond_radius: 103.0 pm
  - melting_point: 2607.0 K
  - orbital_radii: (array(0.60590795010499998) * angstrom, array(0.77259878353999989) * angstrom, array(0.23812976205) * angstrom)
  - thermal_conductivity: 120.0 W/(m*K)
  - ionization_energies: [array(710200.0) * J/mol, array(1620000.0) * J/mol, array(2747000.0) * J/mol]
  - vaporization: 580000.0 J/mol
  - atomic_number: 44
  - rigidity_modulus: 173.0 GPa
  - symbol: Ru
  - covalent_radius: 146.0 pm
  - fusion: 25700.0 J/mol
  - pettifor: 1.0
  - bulk_modulus: 220.0 GPa
  - poisson_ratio: 0.3 dimensionless
  - electron_affinity: 101300.0 J/mol
  - name: Ruthenium
  - boiling_point: 4423.0 K
  - density: 12.37 g*cm**3
  - double_bond_radius: 114.0 pm
  - allred_rochow: 1.42
  - young_modulus: 447.0 GPa
  - thermal_expansion: 6.4e-06 1/K
  - atomic_radius: 130.0 pm
"""

Rh = Element(**{'single_bond_radius': array(125.0) * pm, 'pauling': 2.2799999999999998, 'molar_volume': array(8.2799999999999994) * cm**3/mol, 'bulk_modulus': array(380.0) * GPa, 'sound_velocity': array(4700.0) * m/s, 'atomic_weight': 102.9055, 'triple_bond_radius': array(106.0) * pm, 'melting_point': array(2237.0) * Kelvin, 'orbital_radii': (array(0.58738674638999999) * angstrom, array(0.74613992108999994) * angstrom, array(0.22225444457999999) * angstrom), 'thermal_conductivity': array(150.0) * Watt/(m*Kelvin), 'ionization_energies': [array(719700.0) * J/mol, array(1740000.0) * J/mol, array(2997000.0) * J/mol], 'vaporization': array(495000.0) * J/mol, 'atomic_number': 45, 'rigidity_modulus': array(150.0) * GPa, 'symbol': 'Rh', 'covalent_radius': array(142.0) * pm, 'fusion': array(21700.0) * J/mol, 'pettifor': 1.0600000000000001, 'atomization': array(556000.0) * J/mol, 'poisson_ratio': array(0.26000000000000001) * dimensionless, 'electron_affinity': array(109700.0) * J/mol, 'name': 'Rhodium', 'boiling_point': array(3968.0) * Kelvin, 'density': array(12.449999999999999) * g*cm**3, 'double_bond_radius': array(110.0) * pm, 'allred_rochow': 1.45, 'young_modulus': array(275.0) * GPa, 'thermal_expansion': array(8.1999999999999994e-06) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Rhodium elemental data.

  - single_bond_radius: 125.0 pm
  - pauling: 2.28
  - molar_volume: 8.28 cm**3/mol
  - atomization: 556000.0 J/mol
  - sound_velocity: 4700.0 m/s
  - atomic_weight: 102.9055
  - triple_bond_radius: 106.0 pm
  - melting_point: 2237.0 K
  - orbital_radii: (array(0.58738674638999999) * angstrom, array(0.74613992108999994) * angstrom, array(0.22225444457999999) * angstrom)
  - thermal_conductivity: 150.0 W/(m*K)
  - ionization_energies: [array(719700.0) * J/mol, array(1740000.0) * J/mol, array(2997000.0) * J/mol]
  - vaporization: 495000.0 J/mol
  - atomic_number: 45
  - rigidity_modulus: 150.0 GPa
  - symbol: Rh
  - covalent_radius: 142.0 pm
  - fusion: 21700.0 J/mol
  - pettifor: 1.06
  - bulk_modulus: 380.0 GPa
  - poisson_ratio: 0.26 dimensionless
  - electron_affinity: 109700.0 J/mol
  - name: Rhodium
  - boiling_point: 3968.0 K
  - density: 12.45 g*cm**3
  - double_bond_radius: 110.0 pm
  - allred_rochow: 1.45
  - young_modulus: 275.0 GPa
  - thermal_expansion: 8.2e-06 1/K
  - atomic_radius: 135.0 pm
"""

Pd = Element(**{'single_bond_radius': array(120.0) * pm, 'pauling': 2.2000000000000002, 'molar_volume': array(8.5600000000000005) * cm**3/mol, 'bulk_modulus': array(180.0) * GPa, 'sound_velocity': array(3070.0) * m/s, 'atomic_weight': 106.42, 'triple_bond_radius': array(112.0) * pm, 'melting_point': array(1828.05) * Kelvin, 'orbital_radii': (array(0.57151142891999995) * angstrom, array(0.72497283112999999) * angstrom, array(0.21167089959999999) * angstrom), 'thermal_conductivity': array(72.0) * Watt/(m*Kelvin), 'ionization_energies': [array(804400.0) * J/mol, array(1870000.0) * J/mol, array(3177000.0) * J/mol], 'vaporization': array(380000.0) * J/mol, 'atomic_number': 46, 'rigidity_modulus': array(44.0) * GPa, 'symbol': 'Pd', 'covalent_radius': array(139.0) * pm, 'fusion': array(16700.0) * J/mol, 'pettifor': 1.1200000000000001, 'atomization': array(377000.0) * J/mol, 'poisson_ratio': array(0.39000000000000001) * dimensionless, 'van_der_waals_radius': array(163.0) * pm, 'electron_affinity': array(53700.0) * J/mol, 'name': 'Palladium', 'boiling_point': array(3236.0) * Kelvin, 'density': array(12.023) * g*cm**3, 'double_bond_radius': array(117.0) * pm, 'allred_rochow': 1.3500000000000001, 'young_modulus': array(121.0) * GPa, 'thermal_expansion': array(1.1800000000000001e-05) * 1/Kelvin, 'atomic_radius': array(140.0) * pm})
"""
Palladium elemental data.

  - single_bond_radius: 120.0 pm
  - pauling: 2.2
  - molar_volume: 8.56 cm**3/mol
  - atomization: 377000.0 J/mol
  - sound_velocity: 3070.0 m/s
  - atomic_weight: 106.42
  - triple_bond_radius: 112.0 pm
  - melting_point: 1828.05 K
  - orbital_radii: (array(0.57151142891999995) * angstrom, array(0.72497283112999999) * angstrom, array(0.21167089959999999) * angstrom)
  - thermal_conductivity: 72.0 W/(m*K)
  - ionization_energies: [array(804400.0) * J/mol, array(1870000.0) * J/mol, array(3177000.0) * J/mol]
  - vaporization: 380000.0 J/mol
  - atomic_number: 46
  - rigidity_modulus: 44.0 GPa
  - symbol: Pd
  - covalent_radius: 139.0 pm
  - fusion: 16700.0 J/mol
  - pettifor: 1.12
  - bulk_modulus: 180.0 GPa
  - poisson_ratio: 0.39 dimensionless
  - van_der_waals_radius: 163.0 pm
  - electron_affinity: 53700.0 J/mol
  - name: Palladium
  - boiling_point: 3236.0 K
  - density: 12.023 g*cm**3
  - double_bond_radius: 117.0 pm
  - allred_rochow: 1.35
  - young_modulus: 121.0 GPa
  - thermal_expansion: 1.18e-05 1/K
  - atomic_radius: 140.0 pm
"""

Ag = Element(**{'mulliken_jaffe': 1.47, 'single_bond_radius': array(128.0) * pm, 'pauling': 1.9299999999999999, 'molar_volume': array(10.27) * cm**3/mol, 'bulk_modulus': array(100.0) * GPa, 'sound_velocity': array(2600.0) * m/s, 'sanderson': 1.8300000000000001, 'atomic_weight': 107.8682, 'triple_bond_radius': array(137.0) * pm, 'melting_point': array(1234.9300000000001) * Kelvin, 'orbital_radii': (array(0.55299022520499996) * angstrom, array(0.70380574116999994) * angstrom, array(0.203733240865) * angstrom), 'thermal_conductivity': array(430.0) * Watt/(m*Kelvin), 'ionization_energies': [array(731000.0) * J/mol, array(2070000.0) * J/mol, array(3361000.0) * J/mol], 'vaporization': array(255000.0) * J/mol, 'atomic_number': 47, 'rigidity_modulus': array(30.0) * GPa, 'symbol': 'Ag', 'covalent_radius': array(145.0) * pm, 'fusion': array(11300.0) * J/mol, 'pettifor': 1.1799999999999999, 'atomization': array(285000.0) * J/mol, 'poisson_ratio': array(0.37) * dimensionless, 'van_der_waals_radius': array(172.0) * pm, 'electron_affinity': array(125600.0) * J/mol, 'name': 'Silver', 'boiling_point': array(2435.0) * Kelvin, 'density': array(10.49) * g*cm**3, 'double_bond_radius': array(139.0) * pm, 'allred_rochow': 1.4199999999999999, 'young_modulus': array(83.0) * GPa, 'thermal_expansion': array(1.8899999999999999e-05) * 1/Kelvin, 'atomic_radius': array(160.0) * pm})
"""
Silver elemental data.

  - mulliken_jaffe: 1.47
  - single_bond_radius: 128.0 pm
  - pauling: 1.93
  - molar_volume: 10.27 cm**3/mol
  - atomization: 285000.0 J/mol
  - sound_velocity: 2600.0 m/s
  - sanderson: 1.83
  - atomic_weight: 107.8682
  - triple_bond_radius: 137.0 pm
  - melting_point: 1234.93 K
  - orbital_radii: (array(0.55299022520499996) * angstrom, array(0.70380574116999994) * angstrom, array(0.203733240865) * angstrom)
  - thermal_conductivity: 430.0 W/(m*K)
  - ionization_energies: [array(731000.0) * J/mol, array(2070000.0) * J/mol, array(3361000.0) * J/mol]
  - vaporization: 255000.0 J/mol
  - atomic_number: 47
  - rigidity_modulus: 30.0 GPa
  - symbol: Ag
  - covalent_radius: 145.0 pm
  - fusion: 11300.0 J/mol
  - pettifor: 1.18
  - bulk_modulus: 100.0 GPa
  - poisson_ratio: 0.37 dimensionless
  - van_der_waals_radius: 172.0 pm
  - electron_affinity: 125600.0 J/mol
  - name: Silver
  - boiling_point: 2435.0 K
  - density: 10.49 g*cm**3
  - double_bond_radius: 139.0 pm
  - allred_rochow: 1.42
  - young_modulus: 83.0 GPa
  - thermal_expansion: 1.89e-05 1/K
  - atomic_radius: 160.0 pm
"""

Cd = Element(**{'mulliken_jaffe': 1.53, 'single_bond_radius': array(136.0) * pm, 'pauling': 1.6899999999999999, 'molar_volume': array(13.0) * cm**3/mol, 'bulk_modulus': array(42.0) * GPa, 'sound_velocity': array(2310.0) * m/s, 'sanderson': 1.98, 'atomic_weight': 112.411, 'melting_point': array(594.22000000000003) * Kelvin, 'orbital_radii': (array(0.521239590265) * angstrom, array(0.65088801626999993) * angstrom, array(0.19579558212999998) * angstrom), 'thermal_conductivity': array(97.0) * Watt/(m*Kelvin), 'ionization_energies': [array(867800.0) * J/mol, array(1631400.0) * J/mol, array(3616000.0) * J/mol], 'vaporization': array(100000.0) * J/mol, 'atomic_number': 48, 'rigidity_modulus': array(19.0) * GPa, 'symbol': 'Cd', 'covalent_radius': array(144.0) * pm, 'fusion': array(6300.0) * J/mol, 'pettifor': 1.3600000000000001, 'atomization': array(112000.0) * J/mol, 'poisson_ratio': array(0.29999999999999999) * dimensionless, 'van_der_waals_radius': array(158.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Cadmium', 'boiling_point': array(1040.0) * Kelvin, 'density': array(8.6500000000000004) * g*cm**3, 'double_bond_radius': array(144.0) * pm, 'allred_rochow': 1.46, 'young_modulus': array(50.0) * GPa, 'thermal_expansion': array(3.0799999999999996e-05) * 1/Kelvin, 'atomic_radius': array(155.0) * pm})
"""
Cadmium elemental data.

  - mulliken_jaffe: 1.53
  - single_bond_radius: 136.0 pm
  - pauling: 1.69
  - molar_volume: 13.0 cm**3/mol
  - atomization: 112000.0 J/mol
  - sound_velocity: 2310.0 m/s
  - sanderson: 1.98
  - atomic_weight: 112.411
  - melting_point: 594.22 K
  - orbital_radii: (array(0.521239590265) * angstrom, array(0.65088801626999993) * angstrom, array(0.19579558212999998) * angstrom)
  - thermal_conductivity: 97.0 W/(m*K)
  - ionization_energies: [array(867800.0) * J/mol, array(1631400.0) * J/mol, array(3616000.0) * J/mol]
  - vaporization: 100000.0 J/mol
  - atomic_number: 48
  - rigidity_modulus: 19.0 GPa
  - symbol: Cd
  - covalent_radius: 144.0 pm
  - fusion: 6300.0 J/mol
  - pettifor: 1.36
  - bulk_modulus: 42.0 GPa
  - poisson_ratio: 0.3 dimensionless
  - van_der_waals_radius: 158.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Cadmium
  - boiling_point: 1040.0 K
  - density: 8.65 g*cm**3
  - double_bond_radius: 144.0 pm
  - allred_rochow: 1.46
  - young_modulus: 50.0 GPa
  - thermal_expansion: 3.08e-05 1/K
  - atomic_radius: 155.0 pm
"""

In = Element(**{'mulliken_jaffe': 1.76, 'single_bond_radius': array(142.0) * pm, 'pauling': 1.78, 'molar_volume': array(15.76) * cm**3/mol, 'atomization': array(243000.0) * J/mol, 'sound_velocity': array(1215.0) * m/s, 'sanderson': 2.1400000000000001, 'atomic_weight': 114.818, 'triple_bond_radius': array(146.0) * pm, 'melting_point': array(429.75) * Kelvin, 'allen': 1.6559999999999999, 'orbital_radii': (array(0.49742661405999994) * angstrom, array(0.58738674638999999) * angstrom, array(0.19050380963999997) * angstrom), 'thermal_conductivity': array(82.0) * Watt/(m*Kelvin), 'ionization_energies': [array(558300.0) * J/mol, array(1820700.0) * J/mol, array(2704000.0) * J/mol, array(5210000.0) * J/mol], 'vaporization': array(230000.0) * J/mol, 'atomic_number': 49, 'symbol': 'In', 'covalent_radius': array(142.0) * pm, 'fusion': array(3260.0) * J/mol, 'pettifor': 1.6000000000000001, 'van_der_waals_radius': array(193.0) * pm, 'electron_affinity': array(28900.0) * J/mol, 'name': 'Indium', 'boiling_point': array(2345.0) * Kelvin, 'density': array(7.3099999999999996) * g*cm**3, 'double_bond_radius': array(136.0) * pm, 'allred_rochow': 1.49, 'young_modulus': array(11.0) * GPa, 'thermal_expansion': array(3.2100000000000001e-05) * 1/Kelvin, 'atomic_radius': array(155.0) * pm})
"""
Indium elemental data.

  - mulliken_jaffe: 1.76
  - single_bond_radius: 142.0 pm
  - pauling: 1.78
  - molar_volume: 15.76 cm**3/mol
  - atomization: 243000.0 J/mol
  - sound_velocity: 1215.0 m/s
  - sanderson: 2.14
  - atomic_weight: 114.818
  - triple_bond_radius: 146.0 pm
  - melting_point: 429.75 K
  - allen: 1.656
  - orbital_radii: (array(0.49742661405999994) * angstrom, array(0.58738674638999999) * angstrom, array(0.19050380963999997) * angstrom)
  - thermal_conductivity: 82.0 W/(m*K)
  - ionization_energies: [array(558300.0) * J/mol, array(1820700.0) * J/mol, array(2704000.0) * J/mol, array(5210000.0) * J/mol]
  - vaporization: 230000.0 J/mol
  - atomic_number: 49
  - symbol: In
  - covalent_radius: 142.0 pm
  - fusion: 3260.0 J/mol
  - pettifor: 1.6
  - van_der_waals_radius: 193.0 pm
  - electron_affinity: 28900.0 J/mol
  - name: Indium
  - boiling_point: 2345.0 K
  - density: 7.31 g*cm**3
  - double_bond_radius: 136.0 pm
  - allred_rochow: 1.49
  - young_modulus: 11.0 GPa
  - thermal_expansion: 3.21e-05 1/K
  - atomic_radius: 155.0 pm
"""

Sn = Element(**{'mulliken_jaffe': 2.21, 'single_bond_radius': array(140.0) * pm, 'pauling': 1.96, 'molar_volume': array(16.289999999999999) * cm**3/mol, 'bulk_modulus': array(58.0) * GPa, 'sound_velocity': array(2500.0) * m/s, 'sanderson': 1.49, 'atomic_weight': 118.70999999999999, 'triple_bond_radius': array(132.0) * pm, 'melting_point': array(505.07999999999998) * Kelvin, 'allen': 1.8240000000000001, 'orbital_radii': (array(0.46567597911999997) * angstrom, array(0.52917724899999996) * angstrom, array(0.18256615090499997) * angstrom), 'thermal_conductivity': array(67.0) * Watt/(m*Kelvin), 'ionization_energies': [array(708600.0) * J/mol, array(1411800.0) * J/mol, array(2943000.0) * J/mol, array(3930300.0) * J/mol, array(7456000.0) * J/mol], 'vaporization': array(290000.0) * J/mol, 'atomic_number': 50, 'rigidity_modulus': array(18.0) * GPa, 'symbol': 'Sn', 'covalent_radius': array(139.0) * pm, 'fusion': array(7000.0) * J/mol, 'pettifor': 1.8400000000000001, 'atomization': array(302000.0) * J/mol, 'poisson_ratio': array(0.35999999999999999) * dimensionless, 'van_der_waals_radius': array(217.0) * pm, 'electron_affinity': array(107300.0) * J/mol, 'name': 'Tin', 'boiling_point': array(2875.0) * Kelvin, 'density': array(7.3099999999999996) * g*cm**3, 'double_bond_radius': array(130.0) * pm, 'allred_rochow': 1.72, 'young_modulus': array(50.0) * GPa, 'thermal_expansion': array(2.1999999999999999e-05) * 1/Kelvin, 'atomic_radius': array(145.0) * pm})
"""
Tin elemental data.

  - mulliken_jaffe: 2.21
  - single_bond_radius: 140.0 pm
  - pauling: 1.96
  - molar_volume: 16.29 cm**3/mol
  - atomization: 302000.0 J/mol
  - sound_velocity: 2500.0 m/s
  - sanderson: 1.49
  - atomic_weight: 118.71
  - triple_bond_radius: 132.0 pm
  - melting_point: 505.08 K
  - allen: 1.824
  - orbital_radii: (array(0.46567597911999997) * angstrom, array(0.52917724899999996) * angstrom, array(0.18256615090499997) * angstrom)
  - thermal_conductivity: 67.0 W/(m*K)
  - ionization_energies: [array(708600.0) * J/mol, array(1411800.0) * J/mol, array(2943000.0) * J/mol, array(3930300.0) * J/mol, array(7456000.0) * J/mol]
  - vaporization: 290000.0 J/mol
  - atomic_number: 50
  - rigidity_modulus: 18.0 GPa
  - symbol: Sn
  - covalent_radius: 139.0 pm
  - fusion: 7000.0 J/mol
  - pettifor: 1.84
  - bulk_modulus: 58.0 GPa
  - poisson_ratio: 0.36 dimensionless
  - van_der_waals_radius: 217.0 pm
  - electron_affinity: 107300.0 J/mol
  - name: Tin
  - boiling_point: 2875.0 K
  - density: 7.31 g*cm**3
  - double_bond_radius: 130.0 pm
  - allred_rochow: 1.72
  - young_modulus: 50.0 GPa
  - thermal_expansion: 2.2e-05 1/K
  - atomic_radius: 145.0 pm
"""

Sb = Element(**{'mulliken_jaffe': 2.1200000000000001, 'single_bond_radius': array(140.0) * pm, 'pauling': 2.0499999999999998, 'molar_volume': array(18.190000000000001) * cm**3/mol, 'bulk_modulus': array(42.0) * GPa, 'sound_velocity': array(3420.0) * m/s, 'sanderson': 2.46, 'atomic_weight': 121.76000000000001, 'triple_bond_radius': array(127.0) * pm, 'melting_point': array(903.77999999999997) * Kelvin, 'allen': 1.984, 'orbital_radii': (array(0.43921711666999996) * angstrom, array(0.49478072781499999) * angstrom, array(0.17727437841499999) * angstrom), 'thermal_conductivity': array(24.0) * Watt/(m*Kelvin), 'ionization_energies': [array(834000.0) * J/mol, array(1594900.0) * J/mol, array(2440000.0) * J/mol, array(4260000.0) * J/mol, array(5400000.0) * J/mol, array(10400000.0) * J/mol], 'vaporization': array(68000.0) * J/mol, 'atomic_number': 51, 'rigidity_modulus': array(20.0) * GPa, 'symbol': 'Sb', 'covalent_radius': array(139.0) * pm, 'fusion': array(19700.0) * J/mol, 'pettifor': 2.0800000000000001, 'atomization': array(262000.0) * J/mol, 'electron_affinity': array(103200.0) * J/mol, 'name': 'Antimony', 'boiling_point': array(1860.0) * Kelvin, 'density': array(6.6970000000000001) * g*cm**3, 'double_bond_radius': array(133.0) * pm, 'allred_rochow': 1.8200000000000001, 'young_modulus': array(55.0) * GPa, 'thermal_expansion': array(1.1e-05) * 1/Kelvin, 'atomic_radius': array(145.0) * pm})
"""
Antimony elemental data.

  - mulliken_jaffe: 2.12
  - single_bond_radius: 140.0 pm
  - pauling: 2.05
  - molar_volume: 18.19 cm**3/mol
  - atomization: 262000.0 J/mol
  - sound_velocity: 3420.0 m/s
  - sanderson: 2.46
  - atomic_weight: 121.76
  - triple_bond_radius: 127.0 pm
  - melting_point: 903.78 K
  - allen: 1.984
  - orbital_radii: (array(0.43921711666999996) * angstrom, array(0.49478072781499999) * angstrom, array(0.17727437841499999) * angstrom)
  - thermal_conductivity: 24.0 W/(m*K)
  - ionization_energies: [array(834000.0) * J/mol, array(1594900.0) * J/mol, array(2440000.0) * J/mol, array(4260000.0) * J/mol, array(5400000.0) * J/mol, array(10400000.0) * J/mol]
  - vaporization: 68000.0 J/mol
  - atomic_number: 51
  - rigidity_modulus: 20.0 GPa
  - symbol: Sb
  - covalent_radius: 139.0 pm
  - fusion: 19700.0 J/mol
  - pettifor: 2.08
  - bulk_modulus: 42.0 GPa
  - electron_affinity: 103200.0 J/mol
  - name: Antimony
  - boiling_point: 1860.0 K
  - density: 6.697 g*cm**3
  - double_bond_radius: 133.0 pm
  - allred_rochow: 1.82
  - young_modulus: 55.0 GPa
  - thermal_expansion: 1.1e-05 1/K
  - atomic_radius: 145.0 pm
"""

Te = Element(**{'mulliken_jaffe': 2.4100000000000001, 'single_bond_radius': array(136.0) * pm, 'pauling': 2.1000000000000001, 'molar_volume': array(20.460000000000001) * cm**3/mol, 'bulk_modulus': array(65.0) * GPa, 'sound_velocity': array(2610.0) * m/s, 'sanderson': 2.6200000000000001, 'atomic_weight': 127.59999999999999, 'triple_bond_radius': array(121.0) * pm, 'melting_point': array(722.65999999999997) * Kelvin, 'allen': 2.1579999999999999, 'orbital_radii': (array(0.41805002670999997) * angstrom, array(0.46567597911999997) * angstrom, array(0.171982605925) * angstrom), 'thermal_conductivity': array(3.0) * Watt/(m*Kelvin), 'ionization_energies': [array(869300.0) * J/mol, array(1790000.0) * J/mol, array(2698000.0) * J/mol, array(3610000.0) * J/mol, array(5668000.0) * J/mol, array(6820000.0) * J/mol, array(13200000.0) * J/mol], 'vaporization': array(48000.0) * J/mol, 'atomic_number': 52, 'rigidity_modulus': array(16.0) * GPa, 'symbol': 'Te', 'covalent_radius': array(138.0) * pm, 'fusion': array(17500.0) * J/mol, 'pettifor': 2.3199999999999998, 'atomization': array(197000.0) * J/mol, 'van_der_waals_radius': array(206.0) * pm, 'electron_affinity': array(190200.0) * J/mol, 'name': 'Tellurium', 'boiling_point': array(1261.0) * Kelvin, 'density': array(6.2400000000000002) * g*cm**3, 'double_bond_radius': array(128.0) * pm, 'allred_rochow': 2.0099999999999998, 'young_modulus': array(43.0) * GPa, 'atomic_radius': array(140.0) * pm})
"""
Tellurium elemental data.

  - mulliken_jaffe: 2.41
  - single_bond_radius: 136.0 pm
  - pauling: 2.1
  - molar_volume: 20.46 cm**3/mol
  - atomization: 197000.0 J/mol
  - sound_velocity: 2610.0 m/s
  - sanderson: 2.62
  - atomic_weight: 127.6
  - triple_bond_radius: 121.0 pm
  - melting_point: 722.66 K
  - allen: 2.158
  - orbital_radii: (array(0.41805002670999997) * angstrom, array(0.46567597911999997) * angstrom, array(0.171982605925) * angstrom)
  - thermal_conductivity: 3.0 W/(m*K)
  - ionization_energies: [array(869300.0) * J/mol, array(1790000.0) * J/mol, array(2698000.0) * J/mol, array(3610000.0) * J/mol, array(5668000.0) * J/mol, array(6820000.0) * J/mol, array(13200000.0) * J/mol]
  - vaporization: 48000.0 J/mol
  - atomic_number: 52
  - rigidity_modulus: 16.0 GPa
  - symbol: Te
  - covalent_radius: 138.0 pm
  - fusion: 17500.0 J/mol
  - pettifor: 2.32
  - bulk_modulus: 65.0 GPa
  - van_der_waals_radius: 206.0 pm
  - electron_affinity: 190200.0 J/mol
  - name: Tellurium
  - boiling_point: 1261.0 K
  - density: 6.24 g*cm**3
  - double_bond_radius: 128.0 pm
  - allred_rochow: 2.01
  - young_modulus: 43.0 GPa
  - atomic_radius: 140.0 pm
"""

I = Element(**{'mulliken_jaffe': 2.7400000000000002, 'single_bond_radius': array(133.0) * pm, 'pauling': 2.6600000000000001, 'molar_volume': array(25.719999999999999) * cm**3/mol, 'bulk_modulus': array(7.7000000000000002) * GPa, 'sanderson': 2.7799999999999998, 'atomic_weight': 126.90447, 'critical_temperature': array(819.0) * Kelvin, 'triple_bond_radius': array(125.0) * pm, 'melting_point': array(386.85000000000002) * Kelvin, 'allen': 2.359, 'orbital_radii': (array(0.39952882299499998) * angstrom, array(0.43921711666999996) * angstrom, array(0.16669083343499999) * angstrom), 'thermal_conductivity': array(0.44900000000000001) * Watt/(m*Kelvin), 'ionization_energies': [array(1008400.0) * J/mol, array(1845900.0) * J/mol, array(3180000.0) * J/mol], 'vaporization': array(20900.0) * J/mol, 'atomic_number': 53, 'symbol': 'I', 'covalent_radius': array(139.0) * pm, 'fusion': array(7760.0) * J/mol, 'pettifor': 2.5600000000000001, 'atomization': array(107000.0) * J/mol, 'van_der_waals_radius': array(198.0) * pm, 'electron_affinity': array(295200.0) * J/mol, 'name': 'Iodine', 'boiling_point': array(457.39999999999998) * Kelvin, 'density': array(4.9400000000000004) * g*cm**3, 'double_bond_radius': array(129.0) * pm, 'allred_rochow': 2.21, 'atomic_radius': array(140.0) * pm})
"""
Iodine elemental data.

  - mulliken_jaffe: 2.74
  - single_bond_radius: 133.0 pm
  - pauling: 2.66
  - molar_volume: 25.72 cm**3/mol
  - atomization: 107000.0 J/mol
  - sanderson: 2.78
  - atomic_weight: 126.90447
  - critical_temperature: 819.0 K
  - triple_bond_radius: 125.0 pm
  - melting_point: 386.85 K
  - allen: 2.359
  - orbital_radii: (array(0.39952882299499998) * angstrom, array(0.43921711666999996) * angstrom, array(0.16669083343499999) * angstrom)
  - thermal_conductivity: 0.449 W/(m*K)
  - ionization_energies: [array(1008400.0) * J/mol, array(1845900.0) * J/mol, array(3180000.0) * J/mol]
  - vaporization: 20900.0 J/mol
  - atomic_number: 53
  - symbol: I
  - covalent_radius: 139.0 pm
  - fusion: 7760.0 J/mol
  - pettifor: 2.56
  - bulk_modulus: 7.7 GPa
  - van_der_waals_radius: 198.0 pm
  - electron_affinity: 295200.0 J/mol
  - name: Iodine
  - boiling_point: 457.4 K
  - density: 4.94 g*cm**3
  - double_bond_radius: 129.0 pm
  - allred_rochow: 2.21
  - atomic_radius: 140.0 pm
"""

Xe = Element(**{'mulliken_jaffe': 2.73, 'single_bond_radius': array(131.0) * pm, 'pauling': 2.6000000000000001, 'molar_volume': array(35.920000000000002) * cm**3/mol, 'atomization': array(0.0) * J/mol, 'sound_velocity': array(1090.0) * m/s, 'sanderson': 2.3399999999999999, 'atomic_weight': 131.29300000000001, 'critical_temperature': array(289.69999999999999) * Kelvin, 'triple_bond_radius': array(122.0) * pm, 'melting_point': array(161.40000000000001) * Kelvin, 'allen': 2.5819999999999999, 'orbital_radii': (array(0.39688293674999997) * angstrom, array(0.42863357168999999) * angstrom, array(0.16139906094499998) * angstrom), 'thermal_conductivity': array(0.0056499999999999996) * Watt/(m*Kelvin), 'ionization_energies': [array(1170400.0) * J/mol, array(2046400.0) * J/mol, array(3099400.0) * J/mol], 'vaporization': array(12640.0) * J/mol, 'atomic_number': 54, 'symbol': 'Xe', 'covalent_radius': array(140.0) * pm, 'fusion': array(2300.0) * J/mol, 'van_der_waals_radius': array(216.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Xenon', 'boiling_point': array(165.09999999999999) * Kelvin, 'double_bond_radius': array(135.0) * pm, 'allred_rochow': 2.3999999999999999})
"""
Xenon elemental data.

  - mulliken_jaffe: 2.73
  - single_bond_radius: 131.0 pm
  - pauling: 2.6
  - molar_volume: 35.92 cm**3/mol
  - atomization: 0.0 J/mol
  - sound_velocity: 1090.0 m/s
  - sanderson: 2.34
  - atomic_weight: 131.293
  - critical_temperature: 289.7 K
  - triple_bond_radius: 122.0 pm
  - melting_point: 161.4 K
  - allen: 2.582
  - orbital_radii: (array(0.39688293674999997) * angstrom, array(0.42863357168999999) * angstrom, array(0.16139906094499998) * angstrom)
  - thermal_conductivity: 0.00565 W/(m*K)
  - ionization_energies: [array(1170400.0) * J/mol, array(2046400.0) * J/mol, array(3099400.0) * J/mol]
  - vaporization: 12640.0 J/mol
  - atomic_number: 54
  - symbol: Xe
  - covalent_radius: 140.0 pm
  - fusion: 2300.0 J/mol
  - van_der_waals_radius: 216.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Xenon
  - boiling_point: 165.1 K
  - double_bond_radius: 135.0 pm
  - allred_rochow: 2.4
"""

Cs = Element(**{'mulliken_jaffe': 0.62, 'single_bond_radius': array(232.0) * pm, 'pauling': 0.79000000000000004, 'molar_volume': array(70.939999999999998) * cm**3/mol, 'bulk_modulus': array(1.6000000000000001) * GPa, 'sanderson': 0.22, 'atomic_weight': 132.9054519, 'critical_temperature': array(1938.0) * Kelvin, 'melting_point': array(301.58999999999997) * Kelvin, 'orbital_radii': (array(0.90489309578999988) * angstrom, array(1.3758608474) * angstrom), 'thermal_conductivity': array(36.0) * Watt/(m*Kelvin), 'ionization_energies': [array(375700.0) * J/mol, array(2234300.0) * J/mol, array(3400000.0) * J/mol], 'vaporization': array(65000.0) * J/mol, 'atomic_number': 55, 'symbol': 'Cs', 'covalent_radius': array(244.0) * pm, 'fusion': array(2090.0) * J/mol, 'pettifor': 0.25, 'atomization': array(76000.0) * J/mol, 'electron_affinity': array(45500.0) * J/mol, 'name': 'Caesium', 'boiling_point': array(944.0) * Kelvin, 'density': array(1.879) * g*cm**3, 'double_bond_radius': array(209.0) * pm, 'allred_rochow': 0.85999999999999999, 'young_modulus': array(1.7) * GPa, 'atomic_radius': array(260.0) * pm})
"""
Caesium elemental data.

  - mulliken_jaffe: 0.62
  - single_bond_radius: 232.0 pm
  - pauling: 0.79
  - molar_volume: 70.94 cm**3/mol
  - atomization: 76000.0 J/mol
  - sanderson: 0.22
  - atomic_weight: 132.9054519
  - critical_temperature: 1938.0 K
  - melting_point: 301.59 K
  - orbital_radii: (array(0.90489309578999988) * angstrom, array(1.3758608474) * angstrom)
  - thermal_conductivity: 36.0 W/(m*K)
  - ionization_energies: [array(375700.0) * J/mol, array(2234300.0) * J/mol, array(3400000.0) * J/mol]
  - vaporization: 65000.0 J/mol
  - atomic_number: 55
  - symbol: Cs
  - covalent_radius: 244.0 pm
  - fusion: 2090.0 J/mol
  - pettifor: 0.25
  - bulk_modulus: 1.6 GPa
  - electron_affinity: 45500.0 J/mol
  - name: Caesium
  - boiling_point: 944.0 K
  - density: 1.879 g*cm**3
  - double_bond_radius: 209.0 pm
  - allred_rochow: 0.86
  - young_modulus: 1.7 GPa
  - atomic_radius: 260.0 pm
"""

Ba = Element(**{'mulliken_jaffe': 0.88, 'single_bond_radius': array(196.0) * pm, 'pauling': 0.89000000000000001, 'molar_volume': array(38.159999999999997) * cm**3/mol, 'bulk_modulus': array(9.5999999999999996) * GPa, 'sound_velocity': array(1620.0) * m/s, 'sanderson': 0.68000000000000005, 'atomic_weight': 137.327, 'triple_bond_radius': array(149.0) * pm, 'melting_point': array(1000.0) * Kelvin, 'orbital_radii': (array(0.8017035322349999) * angstrom, array(0.9985574688629999) * angstrom, array(0.49742661405999994) * angstrom), 'thermal_conductivity': array(18.0) * Watt/(m*Kelvin), 'ionization_energies': [array(502900.0) * J/mol, array(965200.0) * J/mol, array(3600000.0) * J/mol], 'vaporization': array(140000.0) * J/mol, 'atomic_number': 56, 'rigidity_modulus': array(4.9000000000000004) * GPa, 'symbol': 'Ba', 'covalent_radius': array(215.0) * pm, 'fusion': array(8000.0) * J/mol, 'pettifor': 0.5, 'atomization': array(182000.0) * J/mol, 'electron_affinity': array(13950.0) * J/mol, 'name': 'Barium', 'boiling_point': array(2143.0) * Kelvin, 'density': array(3.5099999999999998) * g*cm**3, 'double_bond_radius': array(161.0) * pm, 'allred_rochow': 0.96999999999999997, 'young_modulus': array(13.0) * GPa, 'thermal_expansion': array(2.0599999999999999e-05) * 1/Kelvin, 'atomic_radius': array(215.0) * pm})
"""
Barium elemental data.

  - mulliken_jaffe: 0.88
  - single_bond_radius: 196.0 pm
  - pauling: 0.89
  - molar_volume: 38.16 cm**3/mol
  - atomization: 182000.0 J/mol
  - sound_velocity: 1620.0 m/s
  - sanderson: 0.68
  - atomic_weight: 137.327
  - triple_bond_radius: 149.0 pm
  - melting_point: 1000.0 K
  - orbital_radii: (array(0.8017035322349999) * angstrom, array(0.9985574688629999) * angstrom, array(0.49742661405999994) * angstrom)
  - thermal_conductivity: 18.0 W/(m*K)
  - ionization_energies: [array(502900.0) * J/mol, array(965200.0) * J/mol, array(3600000.0) * J/mol]
  - vaporization: 140000.0 J/mol
  - atomic_number: 56
  - rigidity_modulus: 4.9 GPa
  - symbol: Ba
  - covalent_radius: 215.0 pm
  - fusion: 8000.0 J/mol
  - pettifor: 0.5
  - bulk_modulus: 9.6 GPa
  - electron_affinity: 13950.0 J/mol
  - name: Barium
  - boiling_point: 2143.0 K
  - density: 3.51 g*cm**3
  - double_bond_radius: 161.0 pm
  - allred_rochow: 0.97
  - young_modulus: 13.0 GPa
  - thermal_expansion: 2.06e-05 1/K
  - atomic_radius: 215.0 pm
"""

La = Element(**{'single_bond_radius': array(180.0) * pm, 'pauling': 1.1000000000000001, 'molar_volume': array(22.390000000000001) * cm**3/mol, 'bulk_modulus': array(28.0) * GPa, 'sound_velocity': array(2475.0) * m/s, 'atomic_weight': 138.90547000000001, 'triple_bond_radius': array(139.0) * pm, 'melting_point': array(1193.0) * Kelvin, 'orbital_radii': (array(0.72761871737499995) * angstrom, array(0.90224720954499993) * angstrom, array(0.46250091562599999) * angstrom), 'thermal_conductivity': array(13.0) * Watt/(m*Kelvin), 'ionization_energies': [array(538100.0) * J/mol, array(1067000.0) * J/mol, array(1850300.0) * J/mol, array(4819000.0) * J/mol, array(5940000.0) * J/mol], 'vaporization': array(400000.0) * J/mol, 'atomic_number': 57, 'rigidity_modulus': array(14.0) * GPa, 'symbol': 'La', 'covalent_radius': array(207.0) * pm, 'fusion': array(6200.0) * J/mol, 'pettifor': 0.748, 'atomization': array(431000.0) * J/mol, 'poisson_ratio': array(0.28000000000000003) * dimensionless, 'electron_affinity': array(48000.0) * J/mol, 'name': 'Lanthanum', 'boiling_point': array(3743.0) * Kelvin, 'density': array(6.1459999999999999) * g*cm**3, 'double_bond_radius': array(139.0) * pm, 'allred_rochow': 1.0800000000000001, 'young_modulus': array(37.0) * GPa, 'thermal_expansion': array(1.2099999999999999e-05) * 1/Kelvin, 'atomic_radius': array(195.0) * pm})
"""
Lanthanum elemental data.

  - single_bond_radius: 180.0 pm
  - pauling: 1.1
  - molar_volume: 22.39 cm**3/mol
  - atomization: 431000.0 J/mol
  - sound_velocity: 2475.0 m/s
  - atomic_weight: 138.90547
  - triple_bond_radius: 139.0 pm
  - melting_point: 1193.0 K
  - orbital_radii: (array(0.72761871737499995) * angstrom, array(0.90224720954499993) * angstrom, array(0.46250091562599999) * angstrom)
  - thermal_conductivity: 13.0 W/(m*K)
  - ionization_energies: [array(538100.0) * J/mol, array(1067000.0) * J/mol, array(1850300.0) * J/mol, array(4819000.0) * J/mol, array(5940000.0) * J/mol]
  - vaporization: 400000.0 J/mol
  - atomic_number: 57
  - rigidity_modulus: 14.0 GPa
  - symbol: La
  - covalent_radius: 207.0 pm
  - fusion: 6200.0 J/mol
  - pettifor: 0.748
  - bulk_modulus: 28.0 GPa
  - poisson_ratio: 0.28 dimensionless
  - electron_affinity: 48000.0 J/mol
  - name: Lanthanum
  - boiling_point: 3743.0 K
  - density: 6.146 g*cm**3
  - double_bond_radius: 139.0 pm
  - allred_rochow: 1.08
  - young_modulus: 37.0 GPa
  - thermal_expansion: 1.21e-05 1/K
  - atomic_radius: 195.0 pm
"""

Ce = Element(**{'single_bond_radius': array(163.0) * pm, 'pauling': 1.1200000000000001, 'molar_volume': array(20.690000000000001) * cm**3/mol, 'bulk_modulus': array(22.0) * GPa, 'sound_velocity': array(2100.0) * m/s, 'atomic_weight': 140.11600000000001, 'triple_bond_radius': array(131.0) * pm, 'melting_point': array(1068.0) * Kelvin, 'thermal_conductivity': array(11.0) * Watt/(m*Kelvin), 'ionization_energies': [array(534400.0) * J/mol, array(1050000.0) * J/mol, array(1949000.0) * J/mol, array(3547000.0) * J/mol, array(6325000.0) * J/mol, array(7490000.0) * J/mol], 'vaporization': array(350000.0) * J/mol, 'atomic_number': 58, 'rigidity_modulus': array(14.0) * GPa, 'symbol': 'Ce', 'covalent_radius': array(204.0) * pm, 'fusion': array(5500.0) * J/mol, 'atomization': array(423000.0) * J/mol, 'poisson_ratio': array(0.23999999999999999) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Cerium', 'boiling_point': array(3633.0) * Kelvin, 'density': array(6.6890000000000001) * g*cm**3, 'double_bond_radius': array(137.0) * pm, 'allred_rochow': 1.0800000000000001, 'young_modulus': array(34.0) * GPa, 'thermal_expansion': array(6.2999999999999998e-06) * 1/Kelvin, 'atomic_radius': array(185.0) * pm})
"""
Cerium elemental data.

  - single_bond_radius: 163.0 pm
  - pauling: 1.12
  - molar_volume: 20.69 cm**3/mol
  - atomization: 423000.0 J/mol
  - sound_velocity: 2100.0 m/s
  - atomic_weight: 140.116
  - triple_bond_radius: 131.0 pm
  - melting_point: 1068.0 K
  - thermal_conductivity: 11.0 W/(m*K)
  - ionization_energies: [array(534400.0) * J/mol, array(1050000.0) * J/mol, array(1949000.0) * J/mol, array(3547000.0) * J/mol, array(6325000.0) * J/mol, array(7490000.0) * J/mol]
  - vaporization: 350000.0 J/mol
  - atomic_number: 58
  - rigidity_modulus: 14.0 GPa
  - symbol: Ce
  - covalent_radius: 204.0 pm
  - fusion: 5500.0 J/mol
  - bulk_modulus: 22.0 GPa
  - poisson_ratio: 0.24 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Cerium
  - boiling_point: 3633.0 K
  - density: 6.689 g*cm**3
  - double_bond_radius: 137.0 pm
  - allred_rochow: 1.08
  - young_modulus: 34.0 GPa
  - thermal_expansion: 6.3e-06 1/K
  - atomic_radius: 185.0 pm
"""

Pr = Element(**{'single_bond_radius': array(176.0) * pm, 'pauling': 1.1299999999999999, 'molar_volume': array(20.800000000000001) * cm**3/mol, 'bulk_modulus': array(29.0) * GPa, 'sound_velocity': array(2280.0) * m/s, 'atomic_weight': 140.90764999999999, 'triple_bond_radius': array(128.0) * pm, 'melting_point': array(1208.0) * Kelvin, 'thermal_conductivity': array(13.0) * Watt/(m*Kelvin), 'ionization_energies': [array(527000.0) * J/mol, array(1020000.0) * J/mol, array(2086000.0) * J/mol, array(3761000.0) * J/mol, array(5551000.0) * J/mol], 'vaporization': array(330000.0) * J/mol, 'atomic_number': 59, 'rigidity_modulus': array(15.0) * GPa, 'symbol': 'Pr', 'covalent_radius': array(203.0) * pm, 'fusion': array(6900.0) * J/mol, 'atomization': array(356000.0) * J/mol, 'poisson_ratio': array(0.28000000000000003) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Praseodymium', 'boiling_point': array(3563.0) * Kelvin, 'density': array(6.6399999999999997) * g*cm**3, 'double_bond_radius': array(138.0) * pm, 'allred_rochow': 1.0700000000000001, 'young_modulus': array(37.0) * GPa, 'thermal_expansion': array(6.7000000000000002e-06) * 1/Kelvin, 'atomic_radius': array(185.0) * pm})
"""
Praseodymium elemental data.

  - single_bond_radius: 176.0 pm
  - pauling: 1.13
  - molar_volume: 20.8 cm**3/mol
  - atomization: 356000.0 J/mol
  - sound_velocity: 2280.0 m/s
  - atomic_weight: 140.90765
  - triple_bond_radius: 128.0 pm
  - melting_point: 1208.0 K
  - thermal_conductivity: 13.0 W/(m*K)
  - ionization_energies: [array(527000.0) * J/mol, array(1020000.0) * J/mol, array(2086000.0) * J/mol, array(3761000.0) * J/mol, array(5551000.0) * J/mol]
  - vaporization: 330000.0 J/mol
  - atomic_number: 59
  - rigidity_modulus: 15.0 GPa
  - symbol: Pr
  - covalent_radius: 203.0 pm
  - fusion: 6900.0 J/mol
  - bulk_modulus: 29.0 GPa
  - poisson_ratio: 0.28 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Praseodymium
  - boiling_point: 3563.0 K
  - density: 6.64 g*cm**3
  - double_bond_radius: 138.0 pm
  - allred_rochow: 1.07
  - young_modulus: 37.0 GPa
  - thermal_expansion: 6.7e-06 1/K
  - atomic_radius: 185.0 pm
"""

Nd = Element(**{'single_bond_radius': array(174.0) * pm, 'pauling': 1.1399999999999999, 'molar_volume': array(20.59) * cm**3/mol, 'bulk_modulus': array(32.0) * GPa, 'sound_velocity': array(2330.0) * m/s, 'atomic_weight': 144.24199999999999, 'melting_point': array(1297.0) * Kelvin, 'thermal_conductivity': array(17.0) * Watt/(m*Kelvin), 'ionization_energies': [array(533100.0) * J/mol, array(1040000.0) * J/mol, array(2130000.0) * J/mol, array(3900000.0) * J/mol], 'vaporization': array(285000.0) * J/mol, 'atomic_number': 60, 'rigidity_modulus': array(16.0) * GPa, 'symbol': 'Nd', 'covalent_radius': array(201.0) * pm, 'fusion': array(7100.0) * J/mol, 'atomization': array(328000.0) * J/mol, 'poisson_ratio': array(0.28000000000000003) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Neodymium', 'boiling_point': array(3373.0) * Kelvin, 'density': array(6.7999999999999998) * g*cm**3, 'double_bond_radius': array(137.0) * pm, 'allred_rochow': 1.0700000000000001, 'young_modulus': array(41.0) * GPa, 'thermal_expansion': array(9.5999999999999996e-06) * 1/Kelvin, 'atomic_radius': array(185.0) * pm})
"""
Neodymium elemental data.

  - single_bond_radius: 174.0 pm
  - pauling: 1.14
  - molar_volume: 20.59 cm**3/mol
  - atomization: 328000.0 J/mol
  - sound_velocity: 2330.0 m/s
  - atomic_weight: 144.242
  - melting_point: 1297.0 K
  - thermal_conductivity: 17.0 W/(m*K)
  - ionization_energies: [array(533100.0) * J/mol, array(1040000.0) * J/mol, array(2130000.0) * J/mol, array(3900000.0) * J/mol]
  - vaporization: 285000.0 J/mol
  - atomic_number: 60
  - rigidity_modulus: 16.0 GPa
  - symbol: Nd
  - covalent_radius: 201.0 pm
  - fusion: 7100.0 J/mol
  - bulk_modulus: 32.0 GPa
  - poisson_ratio: 0.28 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Neodymium
  - boiling_point: 3373.0 K
  - density: 6.8 g*cm**3
  - double_bond_radius: 137.0 pm
  - allred_rochow: 1.07
  - young_modulus: 41.0 GPa
  - thermal_expansion: 9.6e-06 1/K
  - atomic_radius: 185.0 pm
"""

Pm = Element(**{'single_bond_radius': array(173.0) * pm, 'molar_volume': array(20.23) * cm**3/mol, 'bulk_modulus': array(33.0) * GPa, 'young_modulus': array(46.0) * GPa, 'melting_point': array(1373.0) * Kelvin, 'thermal_conductivity': array(15.0) * Watt/(m*Kelvin), 'ionization_energies': [array(540000.0) * J/mol, array(1050000.0) * J/mol, array(2150000.0) * J/mol, array(3970000.0) * J/mol], 'vaporization': array(290000.0) * J/mol, 'atomic_number': 61, 'rigidity_modulus': array(18.0) * GPa, 'symbol': 'Pm', 'covalent_radius': array(199.0) * pm, 'fusion': array(7700.0) * J/mol, 'atomization': array(350000.0) * J/mol, 'poisson_ratio': array(0.28000000000000003) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Promethium', 'boiling_point': array(3273.0) * Kelvin, 'density': array(7.2640000000000002) * g*cm**3, 'double_bond_radius': array(135.0) * pm, 'allred_rochow': 1.0700000000000001, 'atomic_weight': 145.0, 'thermal_expansion': array(1.1e-05) * 1/Kelvin, 'atomic_radius': array(185.0) * pm})
"""
Promethium elemental data.

  - single_bond_radius: 173.0 pm
  - molar_volume: 20.23 cm**3/mol
  - atomization: 350000.0 J/mol
  - atomic_weight: 145.0
  - melting_point: 1373.0 K
  - thermal_conductivity: 15.0 W/(m*K)
  - ionization_energies: [array(540000.0) * J/mol, array(1050000.0) * J/mol, array(2150000.0) * J/mol, array(3970000.0) * J/mol]
  - vaporization: 290000.0 J/mol
  - atomic_number: 61
  - rigidity_modulus: 18.0 GPa
  - symbol: Pm
  - covalent_radius: 199.0 pm
  - fusion: 7700.0 J/mol
  - bulk_modulus: 33.0 GPa
  - poisson_ratio: 0.28 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Promethium
  - boiling_point: 3273.0 K
  - density: 7.264 g*cm**3
  - double_bond_radius: 135.0 pm
  - allred_rochow: 1.07
  - young_modulus: 46.0 GPa
  - thermal_expansion: 1.1e-05 1/K
  - atomic_radius: 185.0 pm
"""

Sm = Element(**{'single_bond_radius': array(172.0) * pm, 'pauling': 1.1699999999999999, 'molar_volume': array(19.98) * cm**3/mol, 'bulk_modulus': array(38.0) * GPa, 'sound_velocity': array(2130.0) * m/s, 'atomic_weight': 150.36000000000001, 'melting_point': array(1345.0) * Kelvin, 'thermal_conductivity': array(13.0) * Watt/(m*Kelvin), 'ionization_energies': [array(544500.0) * J/mol, array(1070000.0) * J/mol, array(2260000.0) * J/mol, array(3990000.0) * J/mol], 'vaporization': array(175000.0) * J/mol, 'atomic_number': 62, 'rigidity_modulus': array(20.0) * GPa, 'symbol': 'Sm', 'covalent_radius': array(198.0) * pm, 'fusion': array(8600.0) * J/mol, 'atomization': array(207000.0) * J/mol, 'poisson_ratio': array(0.27000000000000002) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Samarium', 'boiling_point': array(2076.0) * Kelvin, 'density': array(7.3529999999999998) * g*cm**3, 'double_bond_radius': array(134.0) * pm, 'allred_rochow': 1.0700000000000001, 'young_modulus': array(50.0) * GPa, 'thermal_expansion': array(1.2699999999999999e-05) * 1/Kelvin, 'atomic_radius': array(185.0) * pm})
"""
Samarium elemental data.

  - single_bond_radius: 172.0 pm
  - pauling: 1.17
  - molar_volume: 19.98 cm**3/mol
  - atomization: 207000.0 J/mol
  - sound_velocity: 2130.0 m/s
  - atomic_weight: 150.36
  - melting_point: 1345.0 K
  - thermal_conductivity: 13.0 W/(m*K)
  - ionization_energies: [array(544500.0) * J/mol, array(1070000.0) * J/mol, array(2260000.0) * J/mol, array(3990000.0) * J/mol]
  - vaporization: 175000.0 J/mol
  - atomic_number: 62
  - rigidity_modulus: 20.0 GPa
  - symbol: Sm
  - covalent_radius: 198.0 pm
  - fusion: 8600.0 J/mol
  - bulk_modulus: 38.0 GPa
  - poisson_ratio: 0.27 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Samarium
  - boiling_point: 2076.0 K
  - density: 7.353 g*cm**3
  - double_bond_radius: 134.0 pm
  - allred_rochow: 1.07
  - young_modulus: 50.0 GPa
  - thermal_expansion: 1.27e-05 1/K
  - atomic_radius: 185.0 pm
"""

Eu = Element(**{'single_bond_radius': array(168.0) * pm, 'molar_volume': array(28.969999999999999) * cm**3/mol, 'bulk_modulus': array(8.3000000000000007) * GPa, 'young_modulus': array(18.0) * GPa, 'melting_point': array(1099.0) * Kelvin, 'thermal_conductivity': array(14.0) * Watt/(m*Kelvin), 'ionization_energies': [array(547100.0) * J/mol, array(1085000.0) * J/mol, array(2404000.0) * J/mol, array(4120000.0) * J/mol], 'vaporization': array(175000.0) * J/mol, 'atomic_number': 63, 'rigidity_modulus': array(7.9000000000000004) * GPa, 'symbol': 'Eu', 'covalent_radius': array(198.0) * pm, 'fusion': array(9200.0) * J/mol, 'atomization': array(175000.0) * J/mol, 'poisson_ratio': array(0.14999999999999999) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Europium', 'boiling_point': array(1800.0) * Kelvin, 'density': array(5.2439999999999998) * g*cm**3, 'double_bond_radius': array(134.0) * pm, 'allred_rochow': 1.01, 'atomic_weight': 151.964, 'thermal_expansion': array(3.4999999999999997e-05) * 1/Kelvin, 'atomic_radius': array(185.0) * pm})
"""
Europium elemental data.

  - single_bond_radius: 168.0 pm
  - molar_volume: 28.97 cm**3/mol
  - atomization: 175000.0 J/mol
  - atomic_weight: 151.964
  - melting_point: 1099.0 K
  - thermal_conductivity: 14.0 W/(m*K)
  - ionization_energies: [array(547100.0) * J/mol, array(1085000.0) * J/mol, array(2404000.0) * J/mol, array(4120000.0) * J/mol]
  - vaporization: 175000.0 J/mol
  - atomic_number: 63
  - rigidity_modulus: 7.9 GPa
  - symbol: Eu
  - covalent_radius: 198.0 pm
  - fusion: 9200.0 J/mol
  - bulk_modulus: 8.3 GPa
  - poisson_ratio: 0.15 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Europium
  - boiling_point: 1800.0 K
  - density: 5.244 g*cm**3
  - double_bond_radius: 134.0 pm
  - allred_rochow: 1.01
  - young_modulus: 18.0 GPa
  - thermal_expansion: 3.5e-05 1/K
  - atomic_radius: 185.0 pm
"""

Gd = Element(**{'single_bond_radius': array(169.0) * pm, 'pauling': 1.2, 'molar_volume': array(19.899999999999999) * cm**3/mol, 'bulk_modulus': array(38.0) * GPa, 'sound_velocity': array(2680.0) * m/s, 'atomic_weight': 157.25, 'triple_bond_radius': array(132.0) * pm, 'melting_point': array(1585.0) * Kelvin, 'thermal_conductivity': array(11.0) * Watt/(m*Kelvin), 'ionization_energies': [array(593400.0) * J/mol, array(1170000.0) * J/mol, array(1990000.0) * J/mol, array(4250000.0) * J/mol], 'vaporization': array(305000.0) * J/mol, 'atomic_number': 64, 'rigidity_modulus': array(22.0) * GPa, 'symbol': 'Gd', 'covalent_radius': array(196.0) * pm, 'fusion': array(10000.0) * J/mol, 'atomization': array(398000.0) * J/mol, 'poisson_ratio': array(0.26000000000000001) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Gadolinium', 'boiling_point': array(3523.0) * Kelvin, 'density': array(7.9009999999999998) * g*cm**3, 'double_bond_radius': array(135.0) * pm, 'allred_rochow': 1.1100000000000001, 'young_modulus': array(55.0) * GPa, 'thermal_expansion': array(9.3999999999999998e-06) * 1/Kelvin, 'atomic_radius': array(180.0) * pm})
"""
Gadolinium elemental data.

  - single_bond_radius: 169.0 pm
  - pauling: 1.2
  - molar_volume: 19.9 cm**3/mol
  - atomization: 398000.0 J/mol
  - sound_velocity: 2680.0 m/s
  - atomic_weight: 157.25
  - triple_bond_radius: 132.0 pm
  - melting_point: 1585.0 K
  - thermal_conductivity: 11.0 W/(m*K)
  - ionization_energies: [array(593400.0) * J/mol, array(1170000.0) * J/mol, array(1990000.0) * J/mol, array(4250000.0) * J/mol]
  - vaporization: 305000.0 J/mol
  - atomic_number: 64
  - rigidity_modulus: 22.0 GPa
  - symbol: Gd
  - covalent_radius: 196.0 pm
  - fusion: 10000.0 J/mol
  - bulk_modulus: 38.0 GPa
  - poisson_ratio: 0.26 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Gadolinium
  - boiling_point: 3523.0 K
  - density: 7.901 g*cm**3
  - double_bond_radius: 135.0 pm
  - allred_rochow: 1.11
  - young_modulus: 55.0 GPa
  - thermal_expansion: 9.4e-06 1/K
  - atomic_radius: 180.0 pm
"""

Tb = Element(**{'single_bond_radius': array(168.0) * pm, 'molar_volume': array(19.300000000000001) * cm**3/mol, 'bulk_modulus': array(38.700000000000003) * GPa, 'sound_velocity': array(2620.0) * m/s, 'atomic_weight': 158.92535000000001, 'melting_point': array(1629.0) * Kelvin, 'thermal_conductivity': array(11.0) * Watt/(m*Kelvin), 'ionization_energies': [array(565800.0) * J/mol, array(1110000.0) * J/mol, array(2114000.0) * J/mol, array(3839000.0) * J/mol], 'vaporization': array(295000.0) * J/mol, 'atomic_number': 65, 'rigidity_modulus': array(22.0) * GPa, 'symbol': 'Tb', 'covalent_radius': array(194.0) * pm, 'fusion': array(10800.0) * J/mol, 'atomization': array(389000.0) * J/mol, 'poisson_ratio': array(0.26000000000000001) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Terbium', 'boiling_point': array(3503.0) * Kelvin, 'density': array(8.2189999999999994) * g*cm**3, 'double_bond_radius': array(135.0) * pm, 'allred_rochow': 1.1000000000000001, 'young_modulus': array(56.0) * GPa, 'thermal_expansion': array(1.03e-05) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Terbium elemental data.

  - single_bond_radius: 168.0 pm
  - molar_volume: 19.3 cm**3/mol
  - atomization: 389000.0 J/mol
  - sound_velocity: 2620.0 m/s
  - atomic_weight: 158.92535
  - melting_point: 1629.0 K
  - thermal_conductivity: 11.0 W/(m*K)
  - ionization_energies: [array(565800.0) * J/mol, array(1110000.0) * J/mol, array(2114000.0) * J/mol, array(3839000.0) * J/mol]
  - vaporization: 295000.0 J/mol
  - atomic_number: 65
  - rigidity_modulus: 22.0 GPa
  - symbol: Tb
  - covalent_radius: 194.0 pm
  - fusion: 10800.0 J/mol
  - bulk_modulus: 38.7 GPa
  - poisson_ratio: 0.26 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Terbium
  - boiling_point: 3503.0 K
  - density: 8.219 g*cm**3
  - double_bond_radius: 135.0 pm
  - allred_rochow: 1.1
  - young_modulus: 56.0 GPa
  - thermal_expansion: 1.03e-05 1/K
  - atomic_radius: 175.0 pm
"""

Dy = Element(**{'single_bond_radius': array(167.0) * pm, 'pauling': 1.22, 'molar_volume': array(19.010000000000002) * cm**3/mol, 'bulk_modulus': array(41.0) * GPa, 'sound_velocity': array(2710.0) * m/s, 'atomic_weight': 162.5, 'melting_point': array(1680.0) * Kelvin, 'thermal_conductivity': array(11.0) * Watt/(m*Kelvin), 'ionization_energies': [array(573000.0) * J/mol, array(1130000.0) * J/mol, array(2200000.0) * J/mol, array(3990000.0) * J/mol], 'vaporization': array(280000.0) * J/mol, 'atomic_number': 66, 'rigidity_modulus': array(25.0) * GPa, 'symbol': 'Dy', 'covalent_radius': array(192.0) * pm, 'fusion': array(11100.0) * J/mol, 'atomization': array(290000.0) * J/mol, 'poisson_ratio': array(0.25) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Dysprosium', 'boiling_point': array(2840.0) * Kelvin, 'density': array(8.5510000000000002) * g*cm**3, 'double_bond_radius': array(133.0) * pm, 'allred_rochow': 1.1000000000000001, 'young_modulus': array(61.0) * GPa, 'thermal_expansion': array(9.9000000000000001e-06) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Dysprosium elemental data.

  - single_bond_radius: 167.0 pm
  - pauling: 1.22
  - molar_volume: 19.01 cm**3/mol
  - atomization: 290000.0 J/mol
  - sound_velocity: 2710.0 m/s
  - atomic_weight: 162.5
  - melting_point: 1680.0 K
  - thermal_conductivity: 11.0 W/(m*K)
  - ionization_energies: [array(573000.0) * J/mol, array(1130000.0) * J/mol, array(2200000.0) * J/mol, array(3990000.0) * J/mol]
  - vaporization: 280000.0 J/mol
  - atomic_number: 66
  - rigidity_modulus: 25.0 GPa
  - symbol: Dy
  - covalent_radius: 192.0 pm
  - fusion: 11100.0 J/mol
  - bulk_modulus: 41.0 GPa
  - poisson_ratio: 0.25 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Dysprosium
  - boiling_point: 2840.0 K
  - density: 8.551 g*cm**3
  - double_bond_radius: 133.0 pm
  - allred_rochow: 1.1
  - young_modulus: 61.0 GPa
  - thermal_expansion: 9.9e-06 1/K
  - atomic_radius: 175.0 pm
"""

Ho = Element(**{'single_bond_radius': array(166.0) * pm, 'pauling': 1.23, 'molar_volume': array(18.739999999999998) * cm**3/mol, 'bulk_modulus': array(40.0) * GPa, 'sound_velocity': array(2760.0) * m/s, 'atomic_weight': 164.93031999999999, 'melting_point': array(1734.0) * Kelvin, 'thermal_conductivity': array(16.0) * Watt/(m*Kelvin), 'ionization_energies': [array(581000.0) * J/mol, array(1140000.0) * J/mol, array(2204000.0) * J/mol, array(4100000.0) * J/mol], 'vaporization': array(265000.0) * J/mol, 'atomic_number': 67, 'rigidity_modulus': array(26.0) * GPa, 'symbol': 'Ho', 'covalent_radius': array(192.0) * pm, 'fusion': array(17000.0) * J/mol, 'atomization': array(301000.0) * J/mol, 'poisson_ratio': array(0.23000000000000001) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Holmium', 'boiling_point': array(2993.0) * Kelvin, 'density': array(8.7949999999999999) * g*cm**3, 'double_bond_radius': array(133.0) * pm, 'allred_rochow': 1.1000000000000001, 'young_modulus': array(65.0) * GPa, 'thermal_expansion': array(1.1199999999999999e-05) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Holmium elemental data.

  - single_bond_radius: 166.0 pm
  - pauling: 1.23
  - molar_volume: 18.74 cm**3/mol
  - atomization: 301000.0 J/mol
  - sound_velocity: 2760.0 m/s
  - atomic_weight: 164.93032
  - melting_point: 1734.0 K
  - thermal_conductivity: 16.0 W/(m*K)
  - ionization_energies: [array(581000.0) * J/mol, array(1140000.0) * J/mol, array(2204000.0) * J/mol, array(4100000.0) * J/mol]
  - vaporization: 265000.0 J/mol
  - atomic_number: 67
  - rigidity_modulus: 26.0 GPa
  - symbol: Ho
  - covalent_radius: 192.0 pm
  - fusion: 17000.0 J/mol
  - bulk_modulus: 40.0 GPa
  - poisson_ratio: 0.23 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Holmium
  - boiling_point: 2993.0 K
  - density: 8.795 g*cm**3
  - double_bond_radius: 133.0 pm
  - allred_rochow: 1.1
  - young_modulus: 65.0 GPa
  - thermal_expansion: 1.12e-05 1/K
  - atomic_radius: 175.0 pm
"""

Er = Element(**{'single_bond_radius': array(165.0) * pm, 'pauling': 1.24, 'molar_volume': array(18.460000000000001) * cm**3/mol, 'bulk_modulus': array(44.0) * GPa, 'sound_velocity': array(2830.0) * m/s, 'atomic_weight': 167.25899999999999, 'melting_point': array(1802.0) * Kelvin, 'thermal_conductivity': array(15.0) * Watt/(m*Kelvin), 'ionization_energies': [array(589300.0) * J/mol, array(1150000.0) * J/mol, array(2194000.0) * J/mol, array(4120000.0) * J/mol], 'vaporization': array(285000.0) * J/mol, 'atomic_number': 68, 'rigidity_modulus': array(28.0) * GPa, 'symbol': 'Er', 'covalent_radius': array(189.0) * pm, 'fusion': array(19900.0) * J/mol, 'atomization': array(317000.0) * J/mol, 'poisson_ratio': array(0.23999999999999999) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Erbium', 'boiling_point': array(3141.0) * Kelvin, 'density': array(9.0660000000000007) * g*cm**3, 'double_bond_radius': array(133.0) * pm, 'allred_rochow': 1.1100000000000001, 'young_modulus': array(70.0) * GPa, 'thermal_expansion': array(1.2199999999999998e-05) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Erbium elemental data.

  - single_bond_radius: 165.0 pm
  - pauling: 1.24
  - molar_volume: 18.46 cm**3/mol
  - atomization: 317000.0 J/mol
  - sound_velocity: 2830.0 m/s
  - atomic_weight: 167.259
  - melting_point: 1802.0 K
  - thermal_conductivity: 15.0 W/(m*K)
  - ionization_energies: [array(589300.0) * J/mol, array(1150000.0) * J/mol, array(2194000.0) * J/mol, array(4120000.0) * J/mol]
  - vaporization: 285000.0 J/mol
  - atomic_number: 68
  - rigidity_modulus: 28.0 GPa
  - symbol: Er
  - covalent_radius: 189.0 pm
  - fusion: 19900.0 J/mol
  - bulk_modulus: 44.0 GPa
  - poisson_ratio: 0.24 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Erbium
  - boiling_point: 3141.0 K
  - density: 9.066 g*cm**3
  - double_bond_radius: 133.0 pm
  - allred_rochow: 1.11
  - young_modulus: 70.0 GPa
  - thermal_expansion: 1.22e-05 1/K
  - atomic_radius: 175.0 pm
"""

Tm = Element(**{'single_bond_radius': array(164.0) * pm, 'pauling': 1.25, 'molar_volume': array(19.100000000000001) * cm**3/mol, 'bulk_modulus': array(45.0) * GPa, 'atomic_weight': 168.93421000000001, 'melting_point': array(1818.0) * Kelvin, 'thermal_conductivity': array(17.0) * Watt/(m*Kelvin), 'ionization_energies': [array(596700.0) * J/mol, array(1160000.0) * J/mol, array(2285000.0) * J/mol, array(4120000.0) * J/mol], 'vaporization': array(250000.0) * J/mol, 'atomic_number': 69, 'rigidity_modulus': array(31.0) * GPa, 'symbol': 'Tm', 'covalent_radius': array(190.0) * pm, 'fusion': array(16800.0) * J/mol, 'atomization': array(232000.0) * J/mol, 'poisson_ratio': array(0.20999999999999999) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Thulium', 'boiling_point': array(2223.0) * Kelvin, 'density': array(9.3209999999999997) * g*cm**3, 'double_bond_radius': array(131.0) * pm, 'allred_rochow': 1.1100000000000001, 'young_modulus': array(74.0) * GPa, 'thermal_expansion': array(1.33e-05) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Thulium elemental data.

  - single_bond_radius: 164.0 pm
  - pauling: 1.25
  - molar_volume: 19.1 cm**3/mol
  - atomization: 232000.0 J/mol
  - atomic_weight: 168.93421
  - melting_point: 1818.0 K
  - thermal_conductivity: 17.0 W/(m*K)
  - ionization_energies: [array(596700.0) * J/mol, array(1160000.0) * J/mol, array(2285000.0) * J/mol, array(4120000.0) * J/mol]
  - vaporization: 250000.0 J/mol
  - atomic_number: 69
  - rigidity_modulus: 31.0 GPa
  - symbol: Tm
  - covalent_radius: 190.0 pm
  - fusion: 16800.0 J/mol
  - bulk_modulus: 45.0 GPa
  - poisson_ratio: 0.21 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Thulium
  - boiling_point: 2223.0 K
  - density: 9.321 g*cm**3
  - double_bond_radius: 131.0 pm
  - allred_rochow: 1.11
  - young_modulus: 74.0 GPa
  - thermal_expansion: 1.33e-05 1/K
  - atomic_radius: 175.0 pm
"""

Yb = Element(**{'single_bond_radius': array(170.0) * pm, 'molar_volume': array(24.84) * cm**3/mol, 'bulk_modulus': array(31.0) * GPa, 'sound_velocity': array(1590.0) * m/s, 'atomic_weight': 173.054, 'melting_point': array(1097.0) * Kelvin, 'thermal_conductivity': array(39.0) * Watt/(m*Kelvin), 'ionization_energies': [array(603400.0) * J/mol, array(1174800.0) * J/mol, array(2417000.0) * J/mol, array(4203000.0) * J/mol], 'vaporization': array(160000.0) * J/mol, 'atomic_number': 70, 'rigidity_modulus': array(9.9000000000000004) * GPa, 'symbol': 'Yb', 'covalent_radius': array(187.0) * pm, 'fusion': array(7700.0) * J/mol, 'atomization': array(152000.0) * J/mol, 'poisson_ratio': array(0.20999999999999999) * dimensionless, 'electron_affinity': array(50000.0) * J/mol, 'name': 'Ytterbium', 'boiling_point': array(1469.0) * Kelvin, 'density': array(6.5700000000000003) * g*cm**3, 'double_bond_radius': array(129.0) * pm, 'allred_rochow': 1.0600000000000001, 'young_modulus': array(24.0) * GPa, 'thermal_expansion': array(2.6299999999999999e-05) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Ytterbium elemental data.

  - single_bond_radius: 170.0 pm
  - molar_volume: 24.84 cm**3/mol
  - atomization: 152000.0 J/mol
  - sound_velocity: 1590.0 m/s
  - atomic_weight: 173.054
  - melting_point: 1097.0 K
  - thermal_conductivity: 39.0 W/(m*K)
  - ionization_energies: [array(603400.0) * J/mol, array(1174800.0) * J/mol, array(2417000.0) * J/mol, array(4203000.0) * J/mol]
  - vaporization: 160000.0 J/mol
  - atomic_number: 70
  - rigidity_modulus: 9.9 GPa
  - symbol: Yb
  - covalent_radius: 187.0 pm
  - fusion: 7700.0 J/mol
  - bulk_modulus: 31.0 GPa
  - poisson_ratio: 0.21 dimensionless
  - electron_affinity: 50000.0 J/mol
  - name: Ytterbium
  - boiling_point: 1469.0 K
  - density: 6.57 g*cm**3
  - double_bond_radius: 129.0 pm
  - allred_rochow: 1.06
  - young_modulus: 24.0 GPa
  - thermal_expansion: 2.63e-05 1/K
  - atomic_radius: 175.0 pm
"""

Lu = Element(**{'single_bond_radius': array(162.0) * pm, 'pauling': 1.27, 'molar_volume': array(17.780000000000001) * cm**3/mol, 'bulk_modulus': array(48.0) * GPa, 'atomic_weight': 174.96680000000001, 'triple_bond_radius': array(131.0) * pm, 'melting_point': array(1925.0) * Kelvin, 'thermal_conductivity': array(16.0) * Watt/(m*Kelvin), 'ionization_energies': [array(523500.0) * J/mol, array(1340000.0) * J/mol, array(2022300.0) * J/mol, array(4370000.0) * J/mol, array(6445000.0) * J/mol], 'vaporization': array(415000.0) * J/mol, 'atomic_number': 71, 'rigidity_modulus': array(27.0) * GPa, 'symbol': 'Lu', 'covalent_radius': array(187.0) * pm, 'fusion': array(22000.0) * J/mol, 'atomization': array(428000.0) * J/mol, 'poisson_ratio': array(0.26000000000000001) * dimensionless, 'electron_affinity': array(33000.0) * J/mol, 'name': 'Lutetium', 'boiling_point': array(3675.0) * Kelvin, 'density': array(9.8409999999999993) * g*cm**3, 'double_bond_radius': array(131.0) * pm, 'allred_rochow': 1.1399999999999999, 'young_modulus': array(69.0) * GPa, 'thermal_expansion': array(9.9000000000000001e-06) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Lutetium elemental data.

  - single_bond_radius: 162.0 pm
  - pauling: 1.27
  - molar_volume: 17.78 cm**3/mol
  - atomization: 428000.0 J/mol
  - atomic_weight: 174.9668
  - triple_bond_radius: 131.0 pm
  - melting_point: 1925.0 K
  - thermal_conductivity: 16.0 W/(m*K)
  - ionization_energies: [array(523500.0) * J/mol, array(1340000.0) * J/mol, array(2022300.0) * J/mol, array(4370000.0) * J/mol, array(6445000.0) * J/mol]
  - vaporization: 415000.0 J/mol
  - atomic_number: 71
  - rigidity_modulus: 27.0 GPa
  - symbol: Lu
  - covalent_radius: 187.0 pm
  - fusion: 22000.0 J/mol
  - bulk_modulus: 48.0 GPa
  - poisson_ratio: 0.26 dimensionless
  - electron_affinity: 33000.0 J/mol
  - name: Lutetium
  - boiling_point: 3675.0 K
  - density: 9.841 g*cm**3
  - double_bond_radius: 131.0 pm
  - allred_rochow: 1.14
  - young_modulus: 69.0 GPa
  - thermal_expansion: 9.9e-06 1/K
  - atomic_radius: 175.0 pm
"""

Hf = Element(**{'single_bond_radius': array(152.0) * pm, 'pauling': 1.3, 'molar_volume': array(13.44) * cm**3/mol, 'bulk_modulus': array(110.0) * GPa, 'sound_velocity': array(3010.0) * m/s, 'atomic_weight': 178.49000000000001, 'triple_bond_radius': array(122.0) * pm, 'melting_point': array(2506.0) * Kelvin, 'orbital_radii': (array(0.68793042370000002) * angstrom, array(0.85197537088999997) * angstrom, array(0.33338166686999998) * angstrom), 'thermal_conductivity': array(23.0) * Watt/(m*Kelvin), 'ionization_energies': [array(658500.0) * J/mol, array(1440000.0) * J/mol, array(2250000.0) * J/mol, array(3216000.0) * J/mol], 'vaporization': array(630000.0) * J/mol, 'atomic_number': 72, 'rigidity_modulus': array(30.0) * GPa, 'symbol': 'Hf', 'covalent_radius': array(175.0) * pm, 'fusion': array(25500.0) * J/mol, 'pettifor': 0.77500000000000002, 'atomization': array(621000.0) * J/mol, 'poisson_ratio': array(0.37) * dimensionless, 'electron_affinity': array(0.0) * J/mol, 'name': 'Hafnium', 'boiling_point': array(4876.0) * Kelvin, 'density': array(13.31) * g*cm**3, 'double_bond_radius': array(128.0) * pm, 'allred_rochow': 1.23, 'young_modulus': array(78.0) * GPa, 'thermal_expansion': array(5.9000000000000003e-06) * 1/Kelvin, 'atomic_radius': array(155.0) * pm})
"""
Hafnium elemental data.

  - single_bond_radius: 152.0 pm
  - pauling: 1.3
  - molar_volume: 13.44 cm**3/mol
  - atomization: 621000.0 J/mol
  - sound_velocity: 3010.0 m/s
  - atomic_weight: 178.49
  - triple_bond_radius: 122.0 pm
  - melting_point: 2506.0 K
  - orbital_radii: (array(0.68793042370000002) * angstrom, array(0.85197537088999997) * angstrom, array(0.33338166686999998) * angstrom)
  - thermal_conductivity: 23.0 W/(m*K)
  - ionization_energies: [array(658500.0) * J/mol, array(1440000.0) * J/mol, array(2250000.0) * J/mol, array(3216000.0) * J/mol]
  - vaporization: 630000.0 J/mol
  - atomic_number: 72
  - rigidity_modulus: 30.0 GPa
  - symbol: Hf
  - covalent_radius: 175.0 pm
  - fusion: 25500.0 J/mol
  - pettifor: 0.775
  - bulk_modulus: 110.0 GPa
  - poisson_ratio: 0.37 dimensionless
  - electron_affinity: 0.0 J/mol
  - name: Hafnium
  - boiling_point: 4876.0 K
  - density: 13.31 g*cm**3
  - double_bond_radius: 128.0 pm
  - allred_rochow: 1.23
  - young_modulus: 78.0 GPa
  - thermal_expansion: 5.9e-06 1/K
  - atomic_radius: 155.0 pm
"""

Ta = Element(**{'single_bond_radius': array(146.0) * pm, 'pauling': 1.5, 'molar_volume': array(10.85) * cm**3/mol, 'bulk_modulus': array(200.0) * GPa, 'sound_velocity': array(3400.0) * m/s, 'atomic_weight': 180.94788, 'triple_bond_radius': array(119.0) * pm, 'melting_point': array(3290.0) * Kelvin, 'orbital_radii': (array(0.66147156124999995) * angstrom, array(0.81493296345999999) * angstrom, array(0.32015223564499995) * angstrom), 'thermal_conductivity': array(57.0) * Watt/(m*Kelvin), 'ionization_energies': [array(761000.0) * J/mol, array(1500000.0) * J/mol], 'vaporization': array(735000.0) * J/mol, 'atomic_number': 73, 'rigidity_modulus': array(69.0) * GPa, 'symbol': 'Ta', 'covalent_radius': array(170.0) * pm, 'fusion': array(36000.0) * J/mol, 'pettifor': 0.82999999999999996, 'atomization': array(782000.0) * J/mol, 'poisson_ratio': array(0.34000000000000002) * dimensionless, 'electron_affinity': array(31000.0) * J/mol, 'name': 'Tantalum', 'boiling_point': array(5731.0) * Kelvin, 'density': array(16.649999999999999) * g*cm**3, 'double_bond_radius': array(126.0) * pm, 'allred_rochow': 1.3300000000000001, 'young_modulus': array(186.0) * GPa, 'thermal_expansion': array(6.2999999999999998e-06) * 1/Kelvin, 'atomic_radius': array(145.0) * pm})
"""
Tantalum elemental data.

  - single_bond_radius: 146.0 pm
  - pauling: 1.5
  - molar_volume: 10.85 cm**3/mol
  - atomization: 782000.0 J/mol
  - sound_velocity: 3400.0 m/s
  - atomic_weight: 180.94788
  - triple_bond_radius: 119.0 pm
  - melting_point: 3290.0 K
  - orbital_radii: (array(0.66147156124999995) * angstrom, array(0.81493296345999999) * angstrom, array(0.32015223564499995) * angstrom)
  - thermal_conductivity: 57.0 W/(m*K)
  - ionization_energies: [array(761000.0) * J/mol, array(1500000.0) * J/mol]
  - vaporization: 735000.0 J/mol
  - atomic_number: 73
  - rigidity_modulus: 69.0 GPa
  - symbol: Ta
  - covalent_radius: 170.0 pm
  - fusion: 36000.0 J/mol
  - pettifor: 0.83
  - bulk_modulus: 200.0 GPa
  - poisson_ratio: 0.34 dimensionless
  - electron_affinity: 31000.0 J/mol
  - name: Tantalum
  - boiling_point: 5731.0 K
  - density: 16.65 g*cm**3
  - double_bond_radius: 126.0 pm
  - allred_rochow: 1.33
  - young_modulus: 186.0 GPa
  - thermal_expansion: 6.3e-06 1/K
  - atomic_radius: 145.0 pm
"""

W = Element(**{'single_bond_radius': array(137.0) * pm, 'pauling': 2.3599999999999999, 'molar_volume': array(9.4700000000000006) * cm**3/mol, 'bulk_modulus': array(310.0) * GPa, 'sound_velocity': array(5174.0) * m/s, 'sanderson': 0.97999999999999998, 'atomic_weight': 183.84, 'triple_bond_radius': array(115.0) * pm, 'melting_point': array(3695.0) * Kelvin, 'orbital_radii': (array(0.64559624377999991) * angstrom, array(0.8017035322349999) * angstrom, array(0.31221457690999999) * angstrom), 'thermal_conductivity': array(170.0) * Watt/(m*Kelvin), 'ionization_energies': [array(770000.0) * J/mol, array(1700000.0) * J/mol], 'vaporization': array(800000.0) * J/mol, 'atomic_number': 74, 'rigidity_modulus': array(161.0) * GPa, 'symbol': 'W', 'covalent_radius': array(162.0) * pm, 'fusion': array(35000.0) * J/mol, 'pettifor': 0.88500000000000001, 'atomization': array(860000.0) * J/mol, 'poisson_ratio': array(0.28000000000000003) * dimensionless, 'electron_affinity': array(78600.0) * J/mol, 'name': 'Tungsten', 'boiling_point': array(5828.0) * Kelvin, 'density': array(19.25) * g*cm**3, 'double_bond_radius': array(120.0) * pm, 'allred_rochow': 1.3999999999999999, 'young_modulus': array(411.0) * GPa, 'thermal_expansion': array(4.5000000000000001e-06) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Tungsten elemental data.

  - single_bond_radius: 137.0 pm
  - pauling: 2.36
  - molar_volume: 9.47 cm**3/mol
  - atomization: 860000.0 J/mol
  - sound_velocity: 5174.0 m/s
  - sanderson: 0.98
  - atomic_weight: 183.84
  - triple_bond_radius: 115.0 pm
  - melting_point: 3695.0 K
  - orbital_radii: (array(0.64559624377999991) * angstrom, array(0.8017035322349999) * angstrom, array(0.31221457690999999) * angstrom)
  - thermal_conductivity: 170.0 W/(m*K)
  - ionization_energies: [array(770000.0) * J/mol, array(1700000.0) * J/mol]
  - vaporization: 800000.0 J/mol
  - atomic_number: 74
  - rigidity_modulus: 161.0 GPa
  - symbol: W
  - covalent_radius: 162.0 pm
  - fusion: 35000.0 J/mol
  - pettifor: 0.885
  - bulk_modulus: 310.0 GPa
  - poisson_ratio: 0.28 dimensionless
  - electron_affinity: 78600.0 J/mol
  - name: Tungsten
  - boiling_point: 5828.0 K
  - density: 19.25 g*cm**3
  - double_bond_radius: 120.0 pm
  - allred_rochow: 1.4
  - young_modulus: 411.0 GPa
  - thermal_expansion: 4.5e-06 1/K
  - atomic_radius: 135.0 pm
"""

Re = Element(**{'single_bond_radius': array(131.0) * pm, 'pauling': 1.8999999999999999, 'molar_volume': array(8.8599999999999994) * cm**3/mol, 'bulk_modulus': array(370.0) * GPa, 'sound_velocity': array(4700.0) * m/s, 'atomic_weight': 186.20699999999999, 'triple_bond_radius': array(110.0) * pm, 'melting_point': array(3459.0) * Kelvin, 'orbital_radii': (array(0.62972092630999987) * angstrom, array(0.78847410100999993) * angstrom, array(0.29898514568499995) * angstrom), 'thermal_conductivity': array(48.0) * Watt/(m*Kelvin), 'ionization_energies': [array(760000.0) * J/mol, array(1260000.0) * J/mol, array(2510000.0) * J/mol, array(3640000.0) * J/mol], 'vaporization': array(705000.0) * J/mol, 'atomic_number': 75, 'rigidity_modulus': array(178.0) * GPa, 'symbol': 'Re', 'covalent_radius': array(151.0) * pm, 'fusion': array(33000.0) * J/mol, 'pettifor': 0.93999999999999995, 'atomization': array(776000.0) * J/mol, 'poisson_ratio': array(0.29999999999999999) * dimensionless, 'electron_affinity': array(14500.0) * J/mol, 'name': 'Rhenium', 'boiling_point': array(5869.0) * Kelvin, 'density': array(21.02) * g*cm**3, 'double_bond_radius': array(119.0) * pm, 'allred_rochow': 1.46, 'young_modulus': array(463.0) * GPa, 'thermal_expansion': array(6.1999999999999999e-06) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Rhenium elemental data.

  - single_bond_radius: 131.0 pm
  - pauling: 1.9
  - molar_volume: 8.86 cm**3/mol
  - atomization: 776000.0 J/mol
  - sound_velocity: 4700.0 m/s
  - atomic_weight: 186.207
  - triple_bond_radius: 110.0 pm
  - melting_point: 3459.0 K
  - orbital_radii: (array(0.62972092630999987) * angstrom, array(0.78847410100999993) * angstrom, array(0.29898514568499995) * angstrom)
  - thermal_conductivity: 48.0 W/(m*K)
  - ionization_energies: [array(760000.0) * J/mol, array(1260000.0) * J/mol, array(2510000.0) * J/mol, array(3640000.0) * J/mol]
  - vaporization: 705000.0 J/mol
  - atomic_number: 75
  - rigidity_modulus: 178.0 GPa
  - symbol: Re
  - covalent_radius: 151.0 pm
  - fusion: 33000.0 J/mol
  - pettifor: 0.94
  - bulk_modulus: 370.0 GPa
  - poisson_ratio: 0.3 dimensionless
  - electron_affinity: 14500.0 J/mol
  - name: Rhenium
  - boiling_point: 5869.0 K
  - density: 21.02 g*cm**3
  - double_bond_radius: 119.0 pm
  - allred_rochow: 1.46
  - young_modulus: 463.0 GPa
  - thermal_expansion: 6.2e-06 1/K
  - atomic_radius: 135.0 pm
"""

Os = Element(**{'single_bond_radius': array(129.0) * pm, 'pauling': 2.2000000000000002, 'molar_volume': array(8.4199999999999999) * cm**3/mol, 'atomization': array(789000.0) * J/mol, 'sound_velocity': array(4940.0) * m/s, 'atomic_weight': 190.22999999999999, 'triple_bond_radius': array(109.0) * pm, 'melting_point': array(3306.0) * Kelvin, 'orbital_radii': (array(0.61913738132999996) * angstrom, array(0.78318232851999992) * angstrom, array(0.28734324620700002) * angstrom), 'thermal_conductivity': array(88.0) * Watt/(m*Kelvin), 'ionization_energies': [array(840000.0) * J/mol, array(1600000.0) * J/mol], 'vaporization': array(630000.0) * J/mol, 'atomic_number': 76, 'rigidity_modulus': array(222.0) * GPa, 'symbol': 'Os', 'covalent_radius': array(144.0) * pm, 'fusion': array(31000.0) * J/mol, 'pettifor': 0.995, 'poisson_ratio': array(0.25) * dimensionless, 'electron_affinity': array(106100.0) * J/mol, 'name': 'Osmium', 'boiling_point': array(5285.0) * Kelvin, 'density': array(22.609999999999999) * g*cm**3, 'double_bond_radius': array(116.0) * pm, 'allred_rochow': 1.52, 'thermal_expansion': array(5.0999999999999995e-06) * 1/Kelvin, 'atomic_radius': array(130.0) * pm})
"""
Osmium elemental data.

  - single_bond_radius: 129.0 pm
  - pauling: 2.2
  - molar_volume: 8.42 cm**3/mol
  - atomization: 789000.0 J/mol
  - sound_velocity: 4940.0 m/s
  - atomic_weight: 190.23
  - triple_bond_radius: 109.0 pm
  - melting_point: 3306.0 K
  - orbital_radii: (array(0.61913738132999996) * angstrom, array(0.78318232851999992) * angstrom, array(0.28734324620700002) * angstrom)
  - thermal_conductivity: 88.0 W/(m*K)
  - ionization_energies: [array(840000.0) * J/mol, array(1600000.0) * J/mol]
  - vaporization: 630000.0 J/mol
  - atomic_number: 76
  - rigidity_modulus: 222.0 GPa
  - symbol: Os
  - covalent_radius: 144.0 pm
  - fusion: 31000.0 J/mol
  - pettifor: 0.995
  - poisson_ratio: 0.25 dimensionless
  - electron_affinity: 106100.0 J/mol
  - name: Osmium
  - boiling_point: 5285.0 K
  - density: 22.61 g*cm**3
  - double_bond_radius: 116.0 pm
  - allred_rochow: 1.52
  - thermal_expansion: 5.1e-06 1/K
  - atomic_radius: 130.0 pm
"""

Ir = Element(**{'single_bond_radius': array(122.0) * pm, 'pauling': 2.2000000000000002, 'molar_volume': array(8.5199999999999996) * cm**3/mol, 'bulk_modulus': array(320.0) * GPa, 'sound_velocity': array(4825.0) * m/s, 'atomic_weight': 192.21700000000001, 'triple_bond_radius': array(107.0) * pm, 'melting_point': array(2739.0) * Kelvin, 'orbital_radii': (array(0.61384560883999995) * angstrom, array(0.77683220153199994) * angstrom, array(0.27834723297399999) * angstrom), 'thermal_conductivity': array(150.0) * Watt/(m*Kelvin), 'ionization_energies': [array(880000.0) * J/mol, array(1600000.0) * J/mol], 'vaporization': array(560000.0) * J/mol, 'atomic_number': 77, 'rigidity_modulus': array(210.0) * GPa, 'symbol': 'Ir', 'covalent_radius': array(141.0) * pm, 'fusion': array(26000.0) * J/mol, 'pettifor': 1.05, 'atomization': array(671000.0) * J/mol, 'poisson_ratio': array(0.26000000000000001) * dimensionless, 'electron_affinity': array(151000.0) * J/mol, 'name': 'Iridium', 'boiling_point': array(4701.0) * Kelvin, 'density': array(22.649999999999999) * g*cm**3, 'double_bond_radius': array(115.0) * pm, 'allred_rochow': 1.55, 'young_modulus': array(528.0) * GPa, 'thermal_expansion': array(6.3999999999999997e-06) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Iridium elemental data.

  - single_bond_radius: 122.0 pm
  - pauling: 2.2
  - molar_volume: 8.52 cm**3/mol
  - atomization: 671000.0 J/mol
  - sound_velocity: 4825.0 m/s
  - atomic_weight: 192.217
  - triple_bond_radius: 107.0 pm
  - melting_point: 2739.0 K
  - orbital_radii: (array(0.61384560883999995) * angstrom, array(0.77683220153199994) * angstrom, array(0.27834723297399999) * angstrom)
  - thermal_conductivity: 150.0 W/(m*K)
  - ionization_energies: [array(880000.0) * J/mol, array(1600000.0) * J/mol]
  - vaporization: 560000.0 J/mol
  - atomic_number: 77
  - rigidity_modulus: 210.0 GPa
  - symbol: Ir
  - covalent_radius: 141.0 pm
  - fusion: 26000.0 J/mol
  - pettifor: 1.05
  - bulk_modulus: 320.0 GPa
  - poisson_ratio: 0.26 dimensionless
  - electron_affinity: 151000.0 J/mol
  - name: Iridium
  - boiling_point: 4701.0 K
  - density: 22.65 g*cm**3
  - double_bond_radius: 115.0 pm
  - allred_rochow: 1.55
  - young_modulus: 528.0 GPa
  - thermal_expansion: 6.4e-06 1/K
  - atomic_radius: 135.0 pm
"""

Pt = Element(**{'single_bond_radius': array(123.0) * pm, 'pauling': 2.2799999999999998, 'molar_volume': array(9.0899999999999999) * cm**3/mol, 'bulk_modulus': array(230.0) * GPa, 'sound_velocity': array(2680.0) * m/s, 'atomic_weight': 195.084, 'triple_bond_radius': array(110.0) * pm, 'melting_point': array(2041.4000000000001) * Kelvin, 'orbital_radii': (array(0.65617978875999994) * angstrom, array(0.77259878353999989) * angstrom, array(0.26988039698999999) * angstrom), 'thermal_conductivity': array(72.0) * Watt/(m*Kelvin), 'ionization_energies': [array(870000.0) * J/mol, array(1791000.0) * J/mol], 'vaporization': array(490000.0) * J/mol, 'atomic_number': 78, 'rigidity_modulus': array(61.0) * GPa, 'symbol': 'Pt', 'covalent_radius': array(136.0) * pm, 'fusion': array(20000.0) * J/mol, 'pettifor': 1.105, 'atomization': array(565000.0) * J/mol, 'poisson_ratio': array(0.38) * dimensionless, 'van_der_waals_radius': array(175.0) * pm, 'electron_affinity': array(205300.0) * J/mol, 'name': 'Platinum', 'boiling_point': array(4098.0) * Kelvin, 'density': array(21.09) * g*cm**3, 'double_bond_radius': array(112.0) * pm, 'allred_rochow': 1.4399999999999999, 'young_modulus': array(168.0) * GPa, 'thermal_expansion': array(8.8000000000000004e-06) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Platinum elemental data.

  - single_bond_radius: 123.0 pm
  - pauling: 2.28
  - molar_volume: 9.09 cm**3/mol
  - atomization: 565000.0 J/mol
  - sound_velocity: 2680.0 m/s
  - atomic_weight: 195.084
  - triple_bond_radius: 110.0 pm
  - melting_point: 2041.4 K
  - orbital_radii: (array(0.65617978875999994) * angstrom, array(0.77259878353999989) * angstrom, array(0.26988039698999999) * angstrom)
  - thermal_conductivity: 72.0 W/(m*K)
  - ionization_energies: [array(870000.0) * J/mol, array(1791000.0) * J/mol]
  - vaporization: 490000.0 J/mol
  - atomic_number: 78
  - rigidity_modulus: 61.0 GPa
  - symbol: Pt
  - covalent_radius: 136.0 pm
  - fusion: 20000.0 J/mol
  - pettifor: 1.105
  - bulk_modulus: 230.0 GPa
  - poisson_ratio: 0.38 dimensionless
  - van_der_waals_radius: 175.0 pm
  - electron_affinity: 205300.0 J/mol
  - name: Platinum
  - boiling_point: 4098.0 K
  - density: 21.09 g*cm**3
  - double_bond_radius: 112.0 pm
  - allred_rochow: 1.44
  - young_modulus: 168.0 GPa
  - thermal_expansion: 8.8e-06 1/K
  - atomic_radius: 135.0 pm
"""

Au = Element(**{'mulliken_jaffe': 1.8700000000000001, 'single_bond_radius': array(124.0) * pm, 'pauling': 2.54, 'molar_volume': array(10.210000000000001) * cm**3/mol, 'bulk_modulus': array(220.0) * GPa, 'sound_velocity': array(1740.0) * m/s, 'atomic_weight': 196.96656899999999, 'triple_bond_radius': array(123.0) * pm, 'melting_point': array(1337.3299999999999) * Kelvin, 'orbital_radii': (array(0.6403044712899999) * angstrom, array(0.76730701104999988) * angstrom, array(0.25823849751199995) * angstrom), 'thermal_conductivity': array(320.0) * Watt/(m*Kelvin), 'ionization_energies': [array(890100.0) * J/mol, array(1980000.0) * J/mol], 'vaporization': array(330000.0) * J/mol, 'atomic_number': 79, 'rigidity_modulus': array(27.0) * GPa, 'symbol': 'Au', 'covalent_radius': array(136.0) * pm, 'fusion': array(12500.0) * J/mol, 'pettifor': 1.1599999999999999, 'atomization': array(368000.0) * J/mol, 'poisson_ratio': array(0.44) * dimensionless, 'van_der_waals_radius': array(166.0) * pm, 'electron_affinity': array(222800.0) * J/mol, 'name': 'Gold', 'boiling_point': array(3129.0) * Kelvin, 'density': array(19.300000000000001) * g*cm**3, 'double_bond_radius': array(121.0) * pm, 'allred_rochow': 1.4199999999999999, 'young_modulus': array(78.0) * GPa, 'thermal_expansion': array(1.4199999999999998e-05) * 1/Kelvin, 'atomic_radius': array(135.0) * pm})
"""
Gold elemental data.

  - mulliken_jaffe: 1.87
  - single_bond_radius: 124.0 pm
  - pauling: 2.54
  - molar_volume: 10.21 cm**3/mol
  - atomization: 368000.0 J/mol
  - sound_velocity: 1740.0 m/s
  - atomic_weight: 196.966569
  - triple_bond_radius: 123.0 pm
  - melting_point: 1337.33 K
  - orbital_radii: (array(0.6403044712899999) * angstrom, array(0.76730701104999988) * angstrom, array(0.25823849751199995) * angstrom)
  - thermal_conductivity: 320.0 W/(m*K)
  - ionization_energies: [array(890100.0) * J/mol, array(1980000.0) * J/mol]
  - vaporization: 330000.0 J/mol
  - atomic_number: 79
  - rigidity_modulus: 27.0 GPa
  - symbol: Au
  - covalent_radius: 136.0 pm
  - fusion: 12500.0 J/mol
  - pettifor: 1.16
  - bulk_modulus: 220.0 GPa
  - poisson_ratio: 0.44 dimensionless
  - van_der_waals_radius: 166.0 pm
  - electron_affinity: 222800.0 J/mol
  - name: Gold
  - boiling_point: 3129.0 K
  - density: 19.3 g*cm**3
  - double_bond_radius: 121.0 pm
  - allred_rochow: 1.42
  - young_modulus: 78.0 GPa
  - thermal_expansion: 1.42e-05 1/K
  - atomic_radius: 135.0 pm
"""

Hg = Element(**{'mulliken_jaffe': 1.8100000000000001, 'single_bond_radius': array(133.0) * pm, 'pauling': 2.0, 'molar_volume': array(14.09) * cm**3/mol, 'bulk_modulus': array(25.0) * GPa, 'sound_velocity': array(1407.0) * m/s, 'sanderson': 2.2000000000000002, 'atomic_weight': 200.59, 'critical_temperature': array(1750.0) * Kelvin, 'melting_point': array(234.31999999999999) * Kelvin, 'orbital_radii': (array(0.56621965642999994) * angstrom, array(0.70909751365999996) * angstrom, array(0.25135919327499995) * angstrom), 'thermal_conductivity': array(8.3000000000000007) * Watt/(m*Kelvin), 'ionization_energies': [array(1007100.0) * J/mol, array(1810000.0) * J/mol, array(3300000.0) * J/mol], 'vaporization': array(59200.0) * J/mol, 'atomic_number': 80, 'symbol': 'Hg', 'covalent_radius': array(132.0) * pm, 'fusion': array(2290.0) * J/mol, 'pettifor': 1.3200000000000001, 'atomization': array(64000.0) * J/mol, 'van_der_waals_radius': array(155.0) * pm, 'electron_affinity': array(0.0) * J/mol, 'name': 'Mercury', 'boiling_point': array(629.88) * Kelvin, 'double_bond_radius': array(142.0) * pm, 'allred_rochow': 1.4399999999999999, 'atomic_radius': array(150.0) * pm})
"""
Mercury elemental data.

  - mulliken_jaffe: 1.81
  - single_bond_radius: 133.0 pm
  - pauling: 2.0
  - molar_volume: 14.09 cm**3/mol
  - atomization: 64000.0 J/mol
  - sound_velocity: 1407.0 m/s
  - sanderson: 2.2
  - atomic_weight: 200.59
  - critical_temperature: 1750.0 K
  - melting_point: 234.32 K
  - orbital_radii: (array(0.56621965642999994) * angstrom, array(0.70909751365999996) * angstrom, array(0.25135919327499995) * angstrom)
  - thermal_conductivity: 8.3 W/(m*K)
  - ionization_energies: [array(1007100.0) * J/mol, array(1810000.0) * J/mol, array(3300000.0) * J/mol]
  - vaporization: 59200.0 J/mol
  - atomic_number: 80
  - symbol: Hg
  - covalent_radius: 132.0 pm
  - fusion: 2290.0 J/mol
  - pettifor: 1.32
  - bulk_modulus: 25.0 GPa
  - van_der_waals_radius: 155.0 pm
  - electron_affinity: 0.0 J/mol
  - name: Mercury
  - boiling_point: 629.88 K
  - double_bond_radius: 142.0 pm
  - allred_rochow: 1.44
  - atomic_radius: 150.0 pm
"""

Tl = Element(**{'mulliken_jaffe': 1.96, 'single_bond_radius': array(144.0) * pm, 'pauling': 1.6200000000000001, 'molar_volume': array(17.219999999999999) * cm**3/mol, 'bulk_modulus': array(43.0) * GPa, 'sound_velocity': array(818.0) * m/s, 'sanderson': 2.25, 'atomic_weight': 204.38329999999999, 'triple_bond_radius': array(150.0) * pm, 'melting_point': array(577.0) * Kelvin, 'orbital_radii': (array(0.53711490773499992) * angstrom, array(0.64559624377999991) * angstrom, array(0.245009066287) * angstrom), 'thermal_conductivity': array(46.0) * Watt/(m*Kelvin), 'ionization_energies': [array(589400.0) * J/mol, array(1971000.0) * J/mol, array(2878000.0) * J/mol], 'vaporization': array(165000.0) * J/mol, 'atomic_number': 81, 'rigidity_modulus': array(2.7999999999999998) * GPa, 'symbol': 'Tl', 'covalent_radius': array(145.0) * pm, 'fusion': array(4200.0) * J/mol, 'pettifor': 1.5600000000000001, 'atomization': array(182000.0) * J/mol, 'poisson_ratio': array(0.45000000000000001) * dimensionless, 'van_der_waals_radius': array(196.0) * pm, 'electron_affinity': array(19200.0) * J/mol, 'name': 'Thallium', 'boiling_point': array(1746.0) * Kelvin, 'density': array(11.85) * g*cm**3, 'double_bond_radius': array(142.0) * pm, 'allred_rochow': 1.4399999999999999, 'young_modulus': array(8.0) * GPa, 'thermal_expansion': array(2.9899999999999998e-05) * 1/Kelvin, 'atomic_radius': array(190.0) * pm})
"""
Thallium elemental data.

  - mulliken_jaffe: 1.96
  - single_bond_radius: 144.0 pm
  - pauling: 1.62
  - molar_volume: 17.22 cm**3/mol
  - atomization: 182000.0 J/mol
  - sound_velocity: 818.0 m/s
  - sanderson: 2.25
  - atomic_weight: 204.3833
  - triple_bond_radius: 150.0 pm
  - melting_point: 577.0 K
  - orbital_radii: (array(0.53711490773499992) * angstrom, array(0.64559624377999991) * angstrom, array(0.245009066287) * angstrom)
  - thermal_conductivity: 46.0 W/(m*K)
  - ionization_energies: [array(589400.0) * J/mol, array(1971000.0) * J/mol, array(2878000.0) * J/mol]
  - vaporization: 165000.0 J/mol
  - atomic_number: 81
  - rigidity_modulus: 2.8 GPa
  - symbol: Tl
  - covalent_radius: 145.0 pm
  - fusion: 4200.0 J/mol
  - pettifor: 1.56
  - bulk_modulus: 43.0 GPa
  - poisson_ratio: 0.45 dimensionless
  - van_der_waals_radius: 196.0 pm
  - electron_affinity: 19200.0 J/mol
  - name: Thallium
  - boiling_point: 1746.0 K
  - density: 11.85 g*cm**3
  - double_bond_radius: 142.0 pm
  - allred_rochow: 1.44
  - young_modulus: 8.0 GPa
  - thermal_expansion: 2.99e-05 1/K
  - atomic_radius: 190.0 pm
"""

Pb = Element(**{'mulliken_jaffe': 2.4100000000000001, 'single_bond_radius': array(144.0) * pm, 'pauling': 2.3300000000000001, 'molar_volume': array(18.260000000000002) * cm**3/mol, 'bulk_modulus': array(46.0) * GPa, 'sound_velocity': array(1260.0) * m/s, 'sanderson': 2.29, 'atomic_weight': 207.19999999999999, 'triple_bond_radius': array(137.0) * pm, 'melting_point': array(600.61000000000001) * Kelvin, 'orbital_radii': (array(0.50801015903999991) * angstrom, array(0.59797029136999991) * angstrom, array(0.23812976205) * angstrom), 'thermal_conductivity': array(35.0) * Watt/(m*Kelvin), 'ionization_energies': [array(715600.0) * J/mol, array(1450500.0) * J/mol, array(3081500.0) * J/mol, array(4083000.0) * J/mol, array(6640000.0) * J/mol], 'vaporization': array(178000.0) * J/mol, 'atomic_number': 82, 'rigidity_modulus': array(5.5999999999999996) * GPa, 'symbol': 'Pb', 'covalent_radius': array(146.0) * pm, 'fusion': array(4770.0) * J/mol, 'pettifor': 1.8, 'atomization': array(195000.0) * J/mol, 'poisson_ratio': array(0.44) * dimensionless, 'van_der_waals_radius': array(202.0) * pm, 'electron_affinity': array(35100.0) * J/mol, 'name': 'Lead', 'boiling_point': array(2022.0) * Kelvin, 'density': array(11.34) * g*cm**3, 'double_bond_radius': array(135.0) * pm, 'allred_rochow': 1.55, 'young_modulus': array(16.0) * GPa, 'thermal_expansion': array(2.8899999999999998e-05) * 1/Kelvin, 'atomic_radius': array(180.0) * pm})
"""
Lead elemental data.

  - mulliken_jaffe: 2.41
  - single_bond_radius: 144.0 pm
  - pauling: 2.33
  - molar_volume: 18.26 cm**3/mol
  - atomization: 195000.0 J/mol
  - sound_velocity: 1260.0 m/s
  - sanderson: 2.29
  - atomic_weight: 207.2
  - triple_bond_radius: 137.0 pm
  - melting_point: 600.61 K
  - orbital_radii: (array(0.50801015903999991) * angstrom, array(0.59797029136999991) * angstrom, array(0.23812976205) * angstrom)
  - thermal_conductivity: 35.0 W/(m*K)
  - ionization_energies: [array(715600.0) * J/mol, array(1450500.0) * J/mol, array(3081500.0) * J/mol, array(4083000.0) * J/mol, array(6640000.0) * J/mol]
  - vaporization: 178000.0 J/mol
  - atomic_number: 82
  - rigidity_modulus: 5.6 GPa
  - symbol: Pb
  - covalent_radius: 146.0 pm
  - fusion: 4770.0 J/mol
  - pettifor: 1.8
  - bulk_modulus: 46.0 GPa
  - poisson_ratio: 0.44 dimensionless
  - van_der_waals_radius: 202.0 pm
  - electron_affinity: 35100.0 J/mol
  - name: Lead
  - boiling_point: 2022.0 K
  - density: 11.34 g*cm**3
  - double_bond_radius: 135.0 pm
  - allred_rochow: 1.55
  - young_modulus: 16.0 GPa
  - thermal_expansion: 2.89e-05 1/K
  - atomic_radius: 180.0 pm
"""

Bi = Element(**{'mulliken_jaffe': 2.1499999999999999, 'single_bond_radius': array(151.0) * pm, 'pauling': 2.02, 'molar_volume': array(21.309999999999999) * cm**3/mol, 'bulk_modulus': array(31.0) * GPa, 'sound_velocity': array(1790.0) * m/s, 'sanderson': 2.3399999999999999, 'atomic_weight': 208.9804, 'triple_bond_radius': array(135.0) * pm, 'melting_point': array(544.39999999999998) * Kelvin, 'orbital_radii': (array(0.48684306907999997) * angstrom, array(0.56992389717299996) * angstrom, array(0.23177963506199997) * angstrom), 'thermal_conductivity': array(8.0) * Watt/(m*Kelvin), 'ionization_energies': [array(703000.0) * J/mol, array(1610000.0) * J/mol, array(2466000.0) * J/mol, array(4370000.0) * J/mol, array(5400000.0) * J/mol, array(8520000.0) * J/mol], 'vaporization': array(160000.0) * J/mol, 'atomic_number': 83, 'rigidity_modulus': array(12.0) * GPa, 'symbol': 'Bi', 'covalent_radius': array(148.0) * pm, 'fusion': array(10900.0) * J/mol, 'pettifor': 2.04, 'atomization': array(207000.0) * J/mol, 'poisson_ratio': array(0.33000000000000002) * dimensionless, 'electron_affinity': array(91200.0) * J/mol, 'name': 'Bismuth', 'boiling_point': array(1837.0) * Kelvin, 'density': array(9.7799999999999994) * g*cm**3, 'double_bond_radius': array(141.0) * pm, 'allred_rochow': 1.6699999999999999, 'young_modulus': array(32.0) * GPa, 'thermal_expansion': array(1.34e-05) * 1/Kelvin, 'atomic_radius': array(160.0) * pm})
"""
Bismuth elemental data.

  - mulliken_jaffe: 2.15
  - single_bond_radius: 151.0 pm
  - pauling: 2.02
  - molar_volume: 21.31 cm**3/mol
  - atomization: 207000.0 J/mol
  - sound_velocity: 1790.0 m/s
  - sanderson: 2.34
  - atomic_weight: 208.9804
  - triple_bond_radius: 135.0 pm
  - melting_point: 544.4 K
  - orbital_radii: (array(0.48684306907999997) * angstrom, array(0.56992389717299996) * angstrom, array(0.23177963506199997) * angstrom)
  - thermal_conductivity: 8.0 W/(m*K)
  - ionization_energies: [array(703000.0) * J/mol, array(1610000.0) * J/mol, array(2466000.0) * J/mol, array(4370000.0) * J/mol, array(5400000.0) * J/mol, array(8520000.0) * J/mol]
  - vaporization: 160000.0 J/mol
  - atomic_number: 83
  - rigidity_modulus: 12.0 GPa
  - symbol: Bi
  - covalent_radius: 148.0 pm
  - fusion: 10900.0 J/mol
  - pettifor: 2.04
  - bulk_modulus: 31.0 GPa
  - poisson_ratio: 0.33 dimensionless
  - electron_affinity: 91200.0 J/mol
  - name: Bismuth
  - boiling_point: 1837.0 K
  - density: 9.78 g*cm**3
  - double_bond_radius: 141.0 pm
  - allred_rochow: 1.67
  - young_modulus: 32.0 GPa
  - thermal_expansion: 1.34e-05 1/K
  - atomic_radius: 160.0 pm
"""

Po = Element(**{'mulliken_jaffe': 2.48, 'single_bond_radius': array(145.0) * pm, 'pauling': 2.0, 'molar_volume': array(22.969999999999999) * cm**3/mol, 'atomization': array(142000.0) * J/mol, 'atomic_weight': 209.0, 'triple_bond_radius': array(129.0) * pm, 'melting_point': array(527.0) * Kelvin, 'orbital_radii': (array(0.46567597911999997) * angstrom, array(0.53976079397999999) * angstrom, array(0.22490033082499997) * angstrom), 'thermal_conductivity': array(20.0) * Watt/(m*Kelvin), 'ionization_energies': [array(812100.0) * J/mol], 'vaporization': array(100000.0) * J/mol, 'atomic_number': 84, 'symbol': 'Po', 'covalent_radius': array(140.0) * pm, 'fusion': array(13000.0) * J/mol, 'pettifor': 2.2799999999999998, 'electron_affinity': array(183300.0) * J/mol, 'name': 'Polonium', 'boiling_point': array(1235.0) * Kelvin, 'density': array(9.1959999999999997) * g*cm**3, 'double_bond_radius': array(135.0) * pm, 'allred_rochow': 1.76, 'atomic_radius': array(190.0) * pm})
"""
Polonium elemental data.

  - mulliken_jaffe: 2.48
  - single_bond_radius: 145.0 pm
  - pauling: 2.0
  - molar_volume: 22.97 cm**3/mol
  - atomization: 142000.0 J/mol
  - atomic_weight: 209.0
  - triple_bond_radius: 129.0 pm
  - melting_point: 527.0 K
  - orbital_radii: (array(0.46567597911999997) * angstrom, array(0.53976079397999999) * angstrom, array(0.22490033082499997) * angstrom)
  - thermal_conductivity: 20.0 W/(m*K)
  - ionization_energies: [array(812100.0) * J/mol]
  - vaporization: 100000.0 J/mol
  - atomic_number: 84
  - symbol: Po
  - covalent_radius: 140.0 pm
  - fusion: 13000.0 J/mol
  - pettifor: 2.28
  - electron_affinity: 183300.0 J/mol
  - name: Polonium
  - boiling_point: 1235.0 K
  - density: 9.196 g*cm**3
  - double_bond_radius: 135.0 pm
  - allred_rochow: 1.76
  - atomic_radius: 190.0 pm
"""

At = Element(**{'mulliken_jaffe': 2.8500000000000001, 'electron_affinity': array(270100.0) * J/mol, 'single_bond_radius': array(147.0) * pm, 'pauling': 2.2000000000000002, 'pettifor': 2.52, 'thermal_conductivity': array(2.0) * Watt/(m*Kelvin), 'atomic_number': 85, 'double_bond_radius': array(138.0) * pm, 'symbol': 'At', 'covalent_radius': array(150.0) * pm, 'ionization_energies': [array(920000.0) * J/mol], 'fusion': array(6000.0) * J/mol, 'atomic_weight': 210.0, 'allred_rochow': 1.8999999999999999, 'triple_bond_radius': array(138.0) * pm, 'melting_point': array(575.0) * Kelvin, 'vaporization': array(40000.0) * J/mol, 'orbital_radii': (array(0.44980066164999993) * angstrom, array(0.51859370401999993) * angstrom, array(0.25135919327499995) * angstrom), 'name': 'Astatine'})
"""
Astatine elemental data.

  - mulliken_jaffe: 2.85
  - single_bond_radius: 147.0 pm
  - pauling: 2.2
  - atomic_weight: 210.0
  - triple_bond_radius: 138.0 pm
  - melting_point: 575.0 K
  - orbital_radii: (array(0.44980066164999993) * angstrom, array(0.51859370401999993) * angstrom, array(0.25135919327499995) * angstrom)
  - thermal_conductivity: 2.0 W/(m*K)
  - ionization_energies: [array(920000.0) * J/mol]
  - vaporization: 40000.0 J/mol
  - atomic_number: 85
  - symbol: At
  - covalent_radius: 150.0 pm
  - fusion: 6000.0 J/mol
  - pettifor: 2.52
  - electron_affinity: 270100.0 J/mol
  - name: Astatine
  - double_bond_radius: 138.0 pm
  - allred_rochow: 1.9
"""

Rn = Element(**{'mulliken_jaffe': 2.5899999999999999, 'electron_affinity': array(0.0) * J/mol, 'single_bond_radius': array(142.0) * pm, 'name': 'Radon', 'thermal_conductivity': array(0.0036099999999999999) * Watt/(m*Kelvin), 'atomic_number': 86, 'double_bond_radius': array(145.0) * pm, 'molar_volume': array(50.5) * cm**3/mol, 'covalent_radius': array(150.0) * pm, 'atomization': array(0.0) * J/mol, 'boiling_point': array(211.30000000000001) * Kelvin, 'fusion': array(3000.0) * J/mol, 'atomic_weight': 222.0, 'critical_temperature': array(377.0) * Kelvin, 'ionization_energies': [array(1037000.0) * J/mol], 'triple_bond_radius': array(133.0) * pm, 'melting_point': array(202.0) * Kelvin, 'vaporization': array(17000.0) * J/mol, 'orbital_radii': (array(0.44450888915999998) * angstrom, array(0.49742661405999994) * angstrom, array(0.214316785845) * angstrom), 'symbol': 'Rn'})
"""
Radon elemental data.

  - mulliken_jaffe: 2.59
  - single_bond_radius: 142.0 pm
  - molar_volume: 50.5 cm**3/mol
  - atomization: 0.0 J/mol
  - atomic_weight: 222.0
  - critical_temperature: 377.0 K
  - triple_bond_radius: 133.0 pm
  - melting_point: 202.0 K
  - orbital_radii: (array(0.44450888915999998) * angstrom, array(0.49742661405999994) * angstrom, array(0.214316785845) * angstrom)
  - thermal_conductivity: 0.00361 W/(m*K)
  - ionization_energies: [array(1037000.0) * J/mol]
  - vaporization: 17000.0 J/mol
  - atomic_number: 86
  - symbol: Rn
  - covalent_radius: 150.0 pm
  - fusion: 3000.0 J/mol
  - electron_affinity: 0.0 J/mol
  - name: Radon
  - boiling_point: 211.3 K
  - double_bond_radius: 145.0 pm
"""

Fr = Element(**{'mulliken_jaffe': 0.68000000000000005, 'single_bond_radius': array(223.0) * pm, 'pauling': 0.69999999999999996, 'atomic_number': 87, 'double_bond_radius': array(218.0) * pm, 'symbol': 'Fr', 'covalent_radius': array(260.0) * pm, 'atomization': array(64000.0) * J/mol, 'allred_rochow': 0.85999999999999999, 'fusion': array(2000.0) * J/mol, 'atomic_weight': 223.0, 'ionization_energies': [array(380000.0) * J/mol], 'melting_point': array(300.0) * Kelvin, 'vaporization': array(65000.0) * J/mol, 'name': 'Francium'})
"""
Francium elemental data.

  - mulliken_jaffe: 0.68
  - single_bond_radius: 223.0 pm
  - pauling: 0.7
  - atomization: 64000.0 J/mol
  - atomic_weight: 223.0
  - melting_point: 300.0 K
  - ionization_energies: [array(380000.0) * J/mol]
  - vaporization: 65000.0 J/mol
  - atomic_number: 87
  - symbol: Fr
  - covalent_radius: 260.0 pm
  - fusion: 2000.0 J/mol
  - name: Francium
  - double_bond_radius: 218.0 pm
  - allred_rochow: 0.86
"""

Ra = Element(**{'mulliken_jaffe': 0.92000000000000004, 'single_bond_radius': array(201.0) * pm, 'pauling': 0.90000000000000002, 'thermal_conductivity': array(19.0) * Watt/(m*Kelvin), 'atomic_number': 88, 'double_bond_radius': array(173.0) * pm, 'molar_volume': array(41.090000000000003) * cm**3/mol, 'density': array(5.0) * g*cm**3, 'covalent_radius': array(221.0) * pm, 'atomization': array(159000.0) * J/mol, 'atomic_radius': array(215.0) * pm, 'boiling_point': array(2010.0) * Kelvin, 'fusion': array(8000.0) * J/mol, 'atomic_weight': 226.0, 'ionization_energies': [array(509300.0) * J/mol, array(979000.0) * J/mol], 'triple_bond_radius': array(159.0) * pm, 'melting_point': array(973.0) * Kelvin, 'vaporization': array(125000.0) * J/mol, 'symbol': 'Ra', 'allred_rochow': 0.96999999999999997, 'name': 'Radium'})
"""
Radium elemental data.

  - mulliken_jaffe: 0.92
  - single_bond_radius: 201.0 pm
  - pauling: 0.9
  - molar_volume: 41.09 cm**3/mol
  - atomization: 159000.0 J/mol
  - atomic_weight: 226.0
  - triple_bond_radius: 159.0 pm
  - melting_point: 973.0 K
  - thermal_conductivity: 19.0 W/(m*K)
  - ionization_energies: [array(509300.0) * J/mol, array(979000.0) * J/mol]
  - vaporization: 125000.0 J/mol
  - atomic_number: 88
  - symbol: Ra
  - covalent_radius: 221.0 pm
  - fusion: 8000.0 J/mol
  - name: Radium
  - boiling_point: 2010.0 K
  - density: 5.0 g*cm**3
  - double_bond_radius: 173.0 pm
  - allred_rochow: 0.97
  - atomic_radius: 215.0 pm
"""

Ac = Element(**{'single_bond_radius': array(186.0) * pm, 'pauling': 1.1000000000000001, 'thermal_conductivity': array(12.0) * Watt/(m*Kelvin), 'atomic_number': 89, 'double_bond_radius': array(153.0) * pm, 'molar_volume': array(22.550000000000001) * cm**3/mol, 'density': array(10.07) * g*cm**3, 'covalent_radius': array(215.0) * pm, 'atomization': array(406000.0) * J/mol, 'atomic_radius': array(195.0) * pm, 'boiling_point': array(3573.0) * Kelvin, 'fusion': array(14000.0) * J/mol, 'atomic_weight': 227.0, 'ionization_energies': [array(499000.0) * J/mol, array(1170000.0) * J/mol], 'triple_bond_radius': array(140.0) * pm, 'melting_point': array(1323.0) * Kelvin, 'vaporization': array(400000.0) * J/mol, 'symbol': 'Ac', 'allred_rochow': 1.0, 'name': 'Actinium'})
"""
Actinium elemental data.

  - single_bond_radius: 186.0 pm
  - pauling: 1.1
  - molar_volume: 22.55 cm**3/mol
  - atomization: 406000.0 J/mol
  - atomic_weight: 227.0
  - triple_bond_radius: 140.0 pm
  - melting_point: 1323.0 K
  - thermal_conductivity: 12.0 W/(m*K)
  - ionization_energies: [array(499000.0) * J/mol, array(1170000.0) * J/mol]
  - vaporization: 400000.0 J/mol
  - atomic_number: 89
  - symbol: Ac
  - covalent_radius: 215.0 pm
  - fusion: 14000.0 J/mol
  - name: Actinium
  - boiling_point: 3573.0 K
  - density: 10.07 g*cm**3
  - double_bond_radius: 153.0 pm
  - allred_rochow: 1.0
  - atomic_radius: 195.0 pm
"""

Th = Element(**{'single_bond_radius': array(175.0) * pm, 'pauling': 1.3, 'molar_volume': array(19.800000000000001) * cm**3/mol, 'bulk_modulus': array(54.0) * GPa, 'sound_velocity': array(2490.0) * m/s, 'atomic_weight': 232.03806, 'triple_bond_radius': array(136.0) * pm, 'melting_point': array(2115.0) * Kelvin, 'thermal_conductivity': array(54.0) * Watt/(m*Kelvin), 'ionization_energies': [array(587000.0) * J/mol, array(1110000.0) * J/mol, array(1930000.0) * J/mol, array(2780000.0) * J/mol], 'vaporization': array(530000.0) * J/mol, 'atomic_number': 90, 'rigidity_modulus': array(31.0) * GPa, 'symbol': 'Th', 'covalent_radius': array(206.0) * pm, 'fusion': array(16000.0) * J/mol, 'atomization': array(598000.0) * J/mol, 'poisson_ratio': array(0.27000000000000002) * dimensionless, 'name': 'Thorium', 'boiling_point': array(5093.0) * Kelvin, 'density': array(11.724) * g*cm**3, 'double_bond_radius': array(143.0) * pm, 'allred_rochow': 1.1100000000000001, 'young_modulus': array(79.0) * GPa, 'thermal_expansion': array(1.1e-05) * 1/Kelvin, 'atomic_radius': array(180.0) * pm})
"""
Thorium elemental data.

  - single_bond_radius: 175.0 pm
  - pauling: 1.3
  - molar_volume: 19.8 cm**3/mol
  - atomization: 598000.0 J/mol
  - sound_velocity: 2490.0 m/s
  - atomic_weight: 232.03806
  - triple_bond_radius: 136.0 pm
  - melting_point: 2115.0 K
  - thermal_conductivity: 54.0 W/(m*K)
  - ionization_energies: [array(587000.0) * J/mol, array(1110000.0) * J/mol, array(1930000.0) * J/mol, array(2780000.0) * J/mol]
  - vaporization: 530000.0 J/mol
  - atomic_number: 90
  - rigidity_modulus: 31.0 GPa
  - symbol: Th
  - covalent_radius: 206.0 pm
  - fusion: 16000.0 J/mol
  - bulk_modulus: 54.0 GPa
  - poisson_ratio: 0.27 dimensionless
  - name: Thorium
  - boiling_point: 5093.0 K
  - density: 11.724 g*cm**3
  - double_bond_radius: 143.0 pm
  - allred_rochow: 1.11
  - young_modulus: 79.0 GPa
  - thermal_expansion: 1.1e-05 1/K
  - atomic_radius: 180.0 pm
"""

Pa = Element(**{'single_bond_radius': array(169.0) * pm, 'pauling': 1.5, 'thermal_conductivity': array(47.0) * Watt/(m*Kelvin), 'atomic_number': 91, 'double_bond_radius': array(138.0) * pm, 'molar_volume': array(15.18) * cm**3/mol, 'density': array(15.369999999999999) * g*cm**3, 'covalent_radius': array(200.0) * pm, 'atomization': array(607000.0) * J/mol, 'atomic_radius': array(180.0) * pm, 'allred_rochow': 1.1399999999999999, 'fusion': array(15000.0) * J/mol, 'atomic_weight': 231.03587999999999, 'ionization_energies': [array(568000.0) * J/mol], 'triple_bond_radius': array(129.0) * pm, 'melting_point': array(1841.0) * Kelvin, 'vaporization': array(470000.0) * J/mol, 'symbol': 'Pa', 'name': 'Protactinium'})
"""
Protactinium elemental data.

  - single_bond_radius: 169.0 pm
  - pauling: 1.5
  - molar_volume: 15.18 cm**3/mol
  - atomization: 607000.0 J/mol
  - atomic_weight: 231.03588
  - triple_bond_radius: 129.0 pm
  - melting_point: 1841.0 K
  - thermal_conductivity: 47.0 W/(m*K)
  - ionization_energies: [array(568000.0) * J/mol]
  - vaporization: 470000.0 J/mol
  - atomic_number: 91
  - symbol: Pa
  - covalent_radius: 200.0 pm
  - fusion: 15000.0 J/mol
  - name: Protactinium
  - density: 15.37 g*cm**3
  - double_bond_radius: 138.0 pm
  - allred_rochow: 1.14
  - atomic_radius: 180.0 pm
"""

U = Element(**{'single_bond_radius': array(170.0) * pm, 'pauling': 1.3799999999999999, 'molar_volume': array(12.49) * cm**3/mol, 'bulk_modulus': array(100.0) * GPa, 'sound_velocity': array(3155.0) * m/s, 'atomic_weight': 238.02891, 'triple_bond_radius': array(118.0) * pm, 'melting_point': array(1405.3) * Kelvin, 'thermal_conductivity': array(27.0) * Watt/(m*Kelvin), 'ionization_energies': [array(597600.0) * J/mol, array(1420000.0) * J/mol], 'vaporization': array(420000.0) * J/mol, 'atomic_number': 92, 'rigidity_modulus': array(111.0) * GPa, 'symbol': 'U', 'covalent_radius': array(196.0) * pm, 'fusion': array(14000.0) * J/mol, 'atomization': array(536000.0) * J/mol, 'poisson_ratio': array(0.23000000000000001) * dimensionless, 'van_der_waals_radius': array(186.0) * pm, 'name': 'Uranium', 'boiling_point': array(4200.0) * Kelvin, 'density': array(19.050000000000001) * g*cm**3, 'double_bond_radius': array(134.0) * pm, 'allred_rochow': 1.22, 'young_modulus': array(208.0) * GPa, 'thermal_expansion': array(1.3899999999999999e-05) * 1/Kelvin, 'atomic_radius': array(175.0) * pm})
"""
Uranium elemental data.

  - single_bond_radius: 170.0 pm
  - pauling: 1.38
  - molar_volume: 12.49 cm**3/mol
  - atomization: 536000.0 J/mol
  - sound_velocity: 3155.0 m/s
  - atomic_weight: 238.02891
  - triple_bond_radius: 118.0 pm
  - melting_point: 1405.3 K
  - thermal_conductivity: 27.0 W/(m*K)
  - ionization_energies: [array(597600.0) * J/mol, array(1420000.0) * J/mol]
  - vaporization: 420000.0 J/mol
  - atomic_number: 92
  - rigidity_modulus: 111.0 GPa
  - symbol: U
  - covalent_radius: 196.0 pm
  - fusion: 14000.0 J/mol
  - bulk_modulus: 100.0 GPa
  - poisson_ratio: 0.23 dimensionless
  - van_der_waals_radius: 186.0 pm
  - name: Uranium
  - boiling_point: 4200.0 K
  - density: 19.05 g*cm**3
  - double_bond_radius: 134.0 pm
  - allred_rochow: 1.22
  - young_modulus: 208.0 GPa
  - thermal_expansion: 1.39e-05 1/K
  - atomic_radius: 175.0 pm
"""

Np = Element(**{'single_bond_radius': array(171.0) * pm, 'pauling': 1.3600000000000001, 'thermal_conductivity': array(6.0) * Watt/(m*Kelvin), 'atomic_number': 93, 'double_bond_radius': array(136.0) * pm, 'molar_volume': array(11.59) * cm**3/mol, 'density': array(20.449999999999999) * g*cm**3, 'covalent_radius': array(190.0) * pm, 'ionization_energies': [array(604500.0) * J/mol], 'atomic_radius': array(175.0) * pm, 'boiling_point': array(4273.0) * Kelvin, 'fusion': array(10000.0) * J/mol, 'atomic_weight': 237.0, 'allred_rochow': 1.22, 'triple_bond_radius': array(116.0) * pm, 'melting_point': array(910.0) * Kelvin, 'vaporization': array(335000.0) * J/mol, 'symbol': 'Np', 'name': 'Neptunium'})
"""
Neptunium elemental data.

  - single_bond_radius: 171.0 pm
  - pauling: 1.36
  - molar_volume: 11.59 cm**3/mol
  - atomic_weight: 237.0
  - triple_bond_radius: 116.0 pm
  - melting_point: 910.0 K
  - thermal_conductivity: 6.0 W/(m*K)
  - ionization_energies: [array(604500.0) * J/mol]
  - vaporization: 335000.0 J/mol
  - atomic_number: 93
  - symbol: Np
  - covalent_radius: 190.0 pm
  - fusion: 10000.0 J/mol
  - name: Neptunium
  - boiling_point: 4273.0 K
  - density: 20.45 g*cm**3
  - double_bond_radius: 136.0 pm
  - allred_rochow: 1.22
  - atomic_radius: 175.0 pm
"""

Pu = Element(**{'atomic_radius': array(175.0) * pm, 'single_bond_radius': array(172.0) * pm, 'pauling': 1.28, 'thermal_conductivity': array(6.0) * Watt/(m*Kelvin), 'atomic_number': 94, 'rigidity_modulus': array(43.0) * GPa, 'molar_volume': array(12.289999999999999) * cm**3/mol, 'boiling_point': array(3503.0) * Kelvin, 'covalent_radius': array(187.0) * pm, 'ionization_energies': [array(584700.0) * J/mol], 'sound_velocity': array(2260.0) * m/s, 'young_modulus': array(96.0) * GPa, 'poisson_ratio': array(0.20999999999999999) * dimensionless, 'double_bond_radius': array(135.0) * pm, 'atomic_weight': 244.0, 'allred_rochow': 1.22, 'melting_point': array(912.5) * Kelvin, 'vaporization': array(325000.0) * J/mol, 'density': array(19.815999999999999) * g*cm**3, 'symbol': 'Pu', 'name': 'Plutonium'})
"""
Plutonium elemental data.

  - single_bond_radius: 172.0 pm
  - pauling: 1.28
  - molar_volume: 12.29 cm**3/mol
  - sound_velocity: 2260.0 m/s
  - atomic_weight: 244.0
  - melting_point: 912.5 K
  - thermal_conductivity: 6.0 W/(m*K)
  - ionization_energies: [array(584700.0) * J/mol]
  - vaporization: 325000.0 J/mol
  - atomic_number: 94
  - rigidity_modulus: 43.0 GPa
  - symbol: Pu
  - covalent_radius: 187.0 pm
  - poisson_ratio: 0.21 dimensionless
  - name: Plutonium
  - boiling_point: 3503.0 K
  - density: 19.816 g*cm**3
  - double_bond_radius: 135.0 pm
  - allred_rochow: 1.22
  - young_modulus: 96.0 GPa
  - atomic_radius: 175.0 pm
"""

Am = Element(**{'single_bond_radius': array(166.0) * pm, 'pauling': 1.3, 'thermal_conductivity': array(10.0) * Watt/(m*Kelvin), 'atomic_number': 95, 'double_bond_radius': array(135.0) * pm, 'molar_volume': array(17.629999999999999) * cm**3/mol, 'covalent_radius': array(180.0) * pm, 'ionization_energies': [array(578000.0) * J/mol], 'atomic_radius': array(175.0) * pm, 'boiling_point': array(2880.0) * Kelvin, 'atomic_weight': 243.0, 'allred_rochow': 1.2, 'melting_point': array(1449.0) * Kelvin, 'symbol': 'Am', 'name': 'Americium'})
"""
Americium elemental data.

  - single_bond_radius: 166.0 pm
  - pauling: 1.3
  - molar_volume: 17.63 cm**3/mol
  - atomic_weight: 243.0
  - melting_point: 1449.0 K
  - thermal_conductivity: 10.0 W/(m*K)
  - ionization_energies: [array(578000.0) * J/mol]
  - atomic_number: 95
  - symbol: Am
  - covalent_radius: 180.0 pm
  - name: Americium
  - boiling_point: 2880.0 K
  - double_bond_radius: 135.0 pm
  - allred_rochow: 1.2
  - atomic_radius: 175.0 pm
"""

Cm = Element(**{'single_bond_radius': array(166.0) * pm, 'pauling': 1.3, 'thermal_conductivity': array(8.8000000000000007) * Watt/(m*Kelvin), 'atomic_number': 96, 'double_bond_radius': array(136.0) * pm, 'molar_volume': array(18.050000000000001) * cm**3/mol, 'density': array(13.51) * g*cm**3, 'covalent_radius': array(169.0) * pm, 'ionization_energies': [array(581000.0) * J/mol], 'boiling_point': array(3383.0) * Kelvin, 'atomic_weight': 247.0, 'allred_rochow': 1.2, 'melting_point': array(1613.0) * Kelvin, 'vaporization': array(320000.0) * J/mol, 'symbol': 'Cm', 'name': 'Curium'})
"""
Curium elemental data.

  - single_bond_radius: 166.0 pm
  - pauling: 1.3
  - molar_volume: 18.05 cm**3/mol
  - atomic_weight: 247.0
  - melting_point: 1613.0 K
  - thermal_conductivity: 8.8 W/(m*K)
  - ionization_energies: [array(581000.0) * J/mol]
  - vaporization: 320000.0 J/mol
  - atomic_number: 96
  - symbol: Cm
  - covalent_radius: 169.0 pm
  - name: Curium
  - boiling_point: 3383.0 K
  - density: 13.51 g*cm**3
  - double_bond_radius: 136.0 pm
  - allred_rochow: 1.2
"""

Bk = Element(**{'single_bond_radius': array(168.0) * pm, 'pauling': 1.3, 'thermal_conductivity': array(10.0) * Watt/(m*Kelvin), 'atomic_number': 97, 'molar_volume': array(16.84) * cm**3/mol, 'density': array(14.779999999999999) * g*cm**3, 'double_bond_radius': array(139.0) * pm, 'ionization_energies': [array(601000.0) * J/mol], 'atomic_weight': 247.0, 'allred_rochow': 1.2, 'melting_point': array(1259.0) * Kelvin, 'symbol': 'Bk', 'name': 'Berkelium'})
"""
Berkelium elemental data.

  - single_bond_radius: 168.0 pm
  - pauling: 1.3
  - molar_volume: 16.84 cm**3/mol
  - atomic_weight: 247.0
  - melting_point: 1259.0 K
  - thermal_conductivity: 10.0 W/(m*K)
  - ionization_energies: [array(601000.0) * J/mol]
  - atomic_number: 97
  - symbol: Bk
  - name: Berkelium
  - density: 14.78 g*cm**3
  - double_bond_radius: 139.0 pm
  - allred_rochow: 1.2
"""

Cf = Element(**{'single_bond_radius': array(168.0) * pm, 'pauling': 1.3, 'density': array(15.1) * g*cm**3, 'atomic_number': 98, 'molar_volume': array(16.5) * cm**3/mol, 'double_bond_radius': array(140.0) * pm, 'ionization_energies': [array(608000.0) * J/mol], 'atomic_weight': 251.0, 'allred_rochow': 1.2, 'melting_point': array(1173.0) * Kelvin, 'symbol': 'Cf', 'name': 'Californium'})
"""
Californium elemental data.

  - single_bond_radius: 168.0 pm
  - pauling: 1.3
  - molar_volume: 16.5 cm**3/mol
  - atomic_weight: 251.0
  - melting_point: 1173.0 K
  - ionization_energies: [array(608000.0) * J/mol]
  - atomic_number: 98
  - symbol: Cf
  - name: Californium
  - density: 15.1 g*cm**3
  - double_bond_radius: 140.0 pm
  - allred_rochow: 1.2
"""

Es = Element(**{'single_bond_radius': array(165.0) * pm, 'pauling': 1.3, 'atomic_number': 99, 'molar_volume': array(28.52) * cm**3/mol, 'double_bond_radius': array(140.0) * pm, 'ionization_energies': [array(619000.0) * J/mol], 'atomic_weight': 252.0, 'allred_rochow': 1.2, 'melting_point': array(1133.0) * Kelvin, 'symbol': 'Es', 'name': 'Einsteinium'})
"""
Einsteinium elemental data.

  - single_bond_radius: 165.0 pm
  - pauling: 1.3
  - molar_volume: 28.52 cm**3/mol
  - atomic_weight: 252.0
  - melting_point: 1133.0 K
  - ionization_energies: [array(619000.0) * J/mol]
  - atomic_number: 99
  - symbol: Es
  - name: Einsteinium
  - double_bond_radius: 140.0 pm
  - allred_rochow: 1.2
"""

Fm = Element(**{'single_bond_radius': array(167.0) * pm, 'pauling': 1.3, 'atomic_number': 100, 'symbol': 'Fm', 'ionization_energies': [array(627000.0) * J/mol], 'atomic_weight': 257.0, 'allred_rochow': 1.2, 'melting_point': array(1800.0) * Kelvin, 'name': 'Fermium'})
"""
Fermium elemental data.

  - single_bond_radius: 167.0 pm
  - pauling: 1.3
  - atomic_weight: 257.0
  - melting_point: 1800.0 K
  - ionization_energies: [array(627000.0) * J/mol]
  - atomic_number: 100
  - symbol: Fm
  - name: Fermium
  - allred_rochow: 1.2
"""

Md = Element(**{'single_bond_radius': array(173.0) * pm, 'pauling': 1.3, 'atomic_number': 101, 'symbol': 'Md', 'double_bond_radius': array(139.0) * pm, 'ionization_energies': [array(635000.0) * J/mol], 'atomic_weight': 258.0, 'allred_rochow': 1.2, 'melting_point': array(1100.0) * Kelvin, 'name': 'Mendelevium'})
"""
Mendelevium elemental data.

  - single_bond_radius: 173.0 pm
  - pauling: 1.3
  - atomic_weight: 258.0
  - melting_point: 1100.0 K
  - ionization_energies: [array(635000.0) * J/mol]
  - atomic_number: 101
  - symbol: Md
  - name: Mendelevium
  - double_bond_radius: 139.0 pm
  - allred_rochow: 1.2
"""

No = Element(**{'single_bond_radius': array(176.0) * pm, 'pauling': 1.3, 'atomic_number': 102, 'symbol': 'No', 'ionization_energies': [array(642000.0) * J/mol], 'atomic_weight': 259.0, 'allred_rochow': 1.2, 'melting_point': array(1100.0) * Kelvin, 'name': 'Nobelium'})
"""
Nobelium elemental data.

  - single_bond_radius: 176.0 pm
  - pauling: 1.3
  - atomic_weight: 259.0
  - melting_point: 1100.0 K
  - ionization_energies: [array(642000.0) * J/mol]
  - atomic_number: 102
  - symbol: No
  - name: Nobelium
  - allred_rochow: 1.2
"""

Lr = Element(**{'single_bond_radius': array(161.0) * pm, 'name': 'Lawrencium', 'atomic_number': 103, 'symbol': 'Lr', 'double_bond_radius': array(141.0) * pm, 'atomic_weight': 262.0, 'melting_point': array(1900.0) * Kelvin})
"""
Lawrencium elemental data.

  - single_bond_radius: 161.0 pm
  - atomic_weight: 262.0
  - melting_point: 1900.0 K
  - atomic_number: 103
  - symbol: Lr
  - name: Lawrencium
  - double_bond_radius: 141.0 pm
"""

Rf = Element(**{'single_bond_radius': array(157.0) * pm, 'name': 'Rutherfordium', 'atomic_number': 104, 'symbol': 'Rf', 'double_bond_radius': array(140.0) * pm, 'atomic_weight': 267.0, 'triple_bond_radius': array(131.0) * pm})
"""
Rutherfordium elemental data.

  - single_bond_radius: 157.0 pm
  - atomic_weight: 267.0
  - triple_bond_radius: 131.0 pm
  - atomic_number: 104
  - symbol: Rf
  - name: Rutherfordium
  - double_bond_radius: 140.0 pm
"""

Db = Element(**{'single_bond_radius': array(149.0) * pm, 'name': 'Dubnium', 'atomic_number': 105, 'symbol': 'Db', 'double_bond_radius': array(136.0) * pm, 'atomic_weight': 268.0, 'triple_bond_radius': array(126.0) * pm})
"""
Dubnium elemental data.

  - single_bond_radius: 149.0 pm
  - atomic_weight: 268.0
  - triple_bond_radius: 126.0 pm
  - atomic_number: 105
  - symbol: Db
  - name: Dubnium
  - double_bond_radius: 136.0 pm
"""

Sg = Element(**{'single_bond_radius': array(143.0) * pm, 'name': 'Seaborgium', 'atomic_number': 106, 'symbol': 'Sg', 'double_bond_radius': array(128.0) * pm, 'atomic_weight': 271.0, 'triple_bond_radius': array(121.0) * pm})
"""
Seaborgium elemental data.

  - single_bond_radius: 143.0 pm
  - atomic_weight: 271.0
  - triple_bond_radius: 121.0 pm
  - atomic_number: 106
  - symbol: Sg
  - name: Seaborgium
  - double_bond_radius: 128.0 pm
"""

Bh = Element(**{'single_bond_radius': array(141.0) * pm, 'name': 'Bohrium', 'atomic_number': 107, 'symbol': 'Bh', 'double_bond_radius': array(128.0) * pm, 'atomic_weight': 272.0, 'triple_bond_radius': array(119.0) * pm})
"""
Bohrium elemental data.

  - single_bond_radius: 141.0 pm
  - atomic_weight: 272.0
  - triple_bond_radius: 119.0 pm
  - atomic_number: 107
  - symbol: Bh
  - name: Bohrium
  - double_bond_radius: 128.0 pm
"""

Hs = Element(**{'single_bond_radius': array(134.0) * pm, 'name': 'Hassium', 'atomic_number': 108, 'symbol': 'Hs', 'double_bond_radius': array(125.0) * pm, 'atomic_weight': 270.0, 'triple_bond_radius': array(118.0) * pm})
"""
Hassium elemental data.

  - single_bond_radius: 134.0 pm
  - atomic_weight: 270.0
  - triple_bond_radius: 118.0 pm
  - atomic_number: 108
  - symbol: Hs
  - name: Hassium
  - double_bond_radius: 125.0 pm
"""

Mt = Element(**{'single_bond_radius': array(129.0) * pm, 'name': 'Meitnerium', 'atomic_number': 109, 'symbol': 'Mt', 'double_bond_radius': array(125.0) * pm, 'atomic_weight': 276.0, 'triple_bond_radius': array(113.0) * pm})
"""
Meitnerium elemental data.

  - single_bond_radius: 129.0 pm
  - atomic_weight: 276.0
  - triple_bond_radius: 113.0 pm
  - atomic_number: 109
  - symbol: Mt
  - name: Meitnerium
  - double_bond_radius: 125.0 pm
"""


symbols = [ 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
            'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
            'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
            'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
            'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt']

__all__ = list(symbols)
__all__.extend(['Element', 'iterate', 'find'])

def iterate():
  """ Iterates through all elements. """
  for name in symbols:
    yield globals()[name] 

def find(**kwargs):
  """ Find element according to different criteria. 
  
      :param int atomic_number:
        Find specie according to atomic number
      :param str symbol:
        Find specie according to symbol.
      :param str name:
        Find specie according to name. If the input is a string of two
        characters or less, tries and finds according to symbol.
  """
  from ..error import input, ValueError
  if len(kwargs) != 1:
    raise input('Expected one and only one keyword argument.')

  if 'atomic_number' in kwargs:
    n = int(kwargs['atomic_number'])
    for specie in iterate():
      if specie.atomic_number == n: return specie
    raise ValueError( 'Could not find specie with atomic number {0}.'          \
                      .format(n))
  if 'symbol' in kwargs: 
    symbol = str(kwargs['symbol']).replace('\n', '').rstrip().lstrip()
    if len(symbol) == 1: symbol = symbol.upper()
    elif len(symbol) == 2: symbol = symbol[0].upper() + symbol[1].lower()
    else: raise input('Symbol cannot be longuer than two characters.')
    if symbol not in globals():
      raise ValueError('Unknown specie with symbol {0}.'.format(symbol))
    return globals()[symbol]
  if 'name' in kwargs: 
    name = str(kwargs['name']).replace('\n', '').rstrip().lstrip()
    if len(name) == 0: raise input('Name string is empty.')
    elif len(name) <= 2: return find(symbol=name)
    name = name.lower()
    for specie in iterate(): 
      if name == specie.name.lower(): return specie
    raise ValueError('Unknown specie named {0}.'.format(name))
