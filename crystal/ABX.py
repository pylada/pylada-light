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

""" Defines ABX Lattices. """
__docformat__ = "restructuredtext en"

def s13():
  """ s13 lattice """
  from pylada.crystal import Structure
  return Structure( 14.462, 7.231, -2.12564,\
                    0, 2.785, 0,\
                    0, 0, 9.23657,\
                    scale=1.1, name='s13' )\
           .add_atom(12.2089, 1.30839, 5.17802, 'A')\
           .add_atom(13.5267, 1.30839, 8.67683, 'A')\
           .add_atom(7.35851, 1.47661, 4.05855, 'A')\
           .add_atom(6.04068, 1.47661, 0.559736, 'A')\
           .add_atom(9.78368, 1.3925, 4.61828, 'A')\
           .add_atom(3.6155, 1.3925, 0, 'A')\
           .add_atom(10.1658, 1.31619, 8.0469, 'B')\
           .add_atom(15.5698, 1.31619, 5.80795, 'B')\
           .add_atom(9.40159, 1.46881, 1.18967, 'B')\
           .add_atom(3.9976, 1.46881, 3.42861, 'B')\
           .add_atom(5.63677, 1.22429, 6.92742, 'B')\
           .add_atom(13.9306, 1.56071, 2.30914, 'B')\
           .add_atom(10.8767, 2.56777, 5.62507, 'X')\
           .add_atom(14.8588, 2.56777, 8.22978, 'X')\
           .add_atom(8.69066, 0.21723, 3.6115, 'X')\
           .add_atom(4.70852, 0.21723, 1.00679, 'X')\
           .add_atom(-1.06282, 0, 4.61828, 'X')\
           .add_atom(0, 0, 0, 'X')

def s25():
  """ s25 lattice """
  from pylada.crystal import Structure
  return Structure( 2.0905, 4.181, 0,\
                    2.0905, 0, 4.181,\
                    7.0345, 0, 0,\
                    scale=1.1, name='s25' )\
           .add_atom(2.0905, 4.181, 4.71312, 'A')\
           .add_atom(2.0905, 2.0905, 1.19587, 'A')\
           .add_atom(4.181, 2.0905, 3.51725, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(2.0905, 4.181, 2.37766, 'X')\
           .add_atom(4.181, 4.181, 5.89491, 'X')

def s20():
  """ s20 lattice """
  from pylada.crystal import Structure
  return Structure( 4.25, -2.125, 0,\
                    0, 3.68061, 0,\
                    0, 0, 15.165,\
                    scale=1, name='s20' )\
           .add_atom(-2.12479e-08, 2.45374, 5.51248, 'A')\
           .add_atom(2.125, 1.22687, 9.65252, 'A')\
           .add_atom(2.125, 1.22687, 13.095, 'A')\
           .add_atom(-2.12479e-08, 2.45374, 2.07002, 'A')\
           .add_atom(0, 0, 7.5825, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(0, 0, 3.79125, 'B')\
           .add_atom(0, 0, 11.3737, 'B')\
           .add_atom(-2.12479e-08, 2.45374, 13.3391, 'X')\
           .add_atom(2.125, 1.22687, 1.82587, 'X')\
           .add_atom(2.125, 1.22687, 5.75663, 'X')\
           .add_atom(-2.12479e-08, 2.45374, 9.40837, 'X')

def s21():
  """ s21 lattice """
  from pylada.crystal import Structure
  return Structure( 4.263, 0, 0,\
                    0, 4.263, 0,\
                    0, 0, 6.179,\
                    scale=1.1, name='s21' )\
           .add_atom(2.1315, 2.1315, 3.0895, 'A')\
           .add_atom(0, 0, 3.0895, 'A')\
           .add_atom(2.1315, 2.1315, 0, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(0, 2.1315, 4.68368, 'X')\
           .add_atom(2.1315, 0, 1.49532, 'X')

def s1():
  """ s1 lattice """
  from pylada.crystal import Structure
  return Structure( 0, 3.1, 3.1,\
                    3.1, 0, 3.1,\
                    3.1, 3.1, 0,\
                    scale=1, name='s1' )\
           .add_atom(3.1, 3.1, 3.1, 'A')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(1.55, 1.55, 1.55, 'X')

def s34():
  """ s34 lattice """
  from pylada.crystal import Structure
  return Structure( 6.505, 0, -3.03547,\
                    0, 6.3833, 0,\
                    0, 0, 5.79401,\
                    scale=1, name='s34' )\
           .add_atom(1.74619, 2.2782, 0.970497, 'B')\
           .add_atom(3.24107, 5.46985, 1.92651, 'B')\
           .add_atom(1.72334, 4.1051, 4.82351, 'B')\
           .add_atom(0.228457, 0.91345, 3.8675, 'B')\
           .add_atom(-0.145727, 4.08084, 2.12698, 'B')\
           .add_atom(5.13299, 0.889194, 0.770024, 'B')\
           .add_atom(3.61526, 2.30246, 3.66703, 'B')\
           .add_atom(-1.66346, 5.49411, 5.02399, 'B')\
           .add_atom(0.901909, 6.38011, 1.63217, 'X')\
           .add_atom(4.08536, 3.18846, 1.26483, 'X')\
           .add_atom(2.56762, 0.00319165, 4.16184, 'X')\
           .add_atom(-0.615826, 3.19484, 4.52918, 'X')

def s6():
  """ s6 lattice """
  from pylada.crystal import Structure
  return Structure( 7.78, 0, 0,\
                    0, 4.56, 0,\
                    0, 0, 8.49,\
                    scale=1.1, name='s6' )\
           .add_atom(1.0892, 1.14, 3.76107, 'A')\
           .add_atom(4.9792, 1.14, 0.48393, 'A')\
           .add_atom(6.6908, 3.42, 4.72893, 'A')\
           .add_atom(2.8008, 3.42, 8.00607, 'A')\
           .add_atom(0.1167, 1.14, 6.93633, 'B')\
           .add_atom(4.0067, 1.14, 5.79867, 'B')\
           .add_atom(7.6633, 3.42, 1.55367, 'B')\
           .add_atom(3.7733, 3.42, 2.69133, 'B')\
           .add_atom(6.03728, 1.14, 3.3111, 'X')\
           .add_atom(2.14728, 1.14, 0.9339, 'X')\
           .add_atom(1.74272, 3.42, 5.1789, 'X')\
           .add_atom(5.63272, 3.42, 7.5561, 'X')

def s24():
  """ s24 lattice """
  from pylada.crystal import Structure
  return Structure( 2.1225, 4.245, 0,\
                    2.1225, 0, 4.245,\
                    8.153, 0, 0,\
                    scale=1, name='s24' )\
           .add_atom(2.1225, 2.1225, 2.69049, 'A')\
           .add_atom(4.245, 4.245, 5.46251, 'A')\
           .add_atom(2.1225, 0, 0, 'B')\
           .add_atom(0, 2.1225, 0, 'B')\
           .add_atom(2.1225, 2.1225, 5.82124, 'X')\
           .add_atom(4.245, 4.245, 2.33176, 'X')

def s29():
  """ s29 lattice """
  from pylada.crystal import Structure
  return Structure( 4.308, 0, 0,\
                    0, 13.912, 0,\
                    0, 0, 7.431,\
                    scale=1.1, name='s29' )\
           .add_atom(3.231, 8.93276, 6.85658, 'A')\
           .add_atom(3.231, 11.9352, 6.85658, 'A')\
           .add_atom(1.077, 4.97924, 0.574416, 'A')\
           .add_atom(1.077, 1.97676, 0.574416, 'A')\
           .add_atom(3.231, 1.32039, 4.32023, 'A')\
           .add_atom(3.231, 5.63561, 4.32023, 'A')\
           .add_atom(1.077, 12.5916, 3.11077, 'A')\
           .add_atom(1.077, 8.27639, 3.11077, 'A')\
           .add_atom(3.231, 6.97603, 1.55828, 'B')\
           .add_atom(3.231, 13.892, 1.55828, 'B')\
           .add_atom(1.077, 6.93597, 5.87272, 'B')\
           .add_atom(1.077, 0.0200333, 5.87272, 'B')\
           .add_atom(3.231, 3.478, 2.16688, 'B')\
           .add_atom(1.077, 10.434, 5.26412, 'B')\
           .add_atom(3.231, 10.434, 2.22484, 'B')\
           .add_atom(1.077, 3.478, 5.20616, 'B')\
           .add_atom(3.231, 8.39728, 4.39321, 'X')\
           .add_atom(3.231, 12.4707, 4.39321, 'X')\
           .add_atom(1.077, 5.51472, 3.03779, 'X')\
           .add_atom(1.077, 1.44128, 3.03779, 'X')\
           .add_atom(3.231, 2.18001, 6.76072, 'X')\
           .add_atom(3.231, 4.77599, 6.76072, 'X')\
           .add_atom(1.077, 11.732, 0.670276, 'X')\
           .add_atom(1.077, 9.13601, 0.670276, 'X')

def s11():
  """ s11 lattice """
  from pylada.crystal import Structure
  return Structure( 9.32, 0, 4.66,\
                    0, 9.32, 4.66,\
                    0, 0, 2.73,\
                    scale=1.15, name='s11' )\
           .add_atom(4.66, 6.37488, 1.365, 'A')\
           .add_atom(4.66, 2.94512, 1.365, 'A')\
           .add_atom(7.60512, 9.32, 1.365, 'A')\
           .add_atom(11.0349, 9.32, 1.365, 'A')\
           .add_atom(7.94996, 3.262, 1.365, 'B')\
           .add_atom(10.69, 6.058, 1.365, 'B')\
           .add_atom(10.718, 3.28996, 1.365, 'B')\
           .add_atom(7.922, 6.03004, 1.365, 'B')\
           .add_atom(9.32, 10.9976, 1.365, 'X')\
           .add_atom(9.32, 7.6424, 1.365, 'X')\
           .add_atom(2.9824, 4.66, 1.365, 'X')\
           .add_atom(6.3376, 4.66, 1.365, 'X')

def s41():
  """ s41 lattice """
  from pylada.crystal import Structure
  return Structure( 5.81, 0, 0,\
                    0, 5.96, 0,\
                    0, 0, 11.71,\
                    scale=1.2, name='s41' )\
           .add_atom(5.03146, 1.79992, 0.74944, 'A')\
           .add_atom(2.12646, 1.18008, 10.9606, 'A')\
           .add_atom(0.77854, 4.77992, 5.10556, 'A')\
           .add_atom(3.68354, 4.16008, 6.60444, 'A')\
           .add_atom(0.77854, 4.16008, 10.9606, 'A')\
           .add_atom(3.68354, 4.77992, 0.74944, 'A')\
           .add_atom(5.03146, 1.18008, 6.60444, 'A')\
           .add_atom(2.12646, 1.79992, 5.10556, 'A')\
           .add_atom(3.59058, 0.298, 3.7472, 'B')\
           .add_atom(0.68558, 2.682, 7.9628, 'B')\
           .add_atom(2.21942, 3.278, 2.1078, 'B')\
           .add_atom(5.12442, 5.662, 9.6022, 'B')\
           .add_atom(2.21942, 5.662, 7.9628, 'B')\
           .add_atom(5.12442, 3.278, 3.7472, 'B')\
           .add_atom(3.59058, 2.682, 9.6022, 'B')\
           .add_atom(0.68558, 0.298, 2.1078, 'B')\
           .add_atom(2.98634, 1.03108, 1.33494, 'X')\
           .add_atom(0.08134, 1.94892, 10.3751, 'X')\
           .add_atom(2.82366, 4.01108, 4.52006, 'X')\
           .add_atom(5.72866, 4.92892, 7.18994, 'X')\
           .add_atom(2.82366, 4.92892, 10.3751, 'X')\
           .add_atom(5.72866, 4.01108, 1.33494, 'X')\
           .add_atom(2.98634, 1.94892, 7.18994, 'X')\
           .add_atom(0.08134, 1.03108, 4.52006, 'X')

def s8():
  """ s8 lattice """
  from pylada.crystal import Structure
  return Structure( 3.65, 0, 0,\
                    0, 3.65, 0,\
                    0, 0, 6.8636,\
                    scale=1.15, name='s8' )\
           .add_atom(1.825, 1.825, 3.4318, 'A')\
           .add_atom(1.825, 1.825, 0, 'A')\
           .add_atom(0, 0, 5.1477, 'B')\
           .add_atom(0, 0, 1.7159, 'B')\
           .add_atom(1.825, 0, 3.4318, 'X')\
           .add_atom(0, 1.825, 0, 'X')

def s17():
  """ s17 lattice """
  from pylada.crystal import Structure
  return Structure( 4.244, -2.122, 0,\
                    0, 3.67541, 0,\
                    0, 0, 4.563,\
                    scale=1, name='s17' )\
           .add_atom(0, 0, 0, 'A')\
           .add_atom(2.122, 1.22514, 2.2815, 'B')\
           .add_atom(-2.122e-08, 2.45027, 2.2815, 'X')

def s27():
  """ s27 lattice """
  from pylada.crystal import Structure
  return Structure( 2.3005, 4.601, 0,\
                    3.6855, 0, 7.371,\
                    3.9405, 0, 0,\
                    scale=1, name='s27' )\
           .add_atom(4.601, 5.79125, 0.0191508, 'A')\
           .add_atom(4.601, 1.57975, 0.0191508, 'A')\
           .add_atom(2.3005, 3.6855, 1.00365, 'B')\
           .add_atom(2.3005, 7.371, 1.62664, 'B')\
           .add_atom(4.601, 5.06867, 2.62855, 'X')\
           .add_atom(4.601, 9.67333, 2.62855, 'X')

def s26():
  """ s26 lattice """
  from pylada.crystal import Structure
  return Structure( 6.997, 0, 3.4985,\
                    0, 10.83, 5.415,\
                    0, 0, 3.1435,\
                    scale=1.1, name='s26' )\
           .add_atom(5.24775, 8.65967, 1.86347, 'A')\
           .add_atom(8.74625, 13.0003, 1.86347, 'A')\
           .add_atom(5.24775, 13.2202, 1.70189, 'A')\
           .add_atom(8.74625, 8.43982, 1.70189, 'A')\
           .add_atom(8.74625, 5.43774, 2.62671, 'A')\
           .add_atom(5.24775, 5.39226, 2.62671, 'A')\
           .add_atom(3.70491, 6.75359, 0.75444, 'B')\
           .add_atom(3.29209, 4.07641, 0.75444, 'B')\
           .add_atom(6.79059, 6.75359, 0.75444, 'B')\
           .add_atom(7.20341, 4.07641, 0.75444, 'B')\
           .add_atom(3.4985, 10.83, 1.57238, 'B')\
           .add_atom(6.997, 10.83, 1.57238, 'B')\
           .add_atom(7.03898, 14.431, 3.11395, 'X')\
           .add_atom(6.95502, 7.22903, 3.11395, 'X')\
           .add_atom(10.4535, 14.431, 3.11395, 'X')\
           .add_atom(3.54048, 7.22903, 3.11395, 'X')\
           .add_atom(1.74925, 5.689, 0.0345785, 'X')\
           .add_atom(5.24775, 5.141, 0.0345785, 'X')

def s30():
  """ s30 lattice """
  from pylada.crystal import Structure
  return Structure( 3.75, 0, 0,\
                    0, 7.95, 0,\
                    0, 0, 3.3,\
                    scale=1.2, name='s30' )\
           .add_atom(1.875, 0.65985, 1.65, 'A')\
           .add_atom(0, 7.29015, 0, 'A')\
           .add_atom(1.875, 5.52525, 1.65, 'B')\
           .add_atom(0, 2.42475, 0, 'B')\
           .add_atom(1.875, 7.17885, 0, 'X')\
           .add_atom(0, 0.77115, 1.65, 'X')

def s9():
  """ s9 lattice """
  from pylada.crystal import Structure
  return Structure( 8.994, 4.497, -4.50041,\
                    0, 4.478, 0,\
                    0, 0, 5.62145,\
                    scale=1, name='s9' )\
           .add_atom(3.39338, 1.10159, 3.05245, 'A')\
           .add_atom(-1.15, 1.10159, 5.37973, 'A')\
           .add_atom(5.59721, 3.37641, 2.569, 'A')\
           .add_atom(10.1406, 3.37641, 0.241722, 'A')\
           .add_atom(-2.25021, 0, 2.81073, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(6.7455, 2.239, 0, 'B')\
           .add_atom(-0.0017052, 2.239, 2.81073, 'B')\
           .add_atom(3.69757, 3.35671, 4.21609, 'X')\
           .add_atom(7.53981, 3.35671, 4.21609, 'X')\
           .add_atom(5.29302, 1.12129, 1.40536, 'X')\
           .add_atom(1.45078, 1.12129, 1.40536, 'X')

def s19():
  """ s19 lattice """
  from pylada.crystal import Structure
  return Structure( 4.212, -2.106, 0,\
                    0, 3.6477, 0,\
                    0, 0, 6.803,\
                    scale=1, name='s19' )\
           .add_atom(2.106, 1.2159, 5.33491, 'A')\
           .add_atom(2.106, 1.2159, 1.46809, 'A')\
           .add_atom(0, 0, 3.4015, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(-2.10579e-08, 2.4318, 4.78251, 'X')\
           .add_atom(-2.10579e-08, 2.4318, 2.02049, 'X')

def s4():
  """ s4 lattice """
  from pylada.crystal import Structure
  return Structure( 7.204, -3.602, 0,\
                    0, 6.23885, 0,\
                    0, 0, 4.27,\
                    scale=1, name='s4' )\
           .add_atom(2.70438, 4.68413, 0, 'A')\
           .add_atom(1.79524, 0, 0, 'A')\
           .add_atom(-0.897618, 1.55472, 0, 'A')\
           .add_atom(1.49339, 2.58663, 2.135, 'B')\
           .add_atom(4.21722, 0, 2.135, 'B')\
           .add_atom(-2.10861, 3.65222, 2.135, 'B')\
           .add_atom(3.602, 2.07962, 0, 'X')\
           .add_atom(-3.60236e-08, 4.15923, 0, 'X')\
           .add_atom(0, 0, 2.135, 'X')

def s35():
  """ s35 lattice """
  from pylada.crystal import Structure
  return Structure( 3.917, 0, 0,\
                    0, 3.917, 0,\
                    0, 0, 5.431,\
                    scale=1.2, name='s35' )\
           .add_atom(1.9585, 1.9585, 2.7155, 'A')\
           .add_atom(0, 0, 2.7155, 'A')\
           .add_atom(1.9585, 1.9585, 0, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(1.9585, 0, 1.20568, 'X')\
           .add_atom(0, 1.9585, 4.22532, 'X')

def s40():
  """ s40 lattice """
  from pylada.crystal import Structure
  return Structure( 4.942, 0, 0,\
                    0, 3.776, 0,\
                    0, 0, 7.172,\
                    scale=1.2, name='s40' )\
           .add_atom(2.66176, 0.944, 5.58268, 'A')\
           .add_atom(0.190761, 0.944, 5.17532, 'A')\
           .add_atom(2.28024, 2.832, 1.58932, 'A')\
           .add_atom(4.75124, 2.832, 1.99668, 'A')\
           .add_atom(3.34376, 0.944, 3.12412, 'B')\
           .add_atom(0.872757, 0.944, 0.461877, 'B')\
           .add_atom(1.59824, 2.832, 4.04788, 'B')\
           .add_atom(4.06924, 2.832, 6.71012, 'B')\
           .add_atom(3.45248, 0.944, 0.789637, 'X')\
           .add_atom(0.981481, 0.944, 2.79636, 'X')\
           .add_atom(1.48952, 2.832, 6.38236, 'X')\
           .add_atom(3.96052, 2.832, 4.37564, 'X')

def s32():
  """ s32 lattice """
  from pylada.crystal import Structure
  return Structure( 5.1673, 0, -0.884148,\
                    0, 5.1466, 0,\
                    0, 0, 5.17248,\
                    scale=1.2, name='s32' )\
           .add_atom(1.48822, 3.84708, 5.12075, 'A')\
           .add_atom(2.35286, 1.27378, 2.63796, 'A')\
           .add_atom(2.79494, 1.29952, 0.0517248, 'A')\
           .add_atom(1.93029, 3.87282, 2.53451, 'A')\
           .add_atom(-0.450447, 0.898596, 4.35833, 'B')\
           .add_atom(4.29152, 3.4719, 3.40039, 'B')\
           .add_atom(4.7336, 4.248, 0.814148, 'B')\
           .add_atom(-0.00837301, 1.6747, 1.77209, 'B')\
           .add_atom(0.957602, 2.43589, 3.68746, 'X')\
           .add_atom(2.88348, 5.00919, 4.07126, 'X')\
           .add_atom(3.32555, 2.71071, 1.48502, 'X')\
           .add_atom(1.39968, 0.137414, 1.10122, 'X')

def s14():
  """ s14 lattice """
  from pylada.crystal import Structure
  return Structure( 8.471, 0, 0,\
                    0, 3.676, 0,\
                    0, 0, 5.537,\
                    scale=1.15, name='s14' )\
           .add_atom(4.6388, 0.919, 5.12455, 'A')\
           .add_atom(0.403304, 0.919, 3.18095, 'A')\
           .add_atom(3.8322, 2.757, 0.412451, 'A')\
           .add_atom(8.0677, 2.757, 2.35605, 'A')\
           .add_atom(5.55842, 0.919, 2.62171, 'B')\
           .add_atom(1.32292, 0.919, 0.146786, 'B')\
           .add_atom(2.91258, 2.757, 2.91529, 'B')\
           .add_atom(7.14808, 2.757, 5.39021, 'B')\
           .add_atom(7.68735, 0.919, 1.43862, 'X')\
           .add_atom(3.45185, 0.919, 1.32988, 'X')\
           .add_atom(0.783652, 2.757, 4.09838, 'X')\
           .add_atom(5.01915, 2.757, 4.20712, 'X')

def s22():
  """ s22 lattice """
  from pylada.crystal import Structure
  return Structure( 5.882, 0, -2.46819,\
                    0, 5.753, 0,\
                    0, 0, 6.92,\
                    scale=1, name='s22' )\
           .add_atom(2.929, 3.50818, 1.31501, 'A')\
           .add_atom(-1.24609, 5.12132, 4.77501, 'A')\
           .add_atom(0.484812, 2.24482, 5.605, 'A')\
           .add_atom(4.6599, 0.631679, 2.14499, 'A')\
           .add_atom(0.117578, 3.61346, 2.66351, 'B')\
           .add_atom(1.82448, 5.01604, 6.12351, 'B')\
           .add_atom(3.29624, 2.13954, 4.25649, 'B')\
           .add_atom(1.58933, 0.736959, 0.796493, 'B')\
           .add_atom(-1.34012, 3.56341, 6.6086, 'X')\
           .add_atom(2.83497, 5.06609, 3.1486, 'X')\
           .add_atom(4.75393, 2.18959, 0.3114, 'X')\
           .add_atom(0.578841, 0.686908, 3.7714, 'X')

def s7():
  """ s7 lattice """
  from pylada.crystal import Structure
  return Structure( 7.804, 0, 0,\
                    0, 9.209, 0,\
                    0, 0, 4.578,\
                    scale=1.1, name='s7' )\
           .add_atom(2.79695, 3.98289, 1.1445, 'A')\
           .add_atom(6.69895, 0.621607, 1.1445, 'A')\
           .add_atom(5.00705, 5.22611, 3.4335, 'A')\
           .add_atom(1.10505, 8.58739, 3.4335, 'A')\
           .add_atom(7.59485, 6.14977, 1.1445, 'B')\
           .add_atom(3.69285, 7.66373, 1.1445, 'B')\
           .add_atom(0.209147, 3.05923, 3.4335, 'B')\
           .add_atom(4.11115, 1.54527, 3.4335, 'B')\
           .add_atom(5.82334, 3.64676, 1.1445, 'X')\
           .add_atom(1.92134, 0.957736, 1.1445, 'X')\
           .add_atom(1.98066, 5.56224, 3.4335, 'X')\
           .add_atom(5.88266, 8.25126, 3.4335, 'X')

def s33():
  """ s33 lattice """
  from pylada.crystal import Structure
  return Structure( 5.9341, 0, 0,\
                    0, 5.9341, 0,\
                    0, 0, 5.9341,\
                    scale=1, name='s33' )\
           .add_atom(5.18201, 2.21496, 0.752088, 'A')\
           .add_atom(2.21496, 0.752088, 5.18201, 'A')\
           .add_atom(0.752088, 5.18201, 2.21496, 'A')\
           .add_atom(3.71914, 3.71914, 3.71914, 'A')\
           .add_atom(0.70509, 3.67214, 5.22901, 'B')\
           .add_atom(3.67214, 5.22901, 0.70509, 'B')\
           .add_atom(5.22901, 0.70509, 3.67214, 'B')\
           .add_atom(2.26196, 2.26196, 2.26196, 'B')\
           .add_atom(3.07843, 0.111383, 2.85567, 'X')\
           .add_atom(0.111383, 2.85567, 3.07843, 'X')\
           .add_atom(2.85567, 3.07843, 0.111383, 'X')\
           .add_atom(5.82272, 5.82272, 5.82272, 'X')

def s10():
  """ s10 lattice """
  from pylada.crystal import Structure
  return Structure( 3.3195, 0, 6.639,\
                    4.0595, 0, 0,\
                    0, 6.596, 0,\
                    scale=1, name='s10' )\
           .add_atom(6.639, 3.04706, 1.649, 'A')\
           .add_atom(3.3195, 1.01244, 4.947, 'A')\
           .add_atom(0, 0, 3.298, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(3.3195, 2.4357, 1.649, 'X')\
           .add_atom(6.639, 1.6238, 4.947, 'X')

def s5():
  """ s5 lattice """
  from pylada.crystal import Structure
  return Structure( 3.722, -1.861, 0,\
                    0, 3.22335, 0,\
                    0, 0, 7.232,\
                    scale=1.15, name='s5' )\
           .add_atom(0, 0, 3.616, 'A')\
           .add_atom(0, 0, 0, 'A')\
           .add_atom(-1.86119e-08, 2.1489, 1.808, 'B')\
           .add_atom(1.861, 1.07445, 5.424, 'B')\
           .add_atom(1.861, 1.07445, 1.808, 'X')\
           .add_atom(-1.86119e-08, 2.1489, 5.424, 'X')

def s3():
  """ s3 lattice """
  from pylada.crystal import Structure
  return Structure( 4.086, 0, 0,\
                    0, 4.086, 0,\
                    0, 0, 6.884,\
                    scale=1.15, name='s3' )\
           .add_atom(0, 2.043, 4.38511, 'A')\
           .add_atom(2.043, 0, 2.49889, 'A')\
           .add_atom(2.043, 2.043, 0, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(0, 2.043, 1.41122, 'X')\
           .add_atom(2.043, 0, 5.47278, 'X')

def s16():
  """ s16 lattice """
  from pylada.crystal import Structure
  return Structure( 4.54, 0, -3.32245,\
                    0, 4.76, 0,\
                    0, 0, 4.76262,\
                    scale=1.3, name='s16' )\
           .add_atom(-0.486304, 4.56217, 2.80652, 'A')\
           .add_atom(0.0426271, 2.18217, 4.33742, 'A')\
           .add_atom(1.70385, 0.197826, 1.9561, 'A')\
           .add_atom(1.17492, 2.57783, 0.425207, 'A')\
           .add_atom(-2.60034, 0.516365, 4.24826, 'B')\
           .add_atom(2.15666, 2.89636, 2.89567, 'B')\
           .add_atom(3.81788, 4.24364, 0.514363, 'B')\
           .add_atom(-0.93911, 1.86364, 1.86695, 'B')\
           .add_atom(-1.6393, 1.4508, 3.3305, 'X')\
           .add_atom(1.19563, 3.8308, 3.81343, 'X')\
           .add_atom(2.85685, 3.3092, 1.43212, 'X')\
           .add_atom(0.0219227, 0.9292, 0.949191, 'X')

def s37():
  """ s37 lattice """
  from pylada.crystal import Structure
  return Structure( 5.6824, 0, 0,\
                    0, 5.6824, 0,\
                    0, 0, 5.6824,\
                    scale=1.1, name='s37' )\
           .add_atom(5.2136, 2.3724, 0.468798, 'A')\
           .add_atom(2.3724, 0.468798, 5.2136, 'A')\
           .add_atom(0.468798, 5.2136, 2.3724, 'A')\
           .add_atom(3.31, 3.31, 3.31, 'A')\
           .add_atom(3.38557, 0.544374, 2.29683, 'B')\
           .add_atom(0.544374, 2.29683, 3.38557, 'B')\
           .add_atom(2.29683, 3.38557, 0.544374, 'B')\
           .add_atom(5.13803, 5.13803, 5.13803, 'B')\
           .add_atom(1.02851, 3.86971, 4.65389, 'X')\
           .add_atom(3.86971, 4.65389, 1.02851, 'X')\
           .add_atom(4.65389, 1.02851, 3.86971, 'X')\
           .add_atom(1.81269, 1.81269, 1.81269, 'X')

def s39():
  """ s39 lattice """
  from pylada.crystal import Structure
  return Structure( 1.731, 3.462, 0,\
                    6.165, 0, 12.33,\
                    2.4105, 0, 0,\
                    scale=1, name='s39' )\
           .add_atom(3.462, 6.165, 1.34506, 'A')\
           .add_atom(1.731, 12.33, 1.06544, 'A')\
           .add_atom(8.10998e-17, 2.60163, 1.12935e-16, 'B')\
           .add_atom(3.03259e-16, 9.72837, 4.22303e-16, 'B')\
           .add_atom(1.731, 7.6298, 3.31206e-16, 'X')\
           .add_atom(1.731, 4.7002, 2.04033e-16, 'X')

def s23():
  """ s23 lattice """
  from pylada.crystal import Structure
  return Structure( 6.363, 0, 0,\
                    0, 6.363, 0,\
                    0, 0, 6.363,\
                    scale=1, name='s23' )\
           .add_atom(2.3384, 5.5199, 4.0246, 'A')\
           .add_atom(5.5199, 4.0246, 2.3384, 'A')\
           .add_atom(4.0246, 2.3384, 5.5199, 'A')\
           .add_atom(0.843098, 0.843098, 0.843098, 'A')\
           .add_atom(0.489951, 3.67145, 5.87305, 'B')\
           .add_atom(3.67145, 5.87305, 0.489951, 'B')\
           .add_atom(5.87305, 0.489951, 3.67145, 'B')\
           .add_atom(2.69155, 2.69155, 2.69155, 'B')\
           .add_atom(4.20594, 1.02444, 2.15706, 'X')\
           .add_atom(1.02444, 2.15706, 4.20594, 'X')\
           .add_atom(2.15706, 4.20594, 1.02444, 'X')\
           .add_atom(5.33856, 5.33856, 5.33856, 'X')

def s28():
  """ s28 lattice """
  from pylada.crystal import Structure
  return Structure( 7.342, -3.671, 0,\
                    0, 6.35836, 0,\
                    0, 0, 7.218,\
                    scale=1, name='s28' )\
           .add_atom(1.55419, 2.47041, 1.8045, 'A')\
           .add_atom(4.42546, 0.110763, 1.8045, 'A')\
           .add_atom(5.03334, 3.77718, 1.8045, 'A')\
           .add_atom(1.36234, 2.58118, 5.4135, 'A')\
           .add_atom(-2.11681, 3.88795, 5.4135, 'A')\
           .add_atom(0.754464, 6.2476, 5.4135, 'A')\
           .add_atom(-3.671e-08, 4.23891, 0.241153, 'B')\
           .add_atom(-3.671e-08, 4.23891, 3.36785, 'B')\
           .add_atom(3.671, 2.11945, 3.85015, 'B')\
           .add_atom(3.671, 2.11945, 6.97685, 'B')\
           .add_atom(0, 0, 1.8045, 'B')\
           .add_atom(0, 0, 5.4135, 'B')\
           .add_atom(2.67983, 4.6416, 0, 'X')\
           .add_atom(1.98234, 0, 0, 'X')\
           .add_atom(-0.99117, 1.71676, 0, 'X')\
           .add_atom(2.67983, 4.6416, 3.609, 'X')\
           .add_atom(1.98234, 0, 3.609, 'X')\
           .add_atom(-0.99117, 1.71676, 3.609, 'X')

def s12():
  """ s12 lattice """
  from pylada.crystal import Structure
  return Structure( 8.514, 0, 4.257,\
                    0, 8.514, 4.257,\
                    0, 0, 1.9045,\
                    scale=1.2, name='s12' )\
           .add_atom(1.17153, 4.257, 0, 'A')\
           .add_atom(7.34247, 4.257, 0, 'A')\
           .add_atom(4.257, 1.17153, 0, 'A')\
           .add_atom(4.257, 7.34247, 0, 'A')\
           .add_atom(1.33585, 1.33585, 0, 'B')\
           .add_atom(7.17815, 7.17815, 0, 'B')\
           .add_atom(7.17815, 1.33585, 0, 'B')\
           .add_atom(1.33585, 7.17815, 0, 'B')\
           .add_atom(0, 2.61465, 0, 'X')\
           .add_atom(0, 5.89935, 0, 'X')\
           .add_atom(5.89935, 0, 0, 'X')\
           .add_atom(2.61465, 0, 0, 'X')

def s2():
  """ s2 lattice """
  from pylada.crystal import Structure
  return Structure( 4.175, -2.0875, 0,\
                    0, 3.61566, 0,\
                    0, 0, 7,\
                    scale=1.2, name='s2' )\
           .add_atom(0, 0, 4.816, 'A')\
           .add_atom(0, 0, 1.316, 'A')\
           .add_atom(2.0875, 1.20522, 3.5, 'B')\
           .add_atom(-2.08813e-08, 2.41044, 0, 'B')\
           .add_atom(2.0875, 1.20522, 6.195, 'X')\
           .add_atom(-2.08813e-08, 2.41044, 2.695, 'X')

def s31():
  """ s31 lattice """
  from pylada.crystal import Structure
  return Structure( 4.54, -2.27, 2.27,\
                    0, 3.93176, 1.31059,\
                    0, 0, 10.8967,\
                    scale=1, name='s31' )\
           .add_atom(2.27, 1.31059, 1.03409, 'A')\
           .add_atom(2.27, 3.93176, 9.86257, 'A')\
           .add_atom(2.27, 1.31059, 7.1079, 'B')\
           .add_atom(2.27, 3.93176, 3.78877, 'B')\
           .add_atom(4.54, 2.62117, 9.09327, 'X')\
           .add_atom(0, 2.62117, 1.8034, 'X')

def s38():
  """ s38 lattice """
  from pylada.crystal import Structure
  return Structure( 7.31, -3.655, 0,\
                    0, 6.33065, 0,\
                    0, 0, 4.04,\
                    scale=1.2, name='s38' )\
           .add_atom(1.8275, 3.16532, 0, 'A')\
           .add_atom(3.655, 0, 0, 'A')\
           .add_atom(-1.8275, 3.16532, 0, 'A')\
           .add_atom(0, 0, 2.02, 'B')\
           .add_atom(3.655, 2.11022, 2.02, 'B')\
           .add_atom(-3.655e-08, 4.22043, 2.02, 'B')\
           .add_atom(0, 0, 0, 'X')\
           .add_atom(3.655, 2.11022, 0, 'X')\
           .add_atom(-3.655e-08, 4.22043, 0, 'X')

def s36():
  """ s36 lattice """
  from pylada.crystal import Structure
  return Structure( 2.081, -1.0405, 0,\
                    0, 1.8022, 0,\
                    0, 0, 9.234,\
                    scale=1.2, name='s36' )\
           .add_atom(1.0405, 0.600733, 8.3106, 'A')\
           .add_atom(-1.0405e-08, 1.20147, 0.9234, 'A')\
           .add_atom(-1.0405e-08, 1.20147, 6.16831, 'B')\
           .add_atom(1.0405, 0.600733, 3.06569, 'B')\
           .add_atom(0, 0, 7.8766, 'X')\
           .add_atom(0, 0, 1.3574, 'X')

def s18():
  """ s18 lattice """
  from pylada.crystal import Structure
  return Structure( 9.296, -4.648, 0,\
                    0, 8.05057, 0,\
                    0, 0, 7.346,\
                    scale=1.1, name='s18' )\
           .add_atom(4.648, 5.24897, 4.99528, 'A')\
           .add_atom(2.42626, 1.4008, 4.99528, 'A')\
           .add_atom(6.86974, 1.4008, 4.99528, 'A')\
           .add_atom(0, 2.8016, 1.32228, 'A')\
           .add_atom(2.22174, 6.64977, 1.32228, 'A')\
           .add_atom(-2.22174, 6.64977, 1.32228, 'A')\
           .add_atom(4.64846, 2.68326, 2.29195, 'A')\
           .add_atom(-0.0004648, 5.36732, 5.96495, 'A')\
           .add_atom(0, 7.76075, 3.673, 'B')\
           .add_atom(-2.07301, 4.1702, 3.673, 'B')\
           .add_atom(2.07301, 4.1702, 3.673, 'B')\
           .add_atom(4.648, 0.289821, 0, 'B')\
           .add_atom(6.72101, 3.88038, 0, 'B')\
           .add_atom(2.57499, 3.88038, 0, 'B')\
           .add_atom(0, 0, 3.673, 'B')\
           .add_atom(0, 0, 0, 'B')\
           .add_atom(4.648, 5.31338, 1.89527, 'X')\
           .add_atom(2.37048, 1.3686, 1.89527, 'X')\
           .add_atom(6.92552, 1.3686, 1.89527, 'X')\
           .add_atom(0, 2.73719, 5.56827, 'X')\
           .add_atom(2.27752, 6.68197, 5.56827, 'X')\
           .add_atom(-2.27752, 6.68197, 5.56827, 'X')\
           .add_atom(4.64846, 2.68326, 5.28177, 'X')\
           .add_atom(-0.0004648, 5.36732, 1.60877, 'X')

def s15():
  """ s15 lattice """
  from pylada.crystal import Structure
  return Structure( 9.639, 0, 0,\
                    0, 13.674, 0,\
                    0, 0, 5.432,\
                    scale=1, name='s15' )\
           .add_atom(6.24511, 2.20562, 0.293328, 'A')\
           .add_atom(1.42561, 4.63138, 2.42267, 'A')\
           .add_atom(3.39389, 9.04262, 5.13867, 'A')\
           .add_atom(8.21339, 11.4684, 3.00933, 'A')\
           .add_atom(3.39389, 11.4684, 5.13867, 'A')\
           .add_atom(8.21339, 9.04262, 3.00933, 'A')\
           .add_atom(6.24511, 4.63138, 0.293328, 'A')\
           .add_atom(1.42561, 2.20562, 2.42267, 'A')\
           .add_atom(4.8195, 0, 2.716, 'A')\
           .add_atom(0, 6.837, 0, 'A')\
           .add_atom(4.8195, 6.837, 2.716, 'A')\
           .add_atom(0, 0, 0, 'A')\
           .add_atom(7.98206, 1.20331, 2.85886, 'B')\
           .add_atom(3.16256, 5.63369, 5.28914, 'B')\
           .add_atom(1.65694, 8.04031, 2.57314, 'B')\
           .add_atom(6.47644, 12.4707, 0.142862, 'B')\
           .add_atom(1.65694, 12.4707, 2.57314, 'B')\
           .add_atom(6.47644, 8.04031, 0.142862, 'B')\
           .add_atom(7.98206, 5.63369, 2.85886, 'B')\
           .add_atom(3.16256, 1.20331, 5.28914, 'B')\
           .add_atom(9.61394, 3.4185, 5.33694, 'B')\
           .add_atom(4.79444, 3.4185, 2.81106, 'B')\
           .add_atom(0.0250614, 10.2555, 0.09506, 'B')\
           .add_atom(4.84456, 10.2555, 2.62094, 'B')\
           .add_atom(5.5511, 1.057, 1.48402, 'X')\
           .add_atom(0.7316, 5.78, 1.23198, 'X')\
           .add_atom(4.0879, 7.894, 3.94798, 'X')\
           .add_atom(8.9074, 12.617, 4.20002, 'X')\
           .add_atom(4.0879, 12.617, 3.94798, 'X')\
           .add_atom(8.9074, 7.894, 4.20002, 'X')\
           .add_atom(5.5511, 5.78, 1.48402, 'X')\
           .add_atom(0.7316, 1.057, 1.23198, 'X')\
           .add_atom(6.9796, 3.4185, 4.57157, 'X')\
           .add_atom(2.1601, 3.4185, 3.57643, 'X')\
           .add_atom(2.6594, 10.2555, 0.860429, 'X')\
           .add_atom(7.4789, 10.2555, 1.85557, 'X')

