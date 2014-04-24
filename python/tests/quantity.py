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

""" Checks atom methods and attributes. """
def test():
  """ Test atom initialization. """
  from _quantity import is_quantity, fromC, fromPy, get_angstrom, as_real,     \
                        get_as
  from numpy import abs
  from quantities import meter, angstrom, eV, nanometer# , Ry

  assert not is_quantity("a");
  assert not is_quantity(0);
  assert not is_quantity(0.4);
  assert is_quantity(eV)
  assert is_quantity(5*eV)
  assert is_quantity(5.45*nanometer)

  assert abs(fromC(5, 'meter') - 5*meter) < 1e-8
  assert abs(fromC(float((5*meter).rescale(angstrom)), 'angstrom') - 5*meter) < 1e-8
  assert abs(fromC(float((4*meter).rescale(angstrom)), 'angstrom') - 5*meter) > 1e-8
  assert abs(fromPy(5, meter) - 5*meter) < 1e-8
  assert hasattr(fromPy(5, meter), 'rescale')
  assert abs(get_angstrom(5*meter) - float(5*meter.rescale(angstrom))) < 1e-8
  assert abs(get_angstrom(5) - 5) < 1e-8
  assert abs(get_angstrom(5.5) - 5.5) < 1e-8
  assert get_angstrom(4*eV) is None
  assert abs(as_real(4.5*eV) - 4.5) < 1e-8
  assert as_real('a') is None

  assert abs(get_as(5*meter, angstrom) - float(5*meter.rescale(angstrom))) < 1e-8
  assert get_as(5*meter, eV) is None
  assert get_as(5*meter, 'a') is None
  assert get_as('5*meter', angstrom) is None
  assert abs(get_as(5*meter, 4*angstrom) - 5e10) < 1e-4

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  test()
