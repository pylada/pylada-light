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

""" Checks that point group of cell is determined correctly. """
def test(cell, numops):
  """ Test structure initialization. """
  from numpy import all, abs, dot, array
  from numpy.linalg import inv, det
  from pylada.crystal.cppwrappers import cell_invariants, Structure
  from pylada.math import is_integer

  ops = cell_invariants(cell) 
  if isinstance(cell, Structure): cell = cell.cell.copy()
  assert len(ops) == numops
  for op in ops:
    assert op.shape == (4, 3)
    assert all(abs(op[3, :]) < 1e-8) 
    transformation = dot(dot(inv(cell), op[:3]), cell)
    assert is_integer(transformation)
    assert abs(abs(det(transformation))-1e0) < 1e-8
  if numops != 48:
    allops = cell_invariants(array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]))
    failed = 0
    for op in allops: 
      transformation = dot(dot(inv(cell), op[:3]), cell)
      if not (is_integer(transformation) and abs(abs(det(transformation))-1e0) < 1e-8):
        failed += 1
    assert failed == 48 - numops

if __name__ == "__main__":
  from sys import argv, path 
  if len(argv) > 0: path.extend(argv[1:])
  
  def test_(cell, numops):
    from pylada.crystal.cppwrappers import Structure
    from numpy import array
    test(array(cell), numops)
    # also good test to make sure that structure storage is correct.
    test(Structure(cell).add_atom(0,0,0, "Si"), numops)

  test_([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]], 48)
  test_([[-0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5]], 48)
  test_([[-0.6,0.5,0.5],[0.6,-0.5,0.5],[0.6,0.5,-0.5]], 4)
  test_([[-0.7,0.7,0.7],[0.6,-0.5,0.5],[0.6,0.5,-0.5]], 8)
  test_([[-0.765,0.7,0.7],[0.665,-0.5,0.5],[0.6,0.5,-0.5]], 2)
