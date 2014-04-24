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

import quantities
def _rigged_deepcopy(self, memo):
  """ Rig by Pylada so deepcopy of scalars works.
  
      This likely a numpy bug, since deepcopying a scalar array yields a
      builtin type, rather than a scalar array. It is ticket#1176 in numpy bug
      list. 
  """
  from quantities import Quantity
  if len(self.shape) == 0:
    return super(Quantity, self).__deepcopy__(memo) * self.units
  return super(Quantity, self).__deepcopy__(memo)
quantities.Quantity.__deepcopy__ = _rigged_deepcopy

