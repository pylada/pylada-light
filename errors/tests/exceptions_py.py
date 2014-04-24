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

import sys
sys.path = ["@CMAKE_CURRENT_BINARY_DIR@/..", "@CMAKE_CURRENT_BINARY_DIR@/.."] + sys.path
from pylada import error
import exception_@TYPE@

exception_@TYPE@.nothrow()
try: exception_@TYPE@.dothrow_nomessage()
except error.@TYPE@: pass
try: exception_@TYPE@.dothrow_message()
except error.@TYPE@ as e:  
  assert str(e).find("This is a message.") != -1
try: exception_@TYPE@.dopythrow_message()
except error.@TYPE@ as e: 
  assert str(e).find("This is another message.") != -1

try: raise error.@TYPE@("Whatever")
except error.@TYPE@ as e: 
  assert str(e).find("Whatever") != -1
try: raise error.@TYPE@("Whatever")
except @TYPE@ as e: 
  assert str(e).find("Whatever") != -1
