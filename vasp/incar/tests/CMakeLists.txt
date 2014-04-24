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

if(NOT compile_debug_tests)
  
  add_test(incar_algo python ${CMAKE_CURRENT_SOURCE_DIR}/algo.py)
  add_test(incar_choices python ${CMAKE_CURRENT_SOURCE_DIR}/choices.py)
  add_test(incar_magmom python ${CMAKE_CURRENT_SOURCE_DIR}/magmom.py)
  add_test(incar_nelect python ${CMAKE_CURRENT_SOURCE_DIR}/nelect.py)
  add_test(incar_npar python ${CMAKE_CURRENT_SOURCE_DIR}/npar.py)
  add_test(incar_others python ${CMAKE_CURRENT_SOURCE_DIR}/others.py)
  add_test(incar_relax python ${CMAKE_CURRENT_SOURCE_DIR}/relax.py)
  add_test(incar_restart python ${CMAKE_CURRENT_SOURCE_DIR}/restart.py)
  add_test(incar_uparams python ${CMAKE_CURRENT_SOURCE_DIR}/uparams.py)
  add_test(incar_smearing python ${CMAKE_CURRENT_SOURCE_DIR}/uparams.py)

endif(NOT compile_debug_tests)
