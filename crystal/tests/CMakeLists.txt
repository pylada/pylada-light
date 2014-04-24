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

if(do_tests)

  add_test(crystal_atom python ${CMAKE_CURRENT_SOURCE_DIR}/atom.py ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_atom_memory python ${CMAKE_CURRENT_SOURCE_DIR}/atom_memory.py ${CMAKE_CURRENT_BINARY_DIR})

  add_library (atom_self SHARED "atom_self.cc")
  set_target_properties(atom_self PROPERTIES PREFIX "_")
  set_target_properties(atom_self PROPERTIES SUFFIX ".so")
  target_link_libraries(atom_self ${PYTHON_LIBRARIES})
  add_test(crystal_atom_self python ${CMAKE_CURRENT_SOURCE_DIR}/atom_self.py ${CMAKE_CURRENT_BINARY_DIR})


  add_test(crystal_structure python ${CMAKE_CURRENT_SOURCE_DIR}/structure.py)
  add_test(crystal_structure_memory python ${CMAKE_CURRENT_SOURCE_DIR}/structure_memory.py ${CMAKE_CURRENT_BINARY_DIR})

  add_test(crystal_utilities python ${CMAKE_CURRENT_SOURCE_DIR}/utilities.py ${CMAKE_CURRENT_BINARY_DIR})

  add_test(crystal_supercell python ${CMAKE_CURRENT_SOURCE_DIR}/structure_memory.py ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_primitive python ${CMAKE_CURRENT_SOURCE_DIR}/primitive.py ${CMAKE_CURRENT_BINARY_DIR})

  add_test(crystal_hart-forcade python ${CMAKE_CURRENT_SOURCE_DIR}/hart-forcade.py ${CMAKE_CURRENT_BINARY_DIR})

  add_test(crystal_cellinvariants python ${CMAKE_CURRENT_SOURCE_DIR}/cellinvariants.py ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_space_group python ${CMAKE_CURRENT_SOURCE_DIR}/space_group.py ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_periodic_dnc python ${CMAKE_CURRENT_SOURCE_DIR}/periodic_dnc.py ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_neighbors python ${CMAKE_CURRENT_SOURCE_DIR}/neighbors.py ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_coordination_shells python ${CMAKE_CURRENT_SOURCE_DIR}/coordination_shells.py
           ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_splitconfigs python ${CMAKE_CURRENT_SOURCE_DIR}/splitconfigs.py
           ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_map_sites python ${CMAKE_CURRENT_SOURCE_DIR}/map_sites.py
           ${CMAKE_CURRENT_BINARY_DIR})
  add_test(crystal_which_site python ${CMAKE_CURRENT_SOURCE_DIR}/which_site.py)

  add_test(crystal_gulp python ${CMAKE_CURRENT_SOURCE_DIR}/write_gulp.py)

endif(do_tests)
