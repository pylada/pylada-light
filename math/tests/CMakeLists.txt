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

  set(TYPES 0 1)
  foreach(TYPE ${TYPES})
    add_executable(fuzzy${TYPE} fuzzy.cc)
    set_property( TARGET fuzzy${TYPE} PROPERTY COMPILE_DEFINITIONS
                  PYLADA_TEST_INCTYPE=${TYPE})
    ADD_TEST(math_fuzzy${TYPE} ${CMAKE_CURRENT_BINARY_DIR}/fuzzy${TYPE})
  endforeach(TYPE)

  set(TYPES 0 1 2 3)
  foreach(TYPE ${TYPES})
    add_executable(fuzzyeigen${TYPE} fuzzyeigen.cc)
    set_property( TARGET fuzzyeigen${TYPE} PROPERTY COMPILE_DEFINITIONS
                  PYLADA_TEST_INCTYPE=${TYPE})
    ADD_TEST(math_fuzzyeigen${TYPE} ${CMAKE_CURRENT_BINARY_DIR}/fuzzyeigen${TYPE})
  endforeach(TYPE)

  add_executable(symtests symtests.cc)
  ADD_TEST( math_symtests ${CMAKE_CURRENT_BINARY_DIR}/symtests)

  add_library (gruber SHARED "gruber.cc")
  target_link_libraries(gruber ${PYTHON_LIBRARIES})
  set_target_properties(gruber PROPERTIES PREFIX "_")
  set_target_properties(gruber PROPERTIES SUFFIX ".so")
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set_property(TARGET gruber PROPERTY COMPILE_DEFINITIONS PYLADA_LIM=5)
  else("${CMAKE_BUILD_TYPE}" STREQUAL "Release") 
    set_property(TARGET gruber PROPERTY COMPILE_DEFINITIONS PYLADA_LIM=2)
  endif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")

  add_library (smith SHARED "smith_form.cc")
  set_target_properties(smith PROPERTIES PREFIX "_")
  set_target_properties(smith PROPERTIES OUTPUT_NAME "smith")
  set_target_properties(smith PROPERTIES SUFFIX ".so")
  target_link_libraries(smith ${PYTHON_LIBRARIES})

  add_test(math_basics python ${CMAKE_CURRENT_SOURCE_DIR}/basics.py)
  add_test(math_pygruber python ${CMAKE_CURRENT_SOURCE_DIR}/gruber.py ${CMAKE_CURRENT_BINARY_DIR})
  add_test(math_pysmith python ${CMAKE_CURRENT_SOURCE_DIR}/smith.py ${CMAKE_CURRENT_BINARY_DIR})

endif(do_tests)
