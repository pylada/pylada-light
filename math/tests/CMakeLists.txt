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

if(tests)

  set(TYPES 0 1)
  foreach(TYPE ${TYPES})
    add_executable(fuzzy${TYPE} fuzzy.cc)
    set_property( TARGET fuzzy${TYPE} PROPERTY COMPILE_DEFINITIONS
                  PYLADA_TEST_INCTYPE=${TYPE})
    ADD_TEST(fuzzy${TYPE} ${CMAKE_CURRENT_BINARY_DIR}/fuzzy${TYPE})
    set_tests_properties(fuzzy${TYPE} PROPERTIES LABELS "C;math")
  endforeach(TYPE)

  set(TYPES 0 1 2 3)
  foreach(TYPE ${TYPES})
    add_executable(fuzzyeigen${TYPE} fuzzyeigen.cc)
    set_property( TARGET fuzzyeigen${TYPE} PROPERTY COMPILE_DEFINITIONS
                  PYLADA_TEST_INCTYPE=${TYPE})
    ADD_TEST(fuzzyeigen${TYPE} ${CMAKE_CURRENT_BINARY_DIR}/fuzzyeigen${TYPE})
    set_tests_properties(fuzzyeigen${TYPE} PROPERTIES LABELS "C;math")
  endforeach(TYPE)

  add_executable(symtests symtests.cc)
  ADD_TEST(symtests ${CMAKE_CURRENT_BINARY_DIR}/symtests)
  set_tests_properties(symtests PROPERTIES LABELS "C;math")

  # Python tests are added to pylada.tests rather than pylada.math.tests
  # because pylada.math is an extension, not a subpackage. It does not have a
  # directory of its own.
  add_python_module("pylada.tests._gruber"
      MAIN gruber.cc
      INSTALL
      LABELS math
  )
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set_property(TARGET pylada_tests__gruber
        PROPERTY COMPILE_DEFINITIONS PYLADA_LIM=5)
  else("${CMAKE_BUILD_TYPE}" STREQUAL "Release") 
    set_property(TARGET pylada_tests__gruber
        PROPERTY COMPILE_DEFINITIONS PYLADA_LIM=2)
  endif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")

  add_python_module("pylada.tests._smith"
      MAIN smith_form.cc
      INSTALL
      LABELS math
  )

  add_nose_tests(test_*.py INSTALL "pylada.tests" LABELS math)

endif(tests)
