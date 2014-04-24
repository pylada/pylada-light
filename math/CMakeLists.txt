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

# Location where cmake first looks for cmake modules.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(numpy REQUIRED) # has to be after FortranCInterface_VERIFY for some reason

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES} ${Pylada_SOURCE_DIR})

#include root source directory
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES})
add_library (pymath SHARED module.cc noopt.cc)
SET_SOURCE_FILES_PROPERTIES(noopt.cc PROPERTIES COMPILE_FLAGS -O0)
set_target_properties(pymath PROPERTIES OUTPUT_NAME math)
set_target_properties(pymath PROPERTIES PREFIX "")
set_target_properties(pymath PROPERTIES SUFFIX ".so")
target_link_libraries(pymath ${PYTHON_LIBRARIES})
install(TARGETS pymath DESTINATION ${CMAKE_PYINSTALL_PREFIX}/pylada)

set(PYMATH_HEADERS math.h eigen.h exceptions.h fuzzy.h misc.h
                   symmetry_operator.h algorithms.h )
set(PYMATH_SOURCES gruber.cc smith_normal_form.cc methods.cc)

if(NOT noinstall_headers)
  install(FILES ${PYMATH_HEADERS} DESTINATION include/Pylada/math)
endif(NOT noinstall_headers)

add_subdirectory(tests)
