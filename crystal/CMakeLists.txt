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


set(LIBCRYSTAL_PYSOURCES __init__.py read.py write.py ABX.py A2BX4.py
                         readCif.py bravais.py binary.py iterator.py)


include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES} ${Pylada_SOURCE_DIR})
#include root source directory
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")


add_library (pycrystal SHARED module.cc)
set_target_properties(pycrystal PROPERTIES OUTPUT_NAME cppwrappers)
set_target_properties(pycrystal PROPERTIES PREFIX "")
set_target_properties(pycrystal PROPERTIES SUFFIX ".so")
install(TARGETS pycrystal DESTINATION ${CMAKE_PYINSTALL_PREFIX}/pylada/crystal)
install(FILES ${LIBCRYSTAL_PYSOURCES} DESTINATION ${CMAKE_PYINSTALL_PREFIX}/pylada/crystal)
target_link_libraries(pycrystal ${PYTHON_LIBRARIES})

set(PYCRYSTAL_SOURCES atom/pybase.cc atom/cdi.cc atom/members.cc atom/getset.cc
                      structure/pybase.cc structure/cdi.cc structure/members.cc
                      structure/getset.cc structure/sequence.cc
                      structure/iterator.cc 
                      hart-forcade/pybase.cc hart-forcade/cdi.cc
                      hart-forcade/get.cc hart-forcade/members.cc
                      utilities.cc map_sites.cc equivalent_structures.cc
                      primitive.cc space_group.cc neighbors.cc
                      coordination_shells.cc confsplit.cc periodic_dnc.cc
                      methods.cc )


set(PYCRYSTAL_HEADERS atom/atom.h
                      structure/structure.h
                      hart-forcade/hart-forcade.h hart-forcade/macro.hpp
                      utilities.h supercell.h algorithms.h )

if(NOT noinstall_headers)
  install(FILES ${PYCRYSTAL_HEADERS} DESTINATION include/Pylada/crystal)
endif(NOT noinstall_headers)

add_subdirectory(tests)
