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

set(PYEWALD_PY __init__.py)
set(PYEWALD_F90 ep_com.f90 ewaldf.f90)
set(PYEWALD_HEADERS ewald.h)
set(PYEWALD_SOURCES ewald.cc module.cc erfc.cc)

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES})
add_library (pyewald SHARED ${PYEWALD_SOURCES} ${PYEWALD_HEADERS} ${PYEWALD_F90})
set_target_properties(pyewald PROPERTIES OUTPUT_NAME cppwrappers)
set_target_properties(pyewald PROPERTIES PREFIX "")
set_target_properties(pyewald PROPERTIES SUFFIX ".so")
target_link_libraries(pyewald ${PYTHON_LIBRARIES})
target_link_libraries(pyewald ${Boost_MATH_C99_LIBRARY}
                              ${Boost_MATH_C99L_LIBRARYl}
                              ${BOOST_C99F_LIBRARY} )
install(TARGETS pyewald DESTINATION ${CMAKE_PYINSTALL_PREFIX}/pylada/ewald)
install(FILES ${PYEWALD_PY} DESTINATION ${CMAKE_PYINSTALL_PREFIX}/pylada/ewald)

if(do_tests)
  add_subdirectory(tests)
endif(do_tests)
