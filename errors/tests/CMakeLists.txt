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

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDES})

if(NOT compile_debug_tests)

  set(TYPES AttributeError KeyError ValueError IndexError TypeError)
  foreach(TYPE ${TYPES})
    python_add_module (exception_${TYPE} "" "exceptions.cc")
    set_property( TARGET exception_${TYPE} PROPERTY COMPILE_DEFINITIONS
                  PYLADA_TYPE=${TYPE} PYLADA_TYPENAME="${TYPE}" PYLADA_MODULE=exception_${TYPE})
    mark_as_advanced(PYTHON_ENABLE_MODULE_exception_${TYPE})
    mark_as_advanced(PYTHON_MODULE_exception_${TYPE}_BUILD_SHARED)
    set_target_properties(exception_${TYPE} PROPERTIES OUTPUT_NAME exception_${TYPE})
    target_link_libraries(exception_${TYPE} ${Boost_PYTHON_LIBRARY})
    configure_file (
     "${CMAKE_CURRENT_SOURCE_DIR}/exceptions_py.py"
     "${CMAKE_CURRENT_BINARY_DIR}/exception_${TYPE}.py"
    )
    add_test(exception_${TYPE} python ${CMAKE_CURRENT_BINARY_DIR}/exception_${TYPE}.py)
  endforeach(TYPE ${TYPES})

  set(TYPES root input internal out_of_range infinite_loop)
  foreach(TYPE ${TYPES})
    python_add_module (exception_${TYPE} "" "exceptions.cc")
    set_property( TARGET exception_${TYPE} PROPERTY COMPILE_DEFINITIONS
                  PYLADA_TYPE=${TYPE} PYLADA_TYPENAME="${TYPE}" PYLADA_MODULE=exception_${TYPE})
    mark_as_advanced(PYTHON_ENABLE_MODULE_exception_${TYPE})
    mark_as_advanced(PYTHON_MODULE_exception_${TYPE}_BUILD_SHARED)
    set_target_properties(exception_${TYPE} PROPERTIES OUTPUT_NAME exception_${TYPE})
    target_link_libraries(exception_${TYPE} ${Boost_PYTHON_LIBRARY})
    configure_file (
     "${CMAKE_CURRENT_SOURCE_DIR}/exceptions.py"
     "${CMAKE_CURRENT_BINARY_DIR}/exception_${TYPE}.py"
    )
    add_test(exception_${TYPE} python ${CMAKE_CURRENT_BINARY_DIR}/exception_${TYPE}.py)
  endforeach(TYPE ${TYPES})

endif(NOT compile_debug_tests)
