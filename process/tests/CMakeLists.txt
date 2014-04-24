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
  if(DEFINED MPI_CXX_COMPILER)
    add_executable (pifunc functional.cc)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(pifunc ${MPI_CXX_LIBRARIES} ${PYTHON_LIBRARIES})
    
    add_test(process_pifunc ${MPIEXEC} pifunc --order 4)
    add_test(process_program python ${CMAKE_CURRENT_SOURCE_DIR}/program.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
    add_test(process_call python ${CMAKE_CURRENT_SOURCE_DIR}/call.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
    add_test(process_iterator python ${CMAKE_CURRENT_SOURCE_DIR}/iterator.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
    if(jobfolder_package)
      add_test(process_pool python ${CMAKE_CURRENT_SOURCE_DIR}/pool.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
      add_test(process_jobfolder python ${CMAKE_CURRENT_SOURCE_DIR}/jobfolder.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
    endif(jobfolder_package)
  endif(DEFINED MPI_CXX_COMPILER)

endif(do_tests)
