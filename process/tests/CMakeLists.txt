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

if(tests AND mpi4py_FOUND)

    # Add fake executable to tests
    configure_file(pifunctional.in.py
        "${CMAKE_CURRENT_BINARY_DIR}/pifunctional.py")
    file(COPY  "${CMAKE_CURRENT_BINARY_DIR}/pifunctional.py"
        DESTINATION "${PYTHON_BINARY_DIR}/pylada/process/tests"
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    )
    set(fake_exec "${PYTHON_BINARY_DIR}/pylada/process/tests/pifunctional.py")

    if(MPIEXEC)
        add_test(pifunctional ${MPIEXEC} -n 4 ${fake_exec} --order 4)
        set_tests_properties(pifunctional
            PROPERTIES LABELS "python;mpi;process")
    endif()

    add_python_module("pylada.process.tests"
        PYFILES functional.py 
        INSTALL 
        FAKE_INIT
    )
    add_nose_tests(
        test_program.py
        INSTALL "pylada.process.tests"
        LABELS process
    )
    add_test(process_call python ${CMAKE_CURRENT_SOURCE_DIR}/call.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
    add_test(process_iterator python ${CMAKE_CURRENT_SOURCE_DIR}/iterator.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
    if(jobfolder_package)
      add_test(process_pool python ${CMAKE_CURRENT_SOURCE_DIR}/pool.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
      add_test(process_jobfolder python ${CMAKE_CURRENT_SOURCE_DIR}/jobfolder.py ${CMAKE_CURRENT_BINARY_DIR}/pifunc)
    endif(jobfolder_package)

endif()
