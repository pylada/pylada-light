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

    # Add fake executable to tests
    # In order to setup the right permission, we configure to one place then copy it to the next.
    configure_file(pifunctional.in.py
        "${CMAKE_CURRENT_BINARY_DIR}/pifunctional.py")
    file(COPY  "${CMAKE_CURRENT_BINARY_DIR}/pifunctional.py"
        DESTINATION "${PYTHON_BINARY_DIR}/pylada/process/tests"
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    )
    install(FILES "${PYTHON_BINARY_DIR}/pylada/process/tests/pifunctional.py"
      PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
      DESTINATION "${PYTHON_PKG_DIR}/pylada/process/tests"
    )
    set(fake_exec "${PYTHON_BINARY_DIR}/pylada/process/tests/pifunctional.py")

    if(MPIEXEC AND mpi4py_FOUND)
        add_test(pifunctional ${MPIEXEC} -n 4 ${fake_exec} --order 4)
        set_tests_properties(pifunctional PROPERTIES LABELS "python;mpi;process")
    endif()

    add_pytest(
        globalcomm.py mpi.py placement.py
        functional.py fixtures.py test_call.py test_iterator.py test_program.py
        LABELS process
        INSTALL PREFIX "pylada.process"
        FAKE_INIT
    )

    add_pytest(test_pool.py test_jobfolder.py
        INSTALL PREFIX "pylada.process"
        LABELS process;jobfolder
    )

endif()
