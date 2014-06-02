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
    # Copies test data
    foreach(specie Si Rh O Zn)
        add_python_module("pylada.vasp.tests.pseudos.${specie}"
            PYFILES "pseudos/${specie}/POTCAR"
        )
    endforeach()

    file(WRITE "${CMAKE_BINARY_DIR}/__init__.py")
    add_python_module("pylada.vasp.tests"
        PYFILES "${CMAKE_BINARY_DIR}/__init__.py"
    )
    add_nose_tests(
        test_*.py
        INSTALL "pylada.vasp.tests"
        EXCLUDE test_run*.py
        LABELS vasp
    )

# if(external_program_tests)
#   add_test(vasp_run python ${CMAKE_CURRENT_SOURCE_DIR}/run.py ${CMAKE_CURRENT_SOURCE_DIR})
#   add_test(vasp_runepi python ${CMAKE_CURRENT_SOURCE_DIR}/runepi.py ${CMAKE_CURRENT_SOURCE_DIR})
#   add_test(vasp_runrelax python ${CMAKE_CURRENT_SOURCE_DIR}/runrelax.py ${CMAKE_CURRENT_SOURCE_DIR})
#   add_test(vasp_runepidoc python ${CMAKE_CURRENT_SOURCE_DIR}/runepidoc.py ${CMAKE_CURRENT_SOURCE_DIR})
#   add_test(vasp_runmass python ${CMAKE_CURRENT_SOURCE_DIR}/runmass.py ${CMAKE_CURRENT_SOURCE_DIR})
# endif(external_program_tests)
endif(tests)
