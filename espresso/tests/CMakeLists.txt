###############################
#  This file is part of PyLaDa.
#
#  Copyright (C) 2013 National Renewable Energy Lab
#
#  PyLaDa is a high throughput computational platform for Physics. It aims to
#  make it easier to submit large numbers of jobs on supercomputers. It
#  provides a python interface to physical input, such as crystal structures,
#  as well as to a number of DFT (VASP, CRYSTAL) and atomic potential
#  programs. It is able to organise and launch computational jobs on PBS and
#  SLURM.
#
#  PyLaDa is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License along
#  with PyLaDa.  If not, see <http://www.gnu.org/licenses/>.
###############################

if(tests)
  add_pytest(
      GLOB test_*.py dummy_pwscf.py extract.py fixtures.py
      INSTALL PREFIX "pylada.espresso"
      LABELS espresso
      FAKE_INIT
  )
  install_python(DIRECTORY data/nonscf DESTINATION pylada/espresso/tests/data)
  install_python(DIRECTORY data/cellshape DESTINATION pylada/espresso/tests/data)
  install_python(DIRECTORY data/ions DESTINATION pylada/espresso/tests/data)
  install_python(DIRECTORY features DESTINATION pylada/espresso/tests)
  if(NOT EXISTS "${PYTHON_BINARY_DIR}/pylada/espresso/tests/data/nonscf")
    file(MAKE_DIRECTORY "${PYTHON_BINARY_DIR}/pylada/espresso/tests")
    execute_process(COMMAND
        ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/data"
        "${PYTHON_BINARY_DIR}/pylada/espresso/tests/data"
    )
  endif()
  add_subdirectory(bdd)
endif(tests)
