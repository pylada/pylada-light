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

set(cython_sources
    _space_group.pyx _primitive.pyx _coordination_shells.pyx _map_sites.pyx)

foreach(filename ${cython_sources})
    get_filename_component(name "${filename}" NAME_WE)
    add_cython_target(cython_${name} ${filename} CXX)
    Python_add_library(${name} MODULE ${cython_${name}})
    target_link_libraries(${name} PRIVATE Eigen3::Eigen)
    target_include_directories(${name} PRIVATE
        "${PROJECT_SOURCE_DIR}" "${NumPy_INCLUDE_DIRS}")
    install(TARGETS ${name} LIBRARY DESTINATION pylada/crystal)
endforeach()

add_cython_target(cython_cutilities cutilities.pyx CXX)
Python_add_library(cutilities MODULE ${cython_cutilities} smith_normal_form.cc gruber.cc noopt.cc)
target_include_directories(cutilities PRIVATE 
    "${PROJECT_SOURCE_DIR}" "${NumPy_INCLUDE_DIRS}")
target_link_libraries(cutilities PRIVATE Eigen3::Eigen)
install(TARGETS cutilities LIBRARY DESTINATION pylada/crystal)

add_subdirectory(defects)
