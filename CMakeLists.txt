CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (Pylada C CXX Fortran)
set (Pylada_VERSION_MAJOR 0)
set (Pylada_VERSION_MINOR 1)


########################################################################
## These variables are key; if you need to set them by hand, do so here, or
## better yet, via a separate file with entries like:
#SET( CMAKE_CXX_COMPILER "g++-mp-4.5"  CACHE STRING "c++")
## (note the 'CACHE STRING "xyz" ' is necessary)
# install dir
#set(CMAKE_PYINSTALL_PREFIX "/Users/pgraf/root/mylada")
#set(CMAKE_INSTALL_PREFIX "/Users/pgraf/root/mylada")

# Languages
#SET( CMAKE_CXX_COMPILER "g++-mp-4.5" )
#SET( CMAKE_C_COMPILER "gcc-mp-4.5" )
#SET( CMAKE_Fortran_COMPILER "gfortran-mp-4.5" )

## I find that if I just set "PYTHON_LIBRARY", then find_package works, and sets the rest of the necessary vars
#set(PYTHON_LIBRARY  "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
## eigen header only library
#set( eigen_INCLUDE_DIR  "/Users/pgraf/opt/eigen3.1.2/")

##### edit (above) between these comments, hopefully no where else!
########################################################################

# Location where cmake first looks for cmake modules.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
include(LookUp-GreatCMakeCookOff)

include(FortranCInterface)
FortranCInterface_HEADER(${PROJECT_BINARY_DIR}/FCMangle.h
                         MACRO_NAMESPACE "FC_"
                         SYMBOL_NAMESPACE "FC_" )

include(AddCPP11Flags)

option(tests "Compile tests." on)
option(external_program_tests
    "Allow tests which run external programs (vasp, crystal...)" off)
if(tests)
  enable_testing()
endif(tests)

# Look for dependencies, download or fail as case may be
include(dependencies)

include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
if(EIGEN3_INCLUDE_DIR)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Adds cmake function to create python modules
include(PythonModule)
if(tests)
  include(AddPyTest)
  setup_pytest("${EXTERNAL_ROOT}/python" "${PROJECT_BINARY_DIR}/py.test.sh")
endif()

# required for (at least) crystal
add_subdirectory(errors)
add_subdirectory(crystal)
add_subdirectory(python)
add_subdirectory(physics)
add_subdirectory(misc)
add_subdirectory(tools)
add_subdirectory(ewald)
add_subdirectory(decorations)
add_subdirectory(config)
add_subdirectory(periodic_table)
add_subdirectory(vasp)
add_subdirectory(process)
add_subdirectory(jobfolder)
add_subdirectory(ipython)
add_subdirectory(espresso)

# Add documentation target: make documentation
# include(sphinx)
# with the above, ewald and jobfolder and ipython don't need anything else:
#

#add_custom_target(doc
#      COMMAND make html
#      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sphinx-doc)
