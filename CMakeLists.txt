
## root CMakeLists.txt for pylada2, minimal portion of lada to support
# (1) crystal structure code
# (2) VASP
# (3) ewald
# (4) job scheduler
# (5) database
# Constructed by Peter Graf from Mayeul deAvezac's pylada CMakeLists.txt

# I intend this to be run via cmake (not ccmake).
# I am sick of the failures of the automated detection of stuff; automatically finding the _wrong_ thing is cause
# of much time wasted.  Therefore,
# I am thus expecting the user to HAND edit this file and enter the correct values for a couple of things:
#####
#  CMAKE_PYINSTALL_PREFIX, CMAKE_INSTALL_PREFIX, CMAKE_CXX_COMPILER, CMAKE_C_COMPILER, CMAKE_Fortran_COMPILER, PYTHON_LIBRARY, eigen_INCLUDE_DIR,
######

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (Pylada)
set (Pylada_VERSION_MAJOR 0)
set (Pylada_VERSION_MINOR 1)

########################################################################
##### edit (below) between these comments, hopefully no where else!
# install dir
set(CMAKE_PYINSTALL_PREFIX "/Users/pgraf/root/mylada")
set(CMAKE_INSTALL_PREFIX "/Users/pgraf/root/mylada")

# Languages
SET( CMAKE_CXX_COMPILER "g++-mp-4.5" )
SET( CMAKE_C_COMPILER "gcc-mp-4.5" )
SET( CMAKE_Fortran_COMPILER "gfortran-mp-4.5" )

## I find that if I just set "PYTHON_LIBRARY", then find_package works, and sets the rest of the necessary vars
set(PYTHON_LIBRARY  "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
## eigen header only library
set( eigen_INCLUDE_DIR  "/Users/pgraf/opt/eigen3.1.2/")

##### edit (above) between these comments, hopefully no where else!
########################################################################

# Location where cmake first looks for cmake modules.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#enable_language(C)
#enable_language(CXX)
enable_language(Fortran)
include(FortranCInterface)
FortranCInterface_HEADER(${PROJECT_BINARY_DIR}/FCMangle.h
                         MACRO_NAMESPACE "FC_"
                         SYMBOL_NAMESPACE "FC_" )

## python:
### these are unreliable
#find_package(PythonInterp REQUIRED)
#find_package(PythonLibs REQUIRED)
#find_package(PythonLibrary REQUIRED)
##
## I find that if I just set "PYTHON_LIBRARY", then find_package works, and sets the rest of the necessary vars
find_package(PythonLibs REQUIRED)

## numpy:
find_package(numpy REQUIRED)

## boost:
find_package(Boost 1.33.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

## eigen (v.3+)
set( PYLADA_WITH_EIGEN3 1)
include_directories(${eigen_INCLUDE_DIR})

#include root source directory
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")

#option(do_tests               "Compile tests."                             on)
#option(do_external_program_tests "Allow tests which run external programs (vasp, crystal...)" on)
set (do_tests on)  ## need this variable for subdirectories to trigger "add_test"; don't need it as "option" anymore if not using ccmake
if(do_tests)
  ENABLE_TESTING()
endif(do_tests)

# required for (at least) crystal 
add_subdirectory(crystal)
add_subdirectory(python)
add_subdirectory(math)
add_subdirectory(errors)
add_subdirectory(physics)
add_subdirectory(misc)

# additional requirements for (at least) vasp
add_subdirectory(vasp)
add_subdirectory(config)
add_subdirectory(tools)
add_subdirectory(process)

# with the above, ewald and jobfolder don't need anything else:
add_subdirectory(ewald)

add_subdirectory(jobfolder)
  
message (STATUS ${PROJECT_SOURCE_DIR})
configure_file (
 "${PROJECT_SOURCE_DIR}/PyladaConfig.h.in"
 "${PROJECT_BINARY_DIR}/PyladaConfig.h"
)


