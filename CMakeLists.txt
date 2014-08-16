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
option(tests "Compile tests." on)
option(external_program_tests
    "Allow tests which run external programs (vasp, crystal...)" on)
if(tests)
  enable_testing()
endif(tests)

# Look for dependencies, download or fail as case may be
include(dependencies)

include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
foreach(name Boost EIGEN3)
    if(NOT "${${name}_INCLUDE_DIR}" STREQUAL "" AND ${name}_INCLUDE_DIR)
        include_directories(${${name}_INCLUDE_DIR})
    endif()
endforeach()

option(tests               "Compile tests."                             on)
option(external_program_tests "Allow tests which run external programs (vasp, crystal...)" on)
if(tests)
  enable_testing()
endif(tests)

# Easily add python modules
include(PythonModule)
# Easily add node tests
if(tests)
    include(nose)
endif()


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

# enum is for defects
add_subdirectory(enum)

# Add documentation target: make documentation
include(sphinx)
# with the above, ewald and jobfolder and ipython don't need anything else:
add_subdirectory(ewald)
add_subdirectory(jobfolder)
add_subdirectory(ipython)

# after all that, turns out nrelmatdb uses this:
add_subdirectory(periodic_table)

#add_custom_target(doc 
#      COMMAND make html
#      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sphinx-doc)

  
message (STATUS ${PROJECT_SOURCE_DIR})
message (STATUS ${PROJECT_BINARY_DIR})
# Create general configuration file for C stuff
configure_file (
 "${PROJECT_SOURCE_DIR}/PyladaConfig.h.in"
 "${PROJECT_BINARY_DIR}/PyladaConfig.h"
)


