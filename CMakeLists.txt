CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project (Pylada C CXX Fortran)
set (Pylada_VERSION_MAJOR 0)
set (Pylada_VERSION_MINOR 1)

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
    if(NOT "${${name}_INCLUDE_DIR}" STREQUAL "")
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

# Create general configuration file for C stuff
configure_file (
 "${PROJECT_SOURCE_DIR}/PyladaConfig.h.in"
 "${PROJECT_BINARY_DIR}/PyladaConfig.h"
)

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

# Add documentation target: make documentation
include(sphinx)
# with the above, ewald and jobfolder and ipython don't need anything else:
add_subdirectory(ewald)
add_subdirectory(jobfolder)
add_subdirectory(ipython)
