CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
project(Pylada
    VERSION 1.1.0
    DESCRIPTION "A pythonic computation material science platform"
    HOMEPAGE_URL "https://www.github.com/pylada/pylada-light"
    LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-Wall)
endif()

# Download automatically, you can also just copy the conan.cmake file
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
        "${CMAKE_BINARY_DIR}/conan.cmake")
endif()
include(${CMAKE_BINARY_DIR}/conan.cmake)


if(NOT CONAN_OPTIONS)
    set(CONAN_OPTIONS "ssht:conan_fftw=True")
endif()
if(NOT CONAN_BUILD)
    set(CONAN_BUILD "missing")
endif()
set(_GLIBCXX_USE_CXX11_ABI TRUE)

conan_check(REQUIRED)
conan_cmake_run(REQUIRES eigen/3.3.5@conan/stable
    BASIC_SETUP
    OPTIONS "${CONAN_OPTIONS}"
    KEEP_RPATHS
    CMAKE_TARGETS
    NO_OUTPUT_DIRS
    BUILD ${CONAN_BUILD}
    # import headers to help vim find them
    IMPORTS "include, * -> conan/include")

option(tests "Compile tests." on)
option(external_program_tests
    "Allow tests which run external programs (vasp, crystal...)" off)
if(tests)
  enable_testing()
endif(tests)

set(OLD_CMAKE_FIND_FRAMEWORK "${CMAKE_FIND_FRAMEWORK}")
set(CMAKE_FIND_FRAMEWORK LAST)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
set(CMAKE_FIND_FRAMEWORK "${OLD_CMAKE_FIND_FRAMEWORK}")

if(NOT SKBUILD)
    # get numpy include directory
    execute_process(
        COMMAND
        ${Python3_EXECUTABLE} -c "import skbuild; print(skbuild.__file__)"
        OUTPUT_VARIABLE SKBUILD_LOCATION
        RESULT_VARIABLE SKBUILD_FOUND
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT SKBUILD_FOUND EQUAL 0 OR NOT SKBUILD_LOCATION)
        message(FATAL_ERROR "Could not find numpy includes")
    else()
        set(SKBUILD_FOUND True)
        get_filename_component(SKBUILD_LOCATION "${SKBUILD_LOCATION}" DIRECTORY)
    endif()
    message(STATUS "Found skbuild at ${SKBUILD_LOCATION}")
    list(APPEND CMAKE_MODULE_PATH "${SKBUILD_LOCATION}/resources/cmake")
endif()

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(Eigen3 NO_MODULE)

add_subdirectory(pylada)
