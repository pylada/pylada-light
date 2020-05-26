CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
project(Pylada
    VERSION 1.0.1
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


option(tests "Compile tests." on)
option(external_program_tests
    "Allow tests which run external programs (vasp, crystal...)" off)
if(tests)
  enable_testing()
endif(tests)

if(CHECK_FOR_EIGEN_FIRST)
    find_package (Eigen3 3.3 NO_MODULE)
endif()
if(NOT TARGET Eigen3::Eigen)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${PROJECT_SOURCE_DIR}/cmake_modules/eigen
            -B ${CMAKE_CURRENT_BINARY_DIR}/eigen
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -G "${CMAKE_GENERATOR}"
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/eigen
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND}
            --build ${CMAKE_CURRENT_BINARY_DIR}/eigen
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND}
            --install ${CMAKE_CURRENT_BINARY_DIR}/eigen
    )
    find_package(Eigen3 PATHS ${CMAKE_CURRENT_BINARY_DIR}/eigen REQUIRED NO_MODULE)
    set(EIGEN_DOWNLOADED TRUE CACHE BOOL "eigen was downloaded")
endif()

set(OLD_CMAKE_FIND_FRAMEWORK "${CMAKE_FIND_FRAMEWORK}")
set(CMAKE_FIND_FRAMEWORK LAST)
find_package(Python3 COMPONENTS Development Interpreter)
set(CMAKE_FIND_FRAMEWORK "${OLD_CMAKE_FIND_FRAMEWORK}")
if(NOT Python3_FOUND)
    set(CMAKE_FIND_FRAMEWORK LAST)
    find_package(Python2 COMPONENTS Development Interpreter)
    set(CMAKE_FIND_FRAMEWORK "${OLD_CMAKE_FIND_FRAMEWORK}")
endif()
if(NOT Python3_FOUND AND NOT Python2_FOUND)
    message(FATAL_ERROR "Could find neither python3 nor python2")
endif()

if(Python3_FOUND)
    set(PYVER PY3)
else()
    set(PYVER PY2)
endif()

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

set(PY_ROOT_DIR "src/pylada")
set(PY_HEADER_DIR "${PROJECT_SOURCE_DIR}/src")
add_subdirectory(${PY_ROOT_DIR})

configure_file("tests/process/pifunctional.in.py"
    "${CMAKE_CURRENT_BINARY_DIR}/tests/process/pifunctional.py")
file(COPY "${CMAKE_CURRENT_BINARY_DIR}/tests/process/pifunctional.py"
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/tests/process)
