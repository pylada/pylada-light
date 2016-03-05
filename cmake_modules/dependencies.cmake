include(PackageLookup)
set(PYLADA_WITH_EIGEN3 1)
lookup_package(Eigen3 REQUIRED)

find_package(CoherentPython REQUIRED)
include(PythonPackage)
function(find_or_fail package)
    find_python_package(${package})
    if(NOT ${package}_FOUND)
        message("*********")
        message("${package} required")
        message("It can likely be installed with pip")
        message("*********")
        message(FATAL_ERROR "Aborting")
    endif()
endfunction()

# first looks for python package, second for linkage/include stuff
find_or_fail(numpy)
find_package(Numpy REQUIRED)
find_or_fail(quantities)

# Create local python environment
# If it exists, most cookoff functions will use LOCAL_PYTHON_EXECUTABLE rather
# than PYTHON_EXECUTABLE. In practice, this means that packages installed in
# the build tree can be found.
include(EnvironmentScript)
set(LOCAL_PYTHON_EXECUTABLE "${PROJECT_BINARY_DIR}/localpython.sh")
create_environment_script(
    PYTHON
    EXECUTABLE "${PYTHON_EXECUTABLE}"
    PATH "${LOCAL_PYTHON_EXECUTABLE}"
)

if(tests)
    include(PythonPackageLookup)
    add_to_python_path("${EXTERNAL_ROOT}/python")
    lookup_python_package(pytest)
    # Not required per se but usefull for testing process
    find_python_package(mpi4py)
    find_program(MPIEXEC NAMES mpiexec mpirun)
endif()
