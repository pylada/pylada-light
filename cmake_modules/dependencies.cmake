include(PackageLookup)
set(PYLADA_WITH_EIGEN3 1)
lookup_package(Eigen3 REQUIRED)

find_package(CoherentPython REQUIRED)
include(PythonPackage)
include(PythonPackageLookup)
find_python_package(numpy)
find_python_package(quantities)
# only needed for build. So can install it locally in build dir.
lookup_python_package(cython)
# Finds additional info, like libraries, include dirs...
# no need check numpy features, it's all handled by cython.
set(no_numpy_feature_tests TRUE)
find_package(Numpy REQUIRED)

# Create local python environment
# If it exists, most cookoff functions will use LOCAL_PYTHON_EXECUTABLE rather
# than PYTHON_EXECUTABLE. In practice, this means that packages installed in
# the build tree can be found.
include(EnvironmentScript)
add_to_python_path("${PROJECT_BINARY_DIR}/python")
add_to_python_path("${EXTERNAL_ROOT}/python")
set(LOCAL_PYTHON_EXECUTABLE "${PROJECT_BINARY_DIR}/localpython.sh")
create_environment_script(
    PYTHON
    EXECUTABLE "${PYTHON_EXECUTABLE}"
    PATH "${LOCAL_PYTHON_EXECUTABLE}"
)

if(tests)
    lookup_python_package(pytest)
    # Not required per se but usefull for testing process
    find_python_package(mpi4py)
    find_program(MPIEXEC NAMES mpiexec mpirun)
endif()
