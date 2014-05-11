include(PackageLookup)
lookup_package(Boost 1.33 REQUIRED)
set(PYLADA_WITH_EIGEN3 1)
lookup_package(Eigen3 REQUIRED
    ARGUMENTS
        # Default use hg. This should be more portable.
        URL http://bitbucket.org/eigen/eigen/get/3.2.1.tar.gz
        MD5 a0e0a32d62028218b1c1848ad7121476
        TIMEOUT 60
)

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
