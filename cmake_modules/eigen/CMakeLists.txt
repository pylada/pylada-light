project(external CXX)
cmake_minimum_required(VERSION 3.12)
include("ExternalProject")

if(NOT EIGEN_TAR_ARCHIVE)
    set(EIGEN_TAR_ARCHIVE https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz)
endif()
ExternalProject_Add(
    Eigen
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external"
    URL ${EIGEN_TAR_ARCHIVE}
    URL_HASH MD5=9e30f67e8531477de4117506fe44669b
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON
        -DEIGEN_BUILD_PKGCONFIG=OFF
        -DBUILD_TESTING=OFF
        -DEIGEN_TEST_CXX11=ON
)
