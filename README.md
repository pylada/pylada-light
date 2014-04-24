pylada-crystal
==============

A python computational physics framework

Minimal version of pylada necessary to just run the crystal,VASP,ewald,jobs,and database modules

Constructed by Peter Graf from Mayeul de'Avezac's pylada

concerning installing:
 I intend this to be built via cmake (not necessarily ccmake).  
 I am sick of the failures of the automated detection of stuff; automatically finding the _wrong_ thing is cause
 of much time wasted.  Therefore,
 I am thus expecting the user to HAND edit the root CMakeLists.txt file and enter the correct values for a couple of things:
    CMAKE_CXX_COMPILER, PYTHON_LIBRARY, eigen_INCLUDE_DIR, CMAKE_PYINSTALL_PREFIX, CMAKE_INSTALL_PREFIX
