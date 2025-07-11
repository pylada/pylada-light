set(version "${Pylada_VERSION_MAJOR}, ${Pylada_VERSION_MINOR}, ${Pylada_VERSION_PATCH}")
set(verstr "${Pylada_VERSION_MAJOR}.${Pylada_VERSION_MINOR}.${Pylada_VERSION_PATCH}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
    "from collections import namedtuple\n\n\n"
    "Version = namedtuple('Version', ['major', 'minor', 'patch'])\n\n"
    "version_info = Version(${version})\n"
    "version = '${verstr}'\n"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py" DESTINATION ${PY_ROOT_DIR})

add_subdirectory(crystal)
add_subdirectory(decorations)
add_subdirectory(ewald)
