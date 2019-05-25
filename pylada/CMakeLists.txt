file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
    "from typing import NamedTuple\n\n\n"
    "class Version(NamedTuple):\n"
    "    major: int = ${Pylada_VERSION_MAJOR}\n"
    "    minor: int = ${Pylada_VERSION_MINOR}\n"
    "    patch: int = ${Pylada_VERSION_PATCH}\n"
    "\n"
    "    def __str__(self):\n"
    "        return f\"{self.major}.{self.minor}.{self.patch}\"\n"
    "\n\n"
    "version_info = Version()\n"
    "version = f\"{version_info}\"\n"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py" DESTINATION pylada)

add_subdirectory(crystal)
add_subdirectory(decorations)
add_subdirectory(ewald)
