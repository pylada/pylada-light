find_package(Sphinx)
if(NOT SPHINX_FOUND)
    return()
endif()

if(NOT SPHINX_THEME)
    set(SPHINX_THEME default)
endif()

if(NOT SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()

# configured documentation tools and intermediate build results
set(DOC_BINARY_DIR "${PROJECT_BINARY_DIR}/sphinx/build")
set(DOC_OUTPUT_DIR "${PROJECT_BINARY_DIR}/sphinx")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file(
    "${PROJECT_SOURCE_DIR}/sphinx-doc/source/conf.in.py"
    "${DOC_BINARY_DIR}/_build/conf.py"
    @ONLY)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/source/*")
add_custom_target(documentation
    COMMAND ${LOCAL_PYTHON_EXECUTABLE} ${Sphinx_EXECUTABLE}
        -b html
        -c "${DOC_BINARY_DIR}/_build"
        -d "${DOC_BINARY_DIR}/_doctrees"
        "${PROJECT_SOURCE_DIR}/sphinx-doc/source"
        "${DOC_OUTPUT_DIR}"
    DEPENDS
        pylada_ipython_launch_copy
        pylada_jobfolder_copy pylada_vasp_copy
        pylada_jobfolder_tests_copy pylada_vasp_extract_copy
        pylada_math pylada_vasp_extract_tests_copy 
        pylada pylada_misc_copy pylada_vasp_incar_copy                 
        pylada_config_copy pylada_physics_copy pylada_vasp_incar_tests_copy           
        pylada_copy pylada_process_copy pylada_vasp_nlep_copy                  
        pylada_crystal pylada_crystal_copy pylada_ewald pylada_ewald_copy
        pylada_tools_copy pylada_tools_input_copy pylada_ipython_copy
    SOURCES ${SOURCES} 
    COMMENT "Building HTML documentation with Sphinx"
)

