find_program(RUNIPY_PROGRAM runipy PATHS ${EXTERNAL_ROOT}/python)
function(add_notebook_test filename)
  string(REPLACE " " "_" name "${filename}")
  configure_file("${filename}.ipynb" "${CMAKE_CURRENT_BINARY_DIR}/${name}.ipynb" COPYONLY)
  add_test(
    NAME notebook_${name}
    COMMAND
      ${LOCAL_PYTHON_EXECUTABLE} ${RUNIPY_PROGRAM}
      "${CMAKE_CURRENT_BINARY_DIR}/${name}.ipynb"
  )
  set_tests_properties(notebook_${name} PROPERTIES LABELS "python;notebook")
endfunction()

add_notebook_test("Creating a Job Folder")
add_notebook_test("IPython high-throughput interface")
