if(tests)
  add_pytest(
      GLOB test_*.py
      INSTALL PREFIX "pylada.espresso.tests.bdd"
      LABELS espresso
      FAKE_INIT
  )
  if(NOT EXISTS "${PYTHON_BINARY_DIR}/pylada/espresso/tests/bdd/features/single_run.feature")
    file(MAKE_DIRECTORY "${PYTHON_BINARY_DIR}/pylada/espresso/tests/bdd")
    execute_process(COMMAND
        ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/features"
        "${PYTHON_BINARY_DIR}/pylada/espresso/tests/bdd/features"
    )
  endif()
endif()
