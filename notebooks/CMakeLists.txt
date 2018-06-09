file(COPY "Creating a Job Folder.ipynb"
        "IPython high-throughput interface.ipynb"
    DESTINATION "${PYTHON_BINARY_DIR}/pylada/notebooks")
install(FILES "Creating a Job Folder.ipynb"
        "IPython high-throughput interface.ipynb"
    DESTINATION "${PYTHON_PKG_DIR}/pylada/crystal/notebooks/")

if(tests)
    add_pytest(test_*.py INSTALL PREFIX "pylada.notebooks" FAKE_INIT LABELS
        notebooks)
endif(tests)
