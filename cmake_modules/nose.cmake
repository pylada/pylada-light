# Adds function to create nose tests
include(FindPackageHandleStandardArgs)
include(PythonModule)
# add_nose_test(<source>       # There should be a test_${source}.py file
#                              # or a ${source} file.
#     [NAME <name> ]           # Name for the ctest entry
#     [INSTALL <fully qualified module name> # Location where to install
#         # test in package. Does not install by default.
#     [WORKING_DIRECTORY <wd>] # Launches test from here
#     [LABELS <label list>]    # Test labels, on top of python and nose.
# )
function(add_nose_test testname)
    get_filename_component(abs_testname ${testname} ABSOLUTE)
    if(EXISTS "${abs_testname}")
        set(source "${abs_testname}")
        string(REGEX REPLACE
            ".*/tests?_?(.*)\\.py" "\\1"
            testname "${abs_testname}"
        )
    else()
        get_filename_component(source test_${testname}.py ABSOLUTE)
        if(NOT EXISTS "${source}")
            message(FATAL_ERROR "Could not find test file ${source}")
        endif()
    endif()
    cmake_parse_arguments(${testname}
        "" "NAME;INSTALL;WORKING_DIRECTORY"
        "LABELS" ${ARGN}
    )
    if(${testname}_NAME)
        set(name ${${testname}_NAME})
    elseif(${testname}_INSTALL)
        set(name ${${testname}_INSTALL}.${testname})
    else()
        set(name ${testname})
    endif()
    if(LOCAL_PYTHON_EXECUTABLE)
        set(exec "${LOCAL_PYTHON_EXECUTABLE}")
    elseif(PYTHON_EXECUTABLE)
        set(exec "${PYTHON_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Python executable not  set")
    endif()
    if(${testname}_INSTALL)
        add_python_module(${${testname}_INSTALL} PYFILES ${source})
    endif()
    if(${testname}_WORKING_DIRECTORY)
        set(working_directory "${${testname}_WORKING_DIRECTORY}")
    else()
        set(working_directory "${CMAKE_PROJECT_BINARY_DIR}")
    endif()
    set(labels python nose)
    if(${testname}_LABELS)
        list(APPEND labels ${${testname}_LABELS})
        list(REMOVE_DUPLICATES labels)
    endif()
    set(expression
       "import nose"
       "from sys import exit"
       "exit(nose.run() != True)"
    )
    add_test( NAME ${name}
        WORKING_DIRECTORY ${working_directory}
        COMMAND ${exec} -c "${expression}" ${source}
    )
    set_tests_properties(${name} PROPERTIES LABELS "${labels}")
endfunction()

# Multiple calls over add_nose_test: different sources same arguments
# add_nose_tests(<test0> <test1> ...
#    [arguments for add_nose_test]
# )
function(add_nose_tests)
    # Split list into test sources and test arguments
    set(options NAME INSTALL WORKING_DIRECTORY LABELS EXCLUDE)
    unset(patterns)
    unset(otherargs)
    set(intests TRUE)
    foreach(argument ${ARGN})
        if(intests)
            list(FIND options "${argument}" is_option)
            if(is_option EQUAL -1)
                list(APPEND patterns "${argument}")
            else()
                list(APPEND otherargs "${argument}")
                set(intests FALSE)
            endif()
        else()
            list(APPEND otherargs "${argument}")
        endif()
    endforeach()

    unset(excluded)
    unset(notexcluded)
    set(doing_which 0)
    foreach(argument ${otherargs})

        if(doing_which EQUAL 0 AND "${argument}" STREQUAL "EXCLUDE")
            set(doing_which 1)
        elseif(doing_which EQUAL 1)
            list(FIND options "${argument}" is_option)
            if(is_option EQUAL -1)
                list(APPEND excluded "${argument}")
            else()
                set(doing_which 2)
                list(APPEND notexcluded "${argument}")
            endif()
        else()
            list(APPEND notexcluded "${argument}")
        endif()
    endforeach()
    file(GLOB nosetests ${patterns})
    if(NOT "${excluded}" STREQUAL "")
        file(GLOB excluded ${excluded})
        if(NOT "${excluded}" STREQUAL "")
            list(REMOVE_ITEM nosetests ${excluded})
        endif()
    endif()
    foreach(nosetest ${nosetests})
        add_nose_test(${nosetest} ${notexcluded})
    endforeach()
endfunction()
