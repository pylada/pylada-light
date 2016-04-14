Feature: single run

    Setting up and running a single espresso calculation

Scenario: Writing a pwscf input file

    pwscf.system.ecutwfc is the one parameter that must be set prior to calling the functional (or
    writing it's input)

    Given a Pwscf object
    And pwscf.system.ecutwfc is set to 14.0
    When writing to pwscf.in without specifying a structure
    Then the file pwscf.in appears and can be read
    And pwscf.system.ecutwfc is equal to 14.0


Scenario: Writing a pwscf input file with an unknown parameter

    All attributes in the pwscf namelists are automatically written to file, even those that have
    not yet been setup in pylada. This means any keyword needed by pwscf can be added without
    changing the code. The only restrictions are the allowed types: string, integers, floating
    points and arrays of integers and floating points

    Given a Pwscf object
    And pwscf.system.ecutwfc is set to 14.0
    # And pwscf.electrons.whawha is set to 1.5
    When writing to pwscf.in without specifying a structure
    Then the file pwscf.in appears and can be read
    And pwscf.system.ecutwfc is equal to 14.0
    # And pwscf.electrons.whawha is equal to 1.5

