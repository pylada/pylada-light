Feature: single run

    Setting up and running a single espresso calculation

Scenario: Writing a pwscf input file

        pwscf.system.ecutwfc is the one parameter that must be set prior to calling the functional
        (or writing it's input)

    Given a Pwscf object
    And mandatory attribute pwscf.system.ecutwfc is set to 14.0 Ry
    When writing to pwscf.in without specifying a structure
    Then the file pwscf.in appears and can be read
    And wavefunction cutoff is equal to 14.0 Ry


Scenario: Writing a pwscf input file with an unknown parameter

        All attributes in the pwscf namelists are automatically written to file, even those that
        have not yet been setup in pylada. This means any keyword needed by pwscf can be added
        without changing the code. The only restrictions are the allowed types: string, integers,
        floating points and arrays of integers and floating points

    Given a Pwscf object
    And mandatory attribute pwscf.system.ecutwfc is set to 14.0 Ry
    And pwscf.electrons.whawha is set to 1.5
    When writing to pwscf.in without specifying a structure
    Then the file pwscf.in appears and can be read
    And wavefunction cutoff is equal to 14.0 Ry
    And pwscf.electrons.whawha is equal to 1.5


Scenario: Launching a (fake) pwscf calculation and check input

        Creates and launches a calculation. However, we do not actually launch the calculation.
        Instead we check that the functional is an iterator with two steps:

        - the first step yields a ProgrammProcess object that can execute an external binary
        - the second step yields an extraction object

        Calling pwscf directly (`pwscf(structure, outdir)`) is equivalent to iterating through the
        first step, executing it, and then returning the value of the second step.
        At the first iteration, all the input for pwscf should be present.

    Given a pwscf object setup as follows
        pwscf.system.ecutwfc = 12.0*Ry
        pwscf.kpoints.subtitle = None
        pwscf.kpoints.value = "2\n"\
            "0.25 0.25 0.75 3.0\n"\
            "0.25 0.25 0.25 1.0\n"
        pwscf.add_specie('Al', 'Al.pz-vbc.UPF')
    And a fake pseudo 'Al.pz-vbc.UPF' in the working directory
    And an aluminum structure
    When iterating through the first step
    Then the yielded object is a ProgrammProcess
    And a valid pwscf.in exists
