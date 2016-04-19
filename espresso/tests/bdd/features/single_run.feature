Feature: simple run

    Creates and launches a calculation. However, we do not actually launch the calculation. Instead
    we check that the functional is an iterator with two steps:

    - the first step yields a ProgrammProcess object that can execute an external binary
    - the second step yields an extraction object

    Calling pwscf directly (`pwscf(structure, outdir)`) is equivalent to iterating through the first
    step, executing it, and then returning the value of the second step. At the first iteration,
    all the input for pwscf should be present.

Background:
    Given a pwscf object setup as follows
        pwscf.system.ecutwfc = 12.0*Ry
        pwscf.kpoints.subtitle = None
        pwscf.kpoints.value = "2\n"\
            "0.25 0.25 0.75 3.0\n"\
            "0.25 0.25 0.25 1.0\n"
        pwscf.add_specie('Al', 'Al.pz-vbc.UPF')
    And a fake pseudo 'Al.pz-vbc.UPF' in the working directory
    And an aluminum structure


Scenario: check iteration through first step

    When iterating through the first step
    Then the yielded object is a ProgrammProcess
    And a valid pwscf.in exists
    And the marker file '.pylada_is_running' exists


Scenario: check iteration through the second step

    Given a serial communicator
    When iterating through the first step
    And executing the program process
    And iterating through the second step
    Then the yielded object is an Extract object
    And the extract object says the run is unsuccessful
    And the output file 'pwscf.out' exists
    And the output file 'pwscf.err' exists
    And the marker file '.pylada_is_running' has been removed
