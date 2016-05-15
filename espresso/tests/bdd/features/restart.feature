Feature: run with restart

   Chains two calls to pwscf. This feature checks that the structure and the correct files are
   copied over from one pwscf run to another.

Background:
    Given a pwscf object setup as follows
        pwscf.system.ecutwfc = 12.0*Ry
        pwscf.kpoints.subtitle = None
        pwscf.kpoints.value = "2\n"\
            "0.25 0.25 0.75 3.0\n"\
            "0.25 0.25 0.25 1.0\n"
        pwscf.control.calculation = 'vc-relax'
        pwscf.cell.factor = 2.0
        pwscf.add_specie('Si', 'Si.pz-vbc.UPF')
    And a distorted diamond structure


Scenario: Check restart input

    Given we run pwscf once
    When we iterate through the second chain called to pwscf
    Then the structure on input is the output of the first call
    And the wavefunctions file has been copied over
    And the save directory has been copied over
    And pwscf is told to start from the wavefunction file


Scenario: Check restarted run

    Given we run pwscf once
    When we follow with a static calculation
    Then the second run is successful
    And the initial structure is the output of the first call
