from pytest_bdd import scenarios, given, when, then, parsers
import pytest

scenarios("features/restart.feature")


@pytest.fixture
def passon():
    """ A container to pass information from when to then """
    return []


@given(parsers.parse("a pwscf object setup as follows\n{text}"))
def pwscf(text):
    from pylada.espresso.tests.bdd.test_single_run import pwscf
    return pwscf(text)


@given("a distorted diamond structure")
def distorted_diamond():
    from numpy.random import random
    from pylada.espresso.tests.fixtures import diamond_structure
    structure = diamond_structure()
    structure[1].pos += random(3) * 0.01 - 0.005
    structure.cell += random((3, 3)) * 0.01 - 0.005
    return structure


@given("we run pwscf once")
def extract(tmpdir, distorted_diamond, pwscf):
    return pwscf(distorted_diamond, tmpdir.join("first"))


@when("we iterate through the second chain called to pwscf")
def iter_second_call(tmpdir, extract, pwscf, distorted_diamond):
    assert extract.input_path == tmpdir.join("first", "pwscf.in")
    assert extract.output_path == tmpdir.join("first", "pwscf.out")
    assert extract.success
    from six import next
    iterator = pwscf.iter(distorted_diamond, tmpdir.join("second"), restart=extract)
    next(iterator)


@when("we follow with a static calculation")
def run_second(tmpdir, extract, pwscf, distorted_diamond, passon):
    assert extract.input_path == tmpdir.join("first", "pwscf.in")
    assert extract.output_path == tmpdir.join("first", "pwscf.out")
    assert extract.success
    pwscf.control.calculation = None
    passon.append(pwscf(distorted_diamond, tmpdir.join("second"), restart=extract))


@then("the structure on input is the output of the first call")
def structure_is_passed_on(tmpdir, extract):
    from numpy import allclose
    from pylada.espresso import read_structure
    actual = read_structure(str(tmpdir.join("second", "pwscf.in")))
    assert len(actual) == len(extract.structure)
    assert allclose(actual.cell, extract.structure.cell)
    for a, b in zip(actual, extract.structure):
        assert a.type == b.type
        assert allclose(a.pos, b.pos)


@then("the second run is successful")
def check_second_success(tmpdir, passon):
    assert passon[-1].success


@then("the initial structure is the output of the first call")
def check_initial_structure(tmpdir, extract, passon):
    from numpy import allclose
    actual = passon[-1].initial_structure
    assert len(actual) == len(extract.structure)
    assert allclose(actual.cell, extract.structure.cell)
    for a, b in zip(actual, extract.structure):
        assert a.type == b.type
        assert allclose(a.pos, b.pos)


@then("the wavefunctions file has been copied over")
def check_wfcn(tmpdir, extract):
    assert tmpdir.join("second", "pwscf.wfc1").check(file=True)
    expected_hash = tmpdir.join("first", "pwscf.wfc1").computehash()
    actual_hash = tmpdir.join("second", "pwscf.wfc1").computehash()
    assert expected_hash == actual_hash


@then("pwscf is told to start from the wavefunction file")
def check_start_from_wfcn(tmpdir):
    from pylada.espresso import Pwscf
    pwscf = Pwscf()
    pwscf.read(tmpdir.join("second", "pwscf.in"))
    assert pwscf.electrons.startingwfc == 'file'


@then("the second run restarted form the wavefunctions")
def check_restarted_from_wfc(passon):
    extract = passon[-1]
    assert extract.started_from_wavefunctions_file
