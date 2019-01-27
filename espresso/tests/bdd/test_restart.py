import pytest
from pytest_bdd import given, scenarios, then, when
from pylada.espresso.tests.fixtures import diamond_structure


@pytest.fixture
def passon():
    """ A container to pass information from when to then """
    return []


@given("a simple pwscf object")
def pwscf():
    from pylada.espresso import Pwscf
    from quantities import Ry
    pwscf = Pwscf()
    pwscf.system.ecutwfc = 12.0*Ry
    pwscf.kpoints.subtitle = None
    pwscf.kpoints.value = "2\n"\
        "0.25 0.25 0.75 3.0\n"\
        "0.25 0.25 0.25 1.0\n"
    pwscf.control.calculation = 'vc-relax'
    pwscf.cell.factor = 2.0
    pwscf.add_specie('Si', 'Si.pz-vbc.UPF')
    return pwscf


@given("a distorted diamond structure")
def distorted_diamond(diamond_structure):
    from numpy.random import random
    diamond_structure[1].pos += random(3) * 0.005 - 0.0025
    diamond_structure.cell += random((3, 3)) * 0.005 - 0.0025
    return diamond_structure


@given("we run pwscf once")
def extract(tmpdir, distorted_diamond, pwscf):
    from pylada.espresso.tests.bdd.fixtures import copyoutput, data_path
    src = data_path("restarted", "first")
    tmpdir.join("first", "Si.pz-vbc.UPF").ensure(file=True)
    program = copyoutput(
        tmpdir.join("first_copy.py"), src, tmpdir.join("first"))
    return pwscf(distorted_diamond, tmpdir.join("first"), program=str(program))


@when("we iterate through the second chain called to pwscf")
def iter_second_call(tmpdir, extract, pwscf, distorted_diamond):
    assert extract.input_path == tmpdir.join("first", "pwscf.in")
    assert extract.output_path == tmpdir.join("first", "pwscf.out")
    assert extract.success
    from six import next
    iterator = pwscf.iter(
        distorted_diamond, tmpdir.join("second"), restart=extract)
    next(iterator)


@when("we follow with a static calculation")
def run_second(tmpdir, extract, pwscf, distorted_diamond, passon):
    from pylada.espresso.tests.bdd.fixtures import copyoutput, data_path
    assert extract.input_path == tmpdir.join("first", "pwscf.in")
    assert extract.output_path == tmpdir.join("first", "pwscf.out")
    assert extract.success
    pwscf.control.calculation = None
    src = data_path("restarted", "second")
    tmpdir.join("second", "Si.pz-vbc.UPF").ensure(file=True)
    program = copyoutput(
        tmpdir.join("second_copy.py"), src, tmpdir.join("second"))
    passon.append(
        pwscf(
            distorted_diamond,
            tmpdir.join("second"),
            restart=extract,
            program=str(program)))


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


@then("the save directory has been copied over")
def check_save_dir(tmpdir, extract):
    save_dir = tmpdir.join("second", "%s.save" % extract.prefix)
    assert save_dir.check(dir=True)
    assert save_dir.join('charge-density.dat').check(file=True)


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


scenarios("features/restart.feature")
