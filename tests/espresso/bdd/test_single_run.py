import re

import pytest
from pytest_bdd import given, scenarios, then, when

scenarios("features/single_run.feature")


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
    pwscf.add_specie('Al', 'Al.pz-vbc.UPF')
    return pwscf


@given("a fake pseudo 'Al.pz-vbc.UPF' in the working directory")
def pseudo_filename(tmpdir):
    tmpdir.join('Al.pz-vbc.UPF').ensure(file=True)
    return tmpdir.join('Al.pz-vbc.UPF')


@given("an aluminum structure")
def aluminum():
    from quantities import bohr_radius
    from pylada.crystal.bravais import fcc
    result = fcc()
    result.scale = 7.5 * bohr_radius
    result[0].type = 'Al'
    return result


@given("a serial communicator")
def serialcomm():
    return {'n': 1}


@pytest.fixture
def passon():
    """ A container to pass information from when to then """
    return []


@pytest.fixture
def true(tmpdir):
    from sys import executable
    from stat import S_IREAD, S_IWRITE, S_IEXEC
    result = tmpdir.join("true.py")
    result.write("#! %s\nfrom sys import exit\nexit(0)" % executable)
    result.chmod(S_IREAD | S_IWRITE | S_IEXEC)
    return result

@when("iterating through the first step")
def first_step(pwscf, tmpdir, aluminum, passon, true):
    from six import next
    iterator = pwscf.iter(aluminum, tmpdir, program=str(true))
    passon.extend([iterator, next(iterator)])


@when("executing the program process")
def execute_program(passon, serialcomm):
    passon[-1].start({'n': 1})
    passon[-1].wait()


@when("iterating through the second step")
def second_step(passon):
    from six import next
    iterator = passon[0]
    passon.append(next(iterator))


@when("running pwscf")
def run_nonscf(tmpdir, aluminum, pwscf, passon):
    from pylada.espresso.tests.bdd.fixtures import copyoutput, data_path
    src = data_path("nonscf")
    program = copyoutput(tmpdir, src, tmpdir)
    passon.append(pwscf(aluminum, tmpdir, program=str(program)))


@then("the yielded object is a ProgrammProcess")
def first_yield(passon):
    from pylada.process import ProgramProcess
    iterator, first_step = passon
    assert isinstance(first_step, ProgramProcess)


@then("the yielded object is an Extract object")
def second_yield(passon):
    from pylada.espresso.extract import Extract
    extract = passon[-1]
    assert isinstance(extract, Extract)


@then("a valid pwscf.in exists")
def check_pwscf_input(tmpdir, pwscf):
    from pylada.espresso import Pwscf
    actual = Pwscf()
    actual.read(tmpdir.join("pwscf.in"))
    assert abs(actual.system.ecutwfc - pwscf.system.ecutwfc) < 1e-8
    assert actual.kpoints.subtitle == pwscf.kpoints.subtitle
    assert actual.kpoints.value.rstrip().lstrip() == pwscf.kpoints.value.rstrip().lstrip()


@then("the extract object says the run is unsuccessful")
def unsuccessfull_run(passon):
    extract = passon[-1]
    assert extract.success == False


@then("the marker file .pylada_is_running exists")
def check_marker_file(tmpdir):
    assert tmpdir.join('.pylada_is_running').check(file=True)


@then('the output file pwscf.out exists')
def check_output_file(tmpdir):
    assert tmpdir.join("pwscf.out").check(file=True)


@then('the output file pwscf.err exists')
def check_err_file(tmpdir):
    assert tmpdir.join("pwscf.err").check(file=True)


@then("the marker file .pylada_is_running has been removed")
def check_marker_file_disappeared(tmpdir):
    assert tmpdir.join(".pylada_is_running").check(file=False)


@then("the run is successful")
def result_nonscf(passon):
    from pylada.espresso.extract import Extract
    assert isinstance(passon[0], Extract)
    assert passon[0].success
