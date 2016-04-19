from pytest_bdd import scenarios, given, when, then, parsers
import pytest

scenarios("features/single_run.feature")


@given(parsers.parse("a pwscf object setup as follows\n{text}"))
def pwscf(text):
    from quantities import Ry
    from pylada.espresso import Pwscf
    pwscf = Pwscf()
    exec(text, globals(), {'pwscf': pwscf, 'Ry': Ry})
    return pwscf


@given(parsers.parse("a fake pseudo '{filename}' in the working directory"))
def pseudo_filename(tmpdir, filename):
    tmpdir.join(filename).ensure(file=True)
    return tmpdir.join(filename)


@given("an aluminum structure")
def aluminum():
    from quantities import bohr_radius
    from pylada.crystal.bravais import fcc
    result = fcc()
    result.scale = 7.5 * bohr_radius
    result[0].type = 'Al'
    return result


@pytest.fixture
def passon():
    """ A container to pass information from when to then """
    return []


@when("iterating through the first step")
def first_step(pwscf, tmpdir, aluminum, passon):
    iterator = pwscf.iter(aluminum, tmpdir)
    passon.extend([iterator, iterator.next()])


@then("the yielded object is a ProgrammProcess")
def first_yield(passon):
    from pylada.process import ProgramProcess
    iterator, first_step = passon
    assert isinstance(first_step, ProgramProcess)


@then(parsers.parse("a valid {filename} exists"))
def check_pwscf_input(tmpdir, filename, pwscf):
    from pylada.espresso import Pwscf
    actual = Pwscf()
    actual.read(tmpdir.join(filename))
    assert abs(actual.system.ecutwfc - pwscf.system.ecutwfc) < 1e-8
    assert actual.kpoints.subtitle == pwscf.kpoints.subtitle
    assert actual.kpoints.value.rstrip().lstrip() == pwscf.kpoints.value.rstrip().lstrip()


@then(parsers.parse("the marker file '{filename}' exists"))
def check_marker_file(tmpdir, filename):
    assert tmpdir.join(filename).check(file=True)
