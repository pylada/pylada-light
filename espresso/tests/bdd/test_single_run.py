from pytest_bdd import scenarios, given, when, then, parsers
import pytest

scenarios("features/single_run.feature")


@given("a Pwscf object")
def empty_pwscf():
    from pylada.espresso import Pwscf
    return Pwscf()


@given(parsers.parse("mandatory attribute pwscf.system.ecutwfc is set to {value:f} {units:S}"))
def set_ecutwfc(empty_pwscf, value, units):
    import quantities
    empty_pwscf.system.ecutwfc = value * getattr(quantities, units)


@given(parsers.parse("pwscf.{namelist:w}.{attribute:w} is set to {value:f}"))
def set_an_attribute0(empty_pwscf, namelist, attribute, value):
    # equivalent to pwscf.namelist.attribute = value
    # with namelist and attribute replaced by their values
    setattr(getattr(empty_pwscf, namelist), attribute, value)


@when(parsers.parse("writing to {filename:S} without specifying a structure"))
def writing_no_structure(empty_pwscf, tmpdir, filename):
    empty_pwscf.write(tmpdir.join(filename))


@then(parsers.parse("the file {filename:S} appears and can be read"))
def read_pwscf(tmpdir, filename):
    from pylada.espresso import Pwscf
    assert tmpdir.join(filename).check(file=True)
    result = Pwscf()
    result.read(tmpdir.join(filename))
    return result

pwscf_out = pytest.fixture(read_pwscf)


@then(parsers.parse("pwscf.{namelist:w}.{attribute:w} is equal to {value:f}"))
def check_float_attribute_exists(pwscf_out, namelist, attribute, value):
    from numpy import abs
    assert hasattr(getattr(pwscf_out, namelist), attribute)
    actual = float(getattr(getattr(pwscf_out, namelist), attribute))
    assert abs(actual - value) < 1e-12


@then(parsers.parse("wavefunction cutoff is equal to {value:f} {units:S}"))
def check_ecutwfc(pwscf_out, value, units):
    import quantities
    from numpy import allclose
    assert pwscf_out.system.ecutwfc.units == getattr(quantities, units)
    assert allclose(float(pwscf_out.system.ecutwfc), value)


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
    actual = read_pwscf(tmpdir, filename)
    assert abs(actual.system.ecutwfc - pwscf.system.ecutwfc) < 1e-8
    assert actual.kpoints.subtitle == pwscf.kpoints.subtitle
    assert actual.kpoints.value.rstrip().lstrip() == pwscf.kpoints.value.rstrip().lstrip()
