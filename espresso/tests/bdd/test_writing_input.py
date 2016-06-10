from pytest_bdd import scenarios, given, when, then, parsers
import pytest

scenarios("features/writing_input.feature")


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
