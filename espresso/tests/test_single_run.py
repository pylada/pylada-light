from pytest_bdd import scenarios, given, when, then, parsers
import pytest

scenarios("features/single_run.feature")


@given("a Pwscf object")
def pwscf():
    from pylada.espresso import Pwscf
    return Pwscf()


@given(parsers.parse("pwscf.{namelist:w}.{attribute:w} is set to {value:f}"))
def set_an_attribute0(pwscf, namelist, attribute, value):
    # equivalent to pwscf.namelist.attribute = value
    # with namelist and attribute replaced by their values
    setattr(getattr(pwscf, namelist), attribute, value)

@when(parsers.parse("writing to {filename:S} without specifying a structure"))
def writing_no_structure(pwscf, tmpdir, filename):
    pwscf.write(tmpdir.join(filename))


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
