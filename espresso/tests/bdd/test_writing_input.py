import pytest
from pytest_bdd import given, scenarios, then, when

scenarios("features/writing_input.feature")


@given("a Pwscf object")
def empty_pwscf():
    from pylada.espresso import Pwscf
    return Pwscf()


@given("mandatory attribute pwscf.system.ecutwfc is set to 14.0 Ry")
def set_ecutwfc(empty_pwscf):
    import quantities
    empty_pwscf.system.ecutwfc = 14.0 * quantities.Ry


@given("pwscf.electrons.whawha is set to 1.5")
def set_an_attribute0(empty_pwscf):
    setattr(getattr(empty_pwscf, "electrons"), "whawha", 1.5)


@when("writing to pwscf.in without specifying a structure")
def writing_no_structure(empty_pwscf, tmpdir):
    empty_pwscf.write(tmpdir.join("pwscf.in"))


@then("the file pwscf.in appears and can be read")
def read_pwscf(tmpdir):
    from pylada.espresso import Pwscf
    assert tmpdir.join("pwscf.in").check(file=True)
    result = Pwscf()
    result.read(tmpdir.join("pwscf.in"))
    return result


pwscf_out = pytest.fixture(read_pwscf)


@then("pwscf.electrons.whawha is equal to 1.5")
def check_float_attribute_exists(pwscf_out):
    from numpy import abs
    assert hasattr(getattr(pwscf_out, "electrons"), "whawha")
    actual = float(getattr(getattr(pwscf_out, "electrons"), "whawha"))
    assert abs(actual - 1.5) < 1e-12


@then("wavefunction cutoff is equal to 14.0 Ry")
def check_ecutwfc(pwscf_out):
    import quantities
    from numpy import allclose
    assert pwscf_out.system.ecutwfc.units == quantities.Ry
    assert allclose(float(pwscf_out.system.ecutwfc), 14.0)
