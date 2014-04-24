/******************************
   This file is part of PyLaDa.

   Copyright (C) 2013 National Renewable Energy Lab
  
   PyLaDa is a high throughput computational platform for Physics. It aims to make it easier to submit
   large numbers of jobs on supercomputers. It provides a python interface to physical input, such as
   crystal structures, as well as to a number of DFT (VASP, CRYSTAL) and atomic potential programs. It
   is able to organise and launch computational jobs on PBS and SLURM.
  
   PyLaDa is free software: you can redistribute it and/or modify it under the terms of the GNU General
   Public License as published by the Free Software Foundation, either version 3 of the License, or (at
   your option) any later version.
  
   PyLaDa is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
   Public License for more details.
  
   You should have received a copy of the GNU General Public License along with PyLaDa.  If not, see
   <http://www.gnu.org/licenses/>.
******************************/

PyObject* UnitQuantityClass()
{
  python::Object quant( PyImport_ImportModule("quantities") );
  if(not quant) return NULL;
  return PyObject_GetAttrString(quant.borrowed(), "UnitQuantity");
}
PyObject* QuantityClass()
{
  python::Object quant( PyImport_ImportModule("quantities") );
  if(not quant) return NULL;
  return PyObject_GetAttrString(quant.borrowed(), "Quantity");
}
bool check_quantity(PyObject *_in)
{
  if(not _in) return false;
  PyObject* quantity_class = QuantityClass();
  if(not quantity_class) 
  {
    PyErr_Clear();
    return false;
  }
  bool const resultA = PyObject_IsInstance(_in, quantity_class) == 1;
  Py_DECREF(quantity_class);
  if(resultA) return true;
  quantity_class = UnitQuantityClass();
  if(not quantity_class) 
  {
    PyErr_Clear();
    return false;
  }
  bool const resultB = PyObject_IsInstance(_in, quantity_class) == 1;
  Py_DECREF(quantity_class);
  return resultB;
}

PyObject *fromC_quantity(types::t_real const &_double, std::string const &_units)
{
  // creates global/local dictionary in order to run code.
  python::Object const globals = PyImport_ImportModule("quantities");
  if(not globals) return NULL;
  python::Object locals = PyDict_New();
  if(not locals) return NULL;
  python::Object number = PyFloat_FromDouble(_double);
  if(not number) return NULL;
  python::Object units = PyString_FromString(_units.c_str());
  if(not units) return NULL;
  if(PyDict_SetItemString(locals.borrowed(), "number", number.borrowed()) < 0)
    return NULL;
  if(PyDict_SetItemString(locals.borrowed(), "units", units.borrowed()) < 0)
    return NULL;
  python::Object result(PyRun_String( "quantity.Quantity(number, units)", Py_eval_input,
                              PyModule_GetDict(globals.borrowed()), 
                              locals.borrowed() ));
  if(not result) return NULL;
  return result.release();
}

PyObject *fromC_quantity(types::t_real const &_double, PyObject *_unittemplate)
{
  // creates global/local dictionary in order to run code.
  python::Object const globals = PyImport_ImportModule("quantities");
  if(not globals) return NULL;
  python::Object locals = PyDict_New();
  if(not locals) return NULL;
  python::Object number = PyFloat_FromDouble(_double);
  if(not number) return NULL;
  if(PyDict_SetItemString(locals.borrowed(), "number", number.borrowed()) < 0)
    return NULL;
  if(PyDict_SetItemString(locals.borrowed(), "units", _unittemplate) < 0)
    return NULL;
  python::Object result(PyRun_String( "quantity.Quantity(number, units.units)", Py_eval_input,
                              PyModule_GetDict(globals.borrowed()), 
                              locals.borrowed() ));
  if(not result) return NULL;
  return result.release();
}

bool PyQuantity_Convertible(PyObject *_a, PyObject *_b)
{
  if(not check_quantity(_a)) return false;
  if(not check_quantity(_b)) return false;
  char rescale_str[] = "rescale";
  char s_str[] = "O";
  python::Object units = PyObject_GetAttrString(_b, "units");
  if(not units) return false;
  PyObject *result = PyObject_CallMethod(_a, rescale_str, s_str, units.borrowed());
  if(PyErr_Occurred())
  {
    PyErr_Clear();
    Py_XDECREF(result);
    return false;
  }
  Py_XDECREF(result);
  return true;
}
PyObject *fromPy_quantity(PyObject *_number, PyObject *_units)
{
  if(not check_quantity(_units))
  {
    PYLADA_PYERROR(TypeError, "Expected a quantities object.");
    return NULL;
  }
  if(check_quantity(_number))
  {
    if(not PyQuantity_Convertible(_number, _units))
    {
      PYLADA_PYERROR(TypeError, "Input quantities are not convertible.");
      return NULL;
    }
    Py_INCREF(_number);
    return _number;
  }
  
  // creates global/local dictionary in order to run code.
  python::Object const globals = PyImport_ImportModule("quantities");
  if(not globals) return NULL;
  python::Object locals = PyDict_New();
  if(not locals) return NULL;
  if(PyDict_SetItemString(locals.borrowed(), "number", _number) < 0)
    return NULL;
  if(PyDict_SetItemString(locals.borrowed(), "unitdims", _units) < 0)
    return NULL;
  python::Object result(PyRun_String( "Quantity(number, unitdims.units)",
                              Py_eval_input,
                              PyModule_GetDict(globals.borrowed()), 
                              locals.borrowed() ));
  if(not result) return NULL;
  return result.release();
}


types::t_real get_quantity(PyObject *_in, std::string const &_units)
{
  if(not check_quantity(_in))
  {
    if(PyInt_Check(_in)) return types::t_real(PyInt_AS_LONG(_in));
    if(PyFloat_Check(_in)) return types::t_real(PyFloat_AS_DOUBLE(_in));
    PYLADA_PYERROR(TypeError, "Expected quantity or number in input.");
    return types::t_real(0);
  }
  // creates global/local dictionary in order to run code.
  python::Object const globals = PyImport_ImportModule("quantities");
  if(not globals) return types::t_real(0);
  python::Object locals = PyDict_New();
  if(not locals) return types::t_real(0);
  python::Object units = PyString_FromString(_units.c_str());
  if(not units) return types::t_real(0);
  if(PyDict_SetItemString(locals.borrowed(), "number", _in) < 0)
    return types::t_real(0);
  if(PyDict_SetItemString(locals.borrowed(), "units", units.borrowed()) < 0)
    return types::t_real(0);
  python::Object const result(PyRun_String( "float(number.rescale(units))", Py_eval_input,
                                    PyModule_GetDict(globals.borrowed()), 
                                    locals.borrowed() ));
  if(not result) return types::t_real(0);
  types::t_real const realresult = PyFloat_AsDouble(result.borrowed());
  if(PyErr_Occurred()) return types::t_real(0);
  return realresult;
}
types::t_real get_quantity(PyObject *_number, PyObject *_units)
{
  if(not check_quantity(_number))
  {
    PYLADA_PYERROR(TypeError, "get_quantity: First argument should be a quantity.");
    return types::t_real(0);
  }
  if(not check_quantity(_units))
  {
    PYLADA_PYERROR(TypeError, "get_quantity: Second argument should be a quantity.");
    return types::t_real(0);
  }
  // creates global/local dictionary in order to run code.
  python::Object const globals = PyImport_ImportModule("quantities");
  if(not globals) return types::t_real(0);
  python::Object locals = PyDict_New();
  if(not locals) return types::t_real(0);
  if(PyDict_SetItemString(locals.borrowed(), "number", _number) < 0)
    return types::t_real(0);
  if(PyDict_SetItemString(locals.borrowed(), "units", _units) < 0)
    return types::t_real(0);
  python::Object const result(PyRun_String( "float(number.rescale(units.units))",
                                    Py_eval_input,
                                    PyModule_GetDict(globals.borrowed()), 
                                    locals.borrowed() ));
  if(not result) return types::t_real(0);
  types::t_real const realresult = PyFloat_AsDouble(result.borrowed());
  if(PyErr_Occurred()) return types::t_real(0);
  return realresult;
}

types::t_real get_quantity(PyObject *_in)
{
  PyObject *globals = PyEval_GetBuiltins();
  if(not globals) return types::t_real(0);
  python::Object locals = PyDict_New();
  if(not locals) return types::t_real(0);
  if(PyDict_SetItemString(locals.borrowed(), "number", _in) < 0)
    return types::t_real(0);
  python::Object const result(PyRun_String( "float(number)", Py_eval_input,
                                    globals, 
                                    locals.borrowed() ));
  if(not result) return types::t_real(0);
  types::t_real const realresult = PyFloat_AsDouble(result.borrowed());
  if(PyErr_Occurred()) return types::t_real(0);
  return realresult;
}
