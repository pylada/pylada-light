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

void object_reset(PyObject*& _object, PyObject *_in)
{
  PyObject * const dummy(_object);
  _object = _in;
  Py_XINCREF(_object);
  Py_XDECREF(dummy);
}

bool object_equality_op(Object const& _self, Object const &_b)
{
  if(not _self)
    BOOST_THROW_EXCEPTION( error::internal());
                           //<< error::string("Object yet uninitialized.") );
  if(not _b)
    BOOST_THROW_EXCEPTION( error::internal());
                           //<< error::string("Object yet uninitialized.") );

  PyObject *globals = PyEval_GetBuiltins();
  if(not globals)
    BOOST_THROW_EXCEPTION( error::internal() );
                           //<< error::string("Could not get builtins.") );
  python::Object locals = PyDict_New();
  if(not locals)
    BOOST_THROW_EXCEPTION( error::internal() );
                           //<< error::string("Could not create local dict.") );
  if(PyDict_SetItemString(locals.borrowed(), "a", _self.borrowed()) < 0)
    BOOST_THROW_EXCEPTION( error::internal() );
                           //<< error::string("Could not set item in local dict.") );
  if(PyDict_SetItemString(locals.borrowed(), "b", _b.borrowed()) < 0)
    BOOST_THROW_EXCEPTION( error::internal() );
                           //<< error::string("Could not set item in local dict.") );
  
  python::Object const result(PyRun_String( "(a == b) == True", Py_eval_input, globals, 
                                            locals.borrowed() ));
  return result.borrowed() == Py_True;
};

std::ostream& operator<< (std::ostream &stream, Object const &_ob)
{
  PyObject* const repr = PyObject_Repr(_ob.borrowed());
  if(not repr) BOOST_THROW_EXCEPTION(error::internal());
  char const * const result = PyString_AS_STRING(repr);
  if(not result) BOOST_THROW_EXCEPTION(error::internal()); 
  return stream << result;
}
