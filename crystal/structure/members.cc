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

//! Returns a representation of the object.
PyObject* structure_repr(PyStructureObject* _self);
//! Returns a deepcopy of the atom.
PyObject* structure_copy(PyStructureObject* _self)
  { return (PyObject*) copy_structure(_self, NULL); }
//! Implements shallow copy.
PyObject* structure_shallowcopy(PyStructureObject* _self)
  { Py_INCREF(_self); return (PyObject*)_self; }
//! Returns a dictionary with same info as atom.
PyObject* structure_to_dict(PyStructureObject* _self);
//! Implements getstate for pickling.
PyObject* structure_getstate(PyStructureObject* _self);
//! Implements setstate for pickling.
PyObject* structure_setstate(PyStructureObject* _self, PyObject *_dict);
//! Implements reduce for pickling.
PyObject* structure_reduce(PyStructureObject* _self);
//! Implements add atom.
PyObject* structure_add_atom(PyStructureObject* _self, PyObject* _args, PyObject* _kwargs);
//! Implements structure affine transformation.
PyObject* structure_transform(PyStructureObject* _self, PyObject* _args);

//! Returns a representation of the object.
PyObject* structure_repr(PyStructureObject* _self)
{
  std::string name(_self->ob_type->tp_name);
  name = name.substr(name.rfind('.')+1);
  std::ostringstream result;
  // Create structure variable first.
  result << name << "( "
         <<         _self->cell(0, 0)
         << ", " << _self->cell(0, 1) 
         << ", " << _self->cell(0, 2) 
         << ",\n" << std::string(name.size()+2, ' ')
         <<         _self->cell(1, 0)
         << ", " << _self->cell(1, 1) 
         << ", " << _self->cell(1, 2) 
         << ",\n" << std::string(name.size()+2, ' ')
         <<         _self->cell(2, 0)
         << ", " << _self->cell(2, 1) 
         << ", " << _self->cell(2, 2)
         << ",\n" << std::string(name.size()+2, ' ')
         << "scale=" << python::get_quantity(_self->scale, "angstrom");
      
  // Including python dynamic attributes.
  if(_self->pydict != NULL)
  {
    if(PyDict_Size(_self->pydict) > 0)
    {
      PyObject *key, *value;
      Py_ssize_t pos = 0;
      while (PyDict_Next(_self->pydict, &pos, &key, &value)) 
      {
        PyObject* repr = PyObject_Repr(value);
        if(repr == NULL) return NULL;
        result << ", " << PyString_AsString(key);
        if(PyErr_Occurred() != NULL) {Py_DECREF(repr); return NULL;}
        result << "=" << PyString_AsString(repr);
        Py_DECREF(repr);
        if(PyErr_Occurred() != NULL) return NULL;
      }
    }
  }
  result << " )";
  // Then add atoms.
  if(_self->atoms.size() > 0)
  {
    std::vector<Atom>::const_iterator i_first = _self->atoms.begin();
    std::vector<Atom>::const_iterator const i_end = _self->atoms.end();
    result << "\\\n  .add_atom";
    {
      python::Object const atomstr = PyObject_Repr(i_first->borrowed());
      if(not atomstr) return NULL;
      std::string const atom = PyString_AsString(atomstr.borrowed());
      if(atom.empty()) return NULL;
      else if(atom.substr(0, 4) == "Atom") result << atom.substr(4);
      else result << "(" << atom << ")";
    }
    std::string const prefix("\\\n  .add_atom");
    for(++i_first; i_first != i_end; ++i_first)
    {
      python::Object atomstr = PyObject_Repr(i_first->borrowed());
      if(not atomstr) return NULL;
      std::string atom = PyString_AsString(atomstr.borrowed());
      if(atom.empty()) return NULL;
      else if(atom.substr(0, 4) == "Atom") result << prefix << atom.substr(4);
      else result << prefix << "(" << atom << ")";
    }
  }
  return PyString_FromString(result.str().c_str());
}

// Creates dictionary from atom with shallow copies.
PyObject *structure_to_dict(PyStructureObject* _self)
{
  python::Object result = PyDict_New();
  if(not result) return NULL;

  python::Object const cell = structure_getcell(_self, NULL);
  if(not cell) return NULL;
  if(PyDict_SetItemString(result.borrowed(), "cell", cell.borrowed()) < 0) return NULL;
  python::Object const scale = structure_getscale(_self, NULL);
  if(not scale) return NULL;
  if(PyDict_SetItemString(result.borrowed(), "scale", scale.borrowed()) < 0) return NULL;

  std::vector<Atom>::const_iterator i_atom = _self->atoms.begin();
  std::vector<Atom>::const_iterator const i_end = _self->atoms.end();
  char mname[] = "to_dict";
  for(long i(0); i_atom != i_end; ++i_atom, ++i)
  {
    // Gets dictionary description.
    python::Object const item = PyObject_CallMethod(i_atom->borrowed(), mname, NULL);
    if(not item) return NULL;
    // Then create pyobject index.
    python::Object const index = PyInt_FromLong(i);
    if(not index) return NULL;
    // finally, adds to dictionary.
    if(PyDict_SetItem(result.borrowed(), index.borrowed(), item.borrowed()) < 0) return NULL;
  }
  // Merge attribute dictionary if it exists.
  if(_self->pydict != NULL and PyDict_Merge(result.borrowed(), _self->pydict, 1) < 0) return NULL;

  return result.release();
}

// Implements __reduce__ for pickling.
PyObject* structure_reduce(PyStructureObject* _self)
{
  // Creates return tuple of three elements.
  python::Object type = PyObject_Type((PyObject*)_self);
  if(not type) return NULL;
  // Second element is a null tuple, argument to the callable type above.
  python::Object tuple = PyTuple_New(0);
  if(not tuple) return NULL;
  // Third element is the state of this object.
  char getstate[] = "__getstate__";
  python::Object state = PyObject_CallMethod((PyObject*)_self, getstate, NULL, NULL);
  if(not state) return NULL;
  python::Object iterator = structureiterator_create(_self);
  if(not iterator) return NULL;

  return PyTuple_Pack(4, type.borrowed(), tuple.borrowed(), state.borrowed(), iterator.borrowed());
}

// Implements getstate for pickling.
PyObject* structure_getstate(PyStructureObject* _self)
{
  // get cell attribute.
  python::Object cell = structure_getcell(_self, NULL);
  if(not cell) return NULL;
  // get python dynamic attributes.
  python::Object dict = _self->pydict == NULL ? python::Object::acquire(Py_None): PyDict_New();
  if(not dict) return NULL;
  if(_self->pydict != NULL and PyDict_Merge(dict.borrowed(), _self->pydict, 0) < 0) return NULL;

  return PyTuple_Pack(3, cell.borrowed(), _self->scale, dict.borrowed());
}

// Implements setstate for pickling.
PyObject* structure_setstate(PyStructureObject* _self, PyObject *_tuple)
{
  if(not PyTuple_Check(_tuple))
  {
    PYLADA_PYERROR(TypeError, "Expected state to be a tuple.");
    return NULL;
  }
  if(PyTuple_Size(_tuple) != 3)
  {
    PYLADA_PYERROR(TypeError, "Expected state to be a 4-tuple.");
    return NULL;
  }
  // first cell and scale.
  if(structure_setcell(_self, PyTuple_GET_ITEM(_tuple, 0), NULL) < 0) return NULL;
  _self->scale = PyTuple_GET_ITEM(_tuple, 1); 
  Py_INCREF(_self->scale);

  // finally, dictionary, so we can return without issue on error.
  PyObject *dict = PyTuple_GET_ITEM(_tuple, 2);
  if(dict == Py_None) { Py_RETURN_NONE; }
  if(_self->pydict == NULL)
  {
    _self->pydict = PyDict_New();
    if(_self->pydict == NULL) return NULL;
  }
  if(PyDict_Merge(_self->pydict, dict, 0) < 0) return NULL;
  Py_RETURN_NONE;
}

// Implements add atom.
PyObject* structure_add_atom(PyStructureObject* _self, PyObject* _args, PyObject* _kwargs)
{
  // Check first that _args is not a tuple containing an atom.
  if(PyTuple_Size(_args) == 1)
  {
    PyAtomObject* wrapper = (PyAtomObject*)PyTuple_GET_ITEM(_args, 0);
    if(check_atom(wrapper)) 
    {
      if(_kwargs != NULL)
      {
        PYLADA_PYERROR(TypeError, "Cannot insert an atom and motify in-place.");
        return NULL;
      }
      if(not wrapper)
      {
        PYLADA_PYERROR(internal, "Should never find an empty atom. Internal bug.");
        return NULL;
      }
      _self->atoms.push_back(Atom::acquire_((PyObject*)wrapper));
      Py_INCREF(_self);
      return (PyObject*)_self;
    }
  }
  // create new atom and its wrapper.
  Atom atom(_args, _kwargs);
  if(not atom) return NULL;
  // Add it to the container.
  _self->atoms.push_back(atom);
  // Finally, returns this very function for chaining.
  Py_INCREF(_self);
  return (PyObject*)_self;
}

void itransform_structure( PyStructureObject* _self,
                           Eigen::Matrix<types::t_real, 4, 3> const &_op )
{
  _self->cell = _op.block<3,3>(0,0) * _self->cell;
  std::vector<Atom>::iterator i_atom = _self->atoms.begin();
  std::vector<Atom>::iterator i_atom_end = _self->atoms.end();
  for(; i_atom != i_atom_end; ++i_atom)
    i_atom->pos() = _op.block<3,3>(0,0) * i_atom->pos() + ~_op.block<1, 3>(3, 0);
}
PyObject* structure_transform(PyStructureObject* _self, PyObject* _args)
{
  Eigen::Matrix<types::t_real, 4, 3> op;
  if(not python::numpy::convert_to_matrix(_args, op)) return NULL;
  itransform_structure(_self, op);
  Py_RETURN_NONE;
}
