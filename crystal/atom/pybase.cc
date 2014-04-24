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

// atom getters/settters.
#include "getset.cc"
// atom member functions.
#include "members.cc"
// creation, deallocation, initialization.
#include "cdi.cc"

//! Creates a new atom with a given type.
PyAtomObject* PyAtom_NewWithArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs);
//! Creates a new atom.
PyAtomObject* new_atom()
{
  PyAtomObject* result = (PyAtomObject*) atom_type()->tp_alloc(atom_type(), 0);
  if(not result) return NULL;
  result->weakreflist = NULL;
  result->type = Py_None;
  Py_INCREF(Py_None);
  new(&result->pos) Pylada::math::rVector3d(0,0,0);
  result->pydict = PyDict_New();
  if(result->pydict == NULL) { Py_DECREF(result); return NULL; }
  return result;
}
//! Creates a new atom with a given type.
PyAtomObject* PyAtom_NewWithArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
{
  PyAtomObject* result = (PyAtomObject*)_type->tp_alloc(_type, 0);
  if(not result) return NULL;
  result->weakreflist = NULL;
  result->type = Py_None;
  Py_INCREF(Py_None);
  new(&result->pos) Pylada::math::rVector3d(0,0,0);
  result->pydict = PyDict_New();
  if(result->pydict == NULL) { Py_DECREF(result); return NULL; }
  return result;
}
//! Creates a new atom with a given type, also calling initialization.
PyAtomObject* new_atom(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
{
  PyAtomObject* result = PyAtom_NewWithArgs(_type, _args, _kwargs);
  if(result == NULL) return NULL;
  if(_type->tp_init((PyObject*)result, _args, _kwargs) < 0) {Py_DECREF(result); return NULL; }
  return result;
}

// Creates a deepcopy of atom.
PyAtomObject *copy_atom(PyAtomObject* _self, PyObject *_memo)
{
  PyAtomObject* result = (PyAtomObject*)_self->ob_type->tp_alloc(_self->ob_type, 0);
  if(not result) return NULL;
  result->weakreflist = NULL;
  new(&result->pos) Pylada::math::rVector3d(_self->pos);
  result->pydict = NULL;
  result->type = NULL;
  PyObject* copymod = PyImport_ImportModule("copy");
  if(copymod == NULL) return NULL;
  PyObject *deepcopystr = PyString_FromString("deepcopy");
  if(not deepcopystr) { Py_DECREF(copymod); return NULL; }
  if(_memo == NULL)
    result->type = PyObject_CallMethodObjArgs(copymod, deepcopystr, _self->type, NULL);
  else 
    result->type = PyObject_CallMethodObjArgs(copymod, deepcopystr, _self->type, _memo, NULL);
  if(result->type == NULL) { Py_DECREF(result); }
  else if(_self->pydict != NULL)
  {
    if(_memo == NULL)
      result->pydict = PyObject_CallMethodObjArgs(copymod, deepcopystr, _self->pydict, NULL);
    else
      result->pydict = PyObject_CallMethodObjArgs(copymod, deepcopystr, _self->pydict, _memo, NULL);
    if(result->pydict == NULL) { Py_DECREF(result); }
  }

  Py_DECREF(copymod);
  Py_DECREF(deepcopystr);
  return result;
}
// Returns pointer to atom type.
PyTypeObject* atom_type()
{
# ifdef PYLADA_DECLARE
#   error PYLADA_DECLARE already defined.
# endif
# define PYLADA_DECLARE(name, doc) \
    { const_cast<char*>(#name), (getter) pylada_atom_get ## name, \
      (setter) pylada_atom_set ## name, const_cast<char*>(doc) }
  
  static PyGetSetDef getsetters[] = {
      PYLADA_DECLARE(pos,  "Position in cartesian coordinates.\n\n"
                         "The position does not yet have units. "
                         "Units depend upon `pylada.crystal.Structure.scale`.\n"
                         "Finally, the position is owned internally by the "
                         "atom. It cannot be set to reference an object\n"
                         "(say a list or numpy array). ``atom.pos = "
                         "some_list`` will copy the values of"
                         "``some_list``.\n"),
      PYLADA_DECLARE(type, "Occupation of this atomic site.\n\n"
                         "Can be any object whatsoever."),
      {NULL}  /* Sentinel */
  };
# undef PYLADA_DECLARE
# define PYLADA_DECLARE(name, object, doc) \
    { const_cast<char*>(#name), T_OBJECT_EX, \
      LADA_OFFSETOF(Pylada::crystal::PyAtomObject, object), 0, const_cast<char*>(doc) }
  static PyMemberDef members[] = {
    PYLADA_DECLARE(__dict__, pydict, "Python attribute dictionary."),
#   ifdef PYLADA_DEBUG
      PYLADA_DECLARE(_weakreflist, weakreflist, "List of weak references."),
#   endif
    {NULL, 0, 0, 0, NULL}  /* Sentinel */
  };
# undef PYLADA_DECLARE
# define PYLADA_DECLARE(name, func, args, doc) \
    {#name, (PyCFunction)func, METH_ ## args, doc} 
  static PyMethodDef methods[] = {
      PYLADA_DECLARE(copy, pylada_atom_copy, NOARGS, "Returns a deepcopy of the atom."),
      PYLADA_DECLARE( to_dict, pylada_atom_to_dict, NOARGS, 
                    "Returns a dictionary with shallow copies of items." ),
      PYLADA_DECLARE(__copy__, pylada_atom_shallowcopy, NOARGS, "Shallow copy of an atom."),
      PYLADA_DECLARE(__deepcopy__, *(PyObject*(*)(PyAtomObject*, PyObject*)) copy_atom, O, "Deep copy of an atom."),
      PYLADA_DECLARE(__getstate__, pylada_atom_getstate, NOARGS, "Implements pickle protocol."),
      PYLADA_DECLARE(__setstate__, pylada_atom_setstate, O, "Implements pickle protocol."),
      PYLADA_DECLARE(__reduce__,   pylada_atom_reduce, NOARGS, "Implements pickle protocol."),
      {NULL}  /* Sentinel */
  };
# undef PYLADA_DECLARE
 
  static PyTypeObject dummy = {
      PyObject_HEAD_INIT(NULL)
      0,                                 /*ob_size*/
      "pylada.crystal.cppwrappers.Atom",   /*tp_name*/
      sizeof(PyAtomObject),   /*tp_basicsize*/
      0,                                 /*tp_itemsize*/
      (destructor)pylada_atom_dealloc,     /*tp_dealloc*/
      0,                                 /*tp_print*/
      0,                                 /*tp_getattr*/
      0,                                 /*tp_setattr*/
      0,                                 /*tp_compare*/
      (reprfunc)pylada_atom_repr,          /*tp_repr*/
      0,                                 /*tp_as_number*/
      0,                                 /*tp_as_sequence*/
      0,                                 /*tp_as_mapping*/
      0,                                 /*tp_hash */
      0,                                 /*tp_call*/
      0,                                 /*tp_str*/
      0,                                 /*tp_getattro*/
      0,                                 /*tp_setattro*/
      0,                                 /*tp_as_buffer*/
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
      "Defines an atomic site.\n\n"
        "__init__ accepts different kind of input.\n"
        "  - The position can be given as:\n" 
        "      - the first *three* positional argument\n"
        "      - as a keyword argument ``position``, \n"
        "  - The type can be given as:\n"
        "      - arguments listed after the first three giving the position.\n"
        "        A list is created to hold references to these arguments.\n"
        "      - as a keyword argument ``type``.\n"
        "  - All other keyword arguments become attributes. "
             "In other words, one could add ``magnetic=0.5`` if one wanted to "
             "specify the magnetic moment of an atom.\n\n"
        "For instance, the following will create a silicon atom at the origin::\n\n"
        "  atom = Atom(0, 0, 0, 'Si')\n\n"
        "Or we could place a iron atom with a magntic moment::\n\n"
        "  atom = Atom(0.25, 0, 0.5, 'Si', moment=0.5)\n\n"
        "The ``moment`` keyword will create a corresponding ``atom.moment`` keyword "
        "with a value of 0.5. There are strictly no limits on what kind of type to "
        "include as attributes. However, in order to work well with the rest of Pylada, "
        "it is best if extra attributes are pickle-able.\n\n"
        ".. note:: the position is always owned by the object. "
        "Two atoms will not own the same position object. "
        "The position given on input is *copied*, *not* referenced. "
        "All other attributes behave like other python attributes: "
        "they are refence if complex objects and copies if a basic python type.",
      (traverseproc)pylada_atom_traverse,  /* tp_traverse */
      (inquiry)pylada_atom_gcclear,        /* tp_clear */
      0,		                     /* tp_richcompare */
      LADA_OFFSETOF(PyAtomObject, weakreflist),   /* tp_weaklistoffset */
      0,		                     /* tp_iter */
      0,		                     /* tp_iternext */
      methods,                           /* tp_methods */
      members,                           /* tp_members */
      getsetters,                        /* tp_getset */
      0,                                 /* tp_base */
      0,                                 /* tp_dict */
      0,                                 /* tp_descr_get */
      0,                                 /* tp_descr_set */
      LADA_OFFSETOF(PyAtomObject, pydict),        /* tp_dictoffset */
      (initproc)pylada_atom_init,          /* tp_init */
      0,                                 /* tp_alloc */
      (newfunc)PyAtom_NewWithArgs,                /* tp_new */
  };
  if(dummy.tp_getattro == 0) dummy.tp_getattro = PyObject_GenericGetAttr;
  if(dummy.tp_setattro == 0) dummy.tp_setattro = PyObject_GenericSetAttr;
  return &dummy;
}
