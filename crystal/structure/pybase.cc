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

// structure getters/settters.
#include "getset.cc"
// iterator functions and type.
#include "iterator.cc"
// structure member functions.
#include "members.cc"
// creation, deallocation, initialization.
#include "cdi.cc"
// sequence functions.
#include "sequence.cc"

//! Creates a new structure.
PyStructureObject* new_structure()
{
  PyStructureObject* result = (PyStructureObject*) structure_type()->tp_alloc(structure_type(), 0);
  if(not result) return NULL;
  result->weakreflist = NULL;
  result->scale = NULL;
  new(&result->cell) Pylada::math::rMatrix3d(Pylada::math::rMatrix3d::Identity());
  new(&result->atoms) std::vector<Atom>;
  result->pydict = PyDict_New();
  if(result->pydict == NULL) { Py_DECREF(result); return NULL; }
  return result;
}
//! Creates a new structure with a given type.
PyStructureObject* PyStructure_NewWithArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
{
  PyStructureObject* result = (PyStructureObject*)_type->tp_alloc(_type, 0);
  if(not result) return NULL;
  result->weakreflist = NULL;
  result->scale = NULL;
  new(&result->cell) Pylada::math::rMatrix3d(Pylada::math::rMatrix3d::Identity());
  new(&result->atoms) std::vector<Atom>;
  result->pydict = PyDict_New();
  if(result->pydict == NULL) { Py_DECREF(result); return NULL; }
  return result;
}

// Creates a new structure with a given type, also calling initialization.
PyStructureObject* new_structure(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
{
  PyStructureObject* result = PyStructure_NewWithArgs(_type, _args, _kwargs);
  if(result == NULL) return NULL;
  if(_type->tp_init((PyObject*)result, _args, _kwargs) < 0) {Py_DECREF(result); return NULL; }
  return result;
}

// Creates a deepcopy of structure.
PyStructureObject *copy_structure(PyStructureObject* _self, PyObject *_memo)
{
  PyStructureObject* result = (PyStructureObject*)_self->ob_type->tp_alloc(_self->ob_type, 0);
  if(not result) return NULL;
  result->weakreflist = NULL;
  new(&result->cell) Pylada::math::rMatrix3d(_self->cell);
  new(&result->atoms) std::vector<Atom>;
  result->pydict = NULL;
  Py_INCREF(_self->scale);
  result->scale = _self->scale;
  PyObject* copymod = PyImport_ImportModule("copy");
  if(copymod == NULL) return NULL;
  PyObject *deepcopystr = PyString_FromString("deepcopy");
  if(not deepcopystr) { Py_DECREF(copymod); return NULL; }
  else if(_self->pydict != NULL)
  {
    result->pydict = PyObject_CallMethodObjArgs(copymod, deepcopystr, _self->pydict, _memo, NULL);
    if(result->pydict == NULL) { Py_DECREF(result); }
  }
  std::vector<Atom>::const_iterator i_first = _self->atoms.begin();
  std::vector<Atom>::const_iterator const i_end = _self->atoms.end();
  for(; i_first != i_end; ++i_first)
  {
    Atom atom((PyAtomObject*)PyObject_CallMethodObjArgs(copymod, deepcopystr, i_first->borrowed(), _memo, NULL));
    if(not atom) return NULL;
    result->atoms.push_back(atom);
  }

  Py_DECREF(copymod);
  Py_DECREF(deepcopystr);
  return result;
}
// Returns pointer to structure type.
PyTypeObject* structure_type()
{
  static PyMappingMethods structure_as_mapping = {
      (lenfunc)structure_size,
      (binaryfunc)structure_subscript,
      (objobjargproc)structure_ass_subscript
  };
  static PySequenceMethods structure_as_sequence = {
      (lenfunc)structure_size,                    /* sq_length */
      (binaryfunc)NULL,                           /* sq_concat */
      (ssizeargfunc)NULL,                         /* sq_repeat */
      (ssizeargfunc)structure_getitem,            /* sq_item */
      (ssizessizeargfunc)NULL,                    /* sq_slice */
      (ssizeobjargproc)structure_setitem,         /* sq_ass_item */
      (ssizessizeobjargproc)NULL,                 /* sq_ass_slice */
      (objobjproc)NULL,                           /* sq_contains */
      (binaryfunc)NULL,                           /* sq_inplace_concat */
      (ssizeargfunc)NULL,                         /* sq_inplace_repeat */
  };
#  ifdef PYLADA_DECLARE
#    error PYLADA_DECLARE already defined.
#  endif
#  define PYLADA_DECLARE(name, doc) \
     { const_cast<char*>(#name), (getter) structure_get ## name, \
       (setter) structure_set ## name, const_cast<char*>(doc) }
   
   static PyGetSetDef getsetters[] = {
       PYLADA_DECLARE(cell, "Cell matrix in cartesian coordinates.\n\n"
                          "Unlike most ab-initio codes, cell-vectors are "
                          "given in column vector format. " 
                          "The cell does not yet have units. "
                          "Units depend upon :class:`Structure.scale`. "
                          "Across pylada, it is expected that a cell time "
                          "this scale are angstroms. Finally, the cell "
                          "is owned internally by the structure. It cannot be set to "
                          "reference an object (say a list or numpy array). "
                          "``structure.cell = some_list`` will copy the values of ``some_list``. "),
       PYLADA_DECLARE(scale, "Scale factor of this structure.\n"
                           "Should be a number or unit given by "
                           "the python package `quantities "
                           "<http://packages.python.org/quantities/index.html>`_.\n"
                           "If given as a number, then the current units are kept.\n"
                           "Otherwise, it changes the units."),
       { const_cast<char*>("volume"), (getter) structure_getvolume, NULL, 
         const_cast<char*>("Volume of the structure.\n\nIncludes scale.") },
       {NULL}  /* Sentinel */
   };
#  undef PYLADA_DECLARE
#  define PYLADA_DECLARE(name, object, doc) \
     { const_cast<char*>(#name), T_OBJECT_EX, \
       LADA_OFFSETOF(PyStructureObject, object), 0, const_cast<char*>(doc) }
   static PyMemberDef members[] = {
     PYLADA_DECLARE(__dict__, pydict, "Python attribute dictionary."),
#    ifdef PYLADA_DEBUG
       PYLADA_DECLARE(_weakreflist, weakreflist, "List of weak references."),
#    endif
     {NULL, 0, 0, 0, NULL}  /* Sentinel */
   };
#  undef PYLADA_DECLARE
#  define PYLADA_DECLARE(name, func, args, doc) \
     {#name, (PyCFunction)func, METH_ ## args, doc} 
   static PyMethodDef methods[] = {
       PYLADA_DECLARE(copy, structure_copy, NOARGS, "Returns a deepcopy of the structure."),
       PYLADA_DECLARE( to_dict, structure_to_dict, NOARGS, 
                     "Returns a dictionary with shallow copies of items." ),
       PYLADA_DECLARE(__copy__, structure_shallowcopy, NOARGS, "Shallow copy of an structure."),
       PYLADA_DECLARE(__deepcopy__, copy_structure, O, "Deep copy of an structure."),
       PYLADA_DECLARE(__getstate__, structure_getstate, NOARGS, "Implements pickle protocol."),
       PYLADA_DECLARE(__setstate__, structure_setstate, O, "Implements pickle protocol."),
       PYLADA_DECLARE(__reduce__,   structure_reduce, NOARGS, "Implements pickle protocol."),
       PYLADA_DECLARE( add_atom,   structure_add_atom, KEYWORDS,
                     "Adds atom to structure.\n\n"
                     "The argument to this function is either another atom, "
                     "in which case a reference to that atom is appended to "
                     "the structure. Or, it is any arguments used to "
                     "initialize atoms in :class:`Atom`. "
                     "Finally, this function can be chained as follows::\n\n"
                     "  structure.add_atom(0,0,0, 'Au')                        \\\n"
                     "           .add_atom(0.25, 0.25, 0.25, ['Pd', 'Si'], m=5)\\\n"
                     "           .add_atom(atom_from_another_structure)\n\n"
                     "In the example above, both ``structure`` and the *other* structure will "
                     "reference the same atom (``atom_from_another_structure``). "
                     "Changing, say, that atom's type in one structure will also "
                     "change it in the other.\n\n"
                     ":returns: The structure itself, so that add_atom methods can be chained."),
       PYLADA_DECLARE( insert, structure_insert, VARARGS, 
                     "Inserts atom at given position.\n\n"
                     ":param index:\n    Position at which to insert the atom.\n"
                     ":param atom: \n    :class:`Atom` or subtype to insert." ),
       PYLADA_DECLARE(pop, structure_pop, O, "Removes and returns atom at given position."),
       PYLADA_DECLARE(clear, structure_clear, NOARGS, "Removes all atoms from structure."),
       PYLADA_DECLARE( extend, structure_extend, O, 
                     "Appends list of atoms to structure.\n\n"
                     "The argument is any iterable objects containing only atoms, "
                     "e.g. another Structure." ),
       PYLADA_DECLARE(append, structure_append, O, "Appends an Atom or subtype to the structure.\n"),
       PYLADA_DECLARE( transform, structure_transform, O, 
                     "Transform a structure in-place.\n\n"
                     "Applies an affine transformation to a structure. "
                     "An affine transformation is a 4x3 matrix, where the upper 3 rows "
                     "correspond to a rotation (applied first), and the last row to "
                     "a translation (applied second).\n\n"
                     ":param matrix:\n"
                     "   Affine transformation defined as a 4x3 numpy array.\n\n"
                     ":returns: A new :class:`Structure` (or derived) instance." ),
       PYLADA_DECLARE( __getitem__, structure_subscript, O|METH_COEXIST,
                     "Retrieves atom or slice.\n\n"
                     ":param index:\n"
                     "    If an integer, returns a refence to that atom. "
                         "If a slice, returns a list with all atoms in that slice." ),
       PYLADA_DECLARE( __setitem__, structure_setitemnormal, VARARGS|METH_COEXIST,
                     "Sets atom or atoms.\n\n"
                     ":param index:\n"
                     "    If an integers, sets that atom to the input value. "
                     "    If a slice, then sets all atoms in refered to in the structure "
                          "to the corresponding atom in the value.\n"
                     ":param value:\n"
                     "    If index is an integer, this should be an :class:`atom <Atom>`. "
                         "If index is a slice, then this should be a sequence of "
                         ":class:`atoms <Atom>` of the exact length of the "
                         "slice."),
       {NULL}  /* Sentinel */
   };
#  undef PYLADA_DECLARE
  
   static PyTypeObject dummy = {
       PyObject_HEAD_INIT(NULL)
       0,                                 /*ob_size*/
       "pylada.crystal.cppwrappers.Structure",   /*tp_name*/
       sizeof(PyStructureObject),             /*tp_basicsize*/
       0,                                 /*tp_itemsize*/
       (destructor)structure_dealloc,     /*tp_dealloc*/
       0,                                 /*tp_print*/
       0,                                 /*tp_getattr*/
       0,                                 /*tp_setattr*/
       0,                                 /*tp_compare*/
       (reprfunc)structure_repr,          /*tp_repr*/
       0,                                 /*tp_as_number*/
       &structure_as_sequence,            /*tp_as_sequence*/
       &structure_as_mapping,             /*tp_as_mapping*/
       0,                                 /*tp_hash */
       0,                                 /*tp_call*/
       0,                                 /*tp_str*/
       0,                                 /*tp_getattro*/
       0,                                 /*tp_setattro*/
       0,                                 /*tp_as_buffer*/
       Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_ITER, /*tp_flags*/
       "Defines a structure.\n\n"         /*tp_doc*/
         "A structure is a special kind of sequence containing only "
         ":class:`Atom`. It also sports attributes such a "
         "cell and scale.\n\n"
         "__init__ accepts different kind of input.\n"
         "  - if 9 numbers are given as arguments, these create the cell vectors, "
              "where the first three numbers define the first row of the matrix"
              "(not the first cell vector, eg column).\n"
         "  - if only one object is given it should be the cell matrix.\n"
         "  - the cell can also be given as a keyword argument.\n"
         "  - the scale can only be given as a keyword argument.\n"
         "  - All other keyword arguments become attributes. "
              "In other words, one could add ``magnetic=0.5`` if one wanted to "
              "specify the magnetic moment of a structure. It would later be "
              "accessible as an attribute, eg as ``structure.magnetic``.\n\n"
         ".. note:: The cell is always owned by the object. "
         "Two structures will not own the same cell object. "
         "The cell given on input is *copied*, *not* referenced. "
         "All other attributes behave like other python attributes: "
         "they are refence if complex objects and copies if a basic python type.",
       (traverseproc)structure_traverse,  /* tp_traverse */
       (inquiry)structure_gcclear,        /* tp_clear */
       0,		                     /* tp_richcompare */
       LADA_OFFSETOF(PyStructureObject, weakreflist),   /* tp_weaklistoffset */
       (getiterfunc)structureiterator_create,  /* tp_iter */
       0,		                     /* tp_iternext */
       methods,                           /* tp_methods */
       members,                           /* tp_members */
       getsetters,                        /* tp_getset */
       0,                                 /* tp_base */
       0,                                 /* tp_dict */
       0,                                 /* tp_descr_get */
       0,                                 /* tp_descr_set */
       LADA_OFFSETOF(PyStructureObject, pydict),   /* tp_dictoffset */
       (initproc)structure_init,          /* tp_init */
       0,                                 /* tp_alloc */
       (newfunc)PyStructure_NewWithArgs,  /* tp_new */
   };
   if(dummy.tp_getattro == 0) dummy.tp_getattro = PyObject_GenericGetAttr;
   if(dummy.tp_setattro == 0) dummy.tp_setattro = PyObject_GenericSetAttr;
   return &dummy;
}

// Returns pointer to structure iterator type.
PyTypeObject* structureiterator_type()
{ 
  static PyTypeObject type = {
      PyObject_HEAD_INIT(NULL)
      0,                                          /*ob_size*/
      "pylada.crystal.cppwrappers.StructureIter",   /*tp_name*/
      sizeof(StructureIterator),                  /*tp_basicsize*/
      0,                                          /*tp_itemsize*/
      (destructor)structureiterator_dealloc,      /*tp_dealloc*/
      0,                                          /*tp_print*/
      0,                                          /*tp_getattr*/
      0,                                          /*tp_setattr*/
      0,                                          /*tp_compare*/
      0,                                          /*tp_repr*/
      0,                                          
      0,                                          /*tp_as_sequence*/
      0,                                          /*tp_as_mapping*/
      0,                                          /*tp_hash */
      0,                                          /*tp_call*/
      0,                                          /*tp_str*/
      0,                                          /*tp_getattro*/
      0,                                          /*tp_setattro*/
      0,                                          /*tp_as_buffer*/
      Py_TPFLAGS_HAVE_ITER,
      "Iterator over atoms in a structure.",
      0,                                          /* tp_traverse */
      0,                                          /* tp_clear */
      0,                                          /* tp_richcompare */
      0,		                              /* tp_weaklistoffset */
      (getiterfunc)structureiterator_iter,        /* tp_iter */
      (iternextfunc)structureiterator_next,       /* tp_iternext */
  };
  return &type;
}
