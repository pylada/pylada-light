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

#if PYLADA_CRYSTAL_MODULE == 100 || PYLADA_CRYSTAL_MODULE == 0
  //! \brief Describes basic atom type. 
  //! \details This is a python object. 
  struct PyAtomObject
  {
    PyObject_HEAD
    //! Holds list of weak pointers.
    PyObject *weakreflist;
    //! Holds python attribute dictionary.
    PyObject *pydict;
    //! Holds the occupation object.
    PyObject *type;
    //! The atomic position in cartesian coordinate.
    math::rVector3d pos;
  };

  namespace 
  {
#endif 

      
#if PYLADA_CRYSTAL_MODULE != 1
  //! Returns pointer to atom type.
  PYLADA_INLINE PyTypeObject* atom_type()
    PYLADA_END(return (PyTypeObject*)api_capsule[PYLADA_SLOT(crystal)];)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)atom_type();
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)
        
#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a new atom.
  PYLADA_INLINE PyAtomObject* new_atom()
    PYLADA_END( return ((PyAtomObject*(*)()) api_capsule[PYLADA_SLOT(crystal)])();)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)((PyAtomObject*(*)())new_atom);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a new atom with a given type, also calling initialization.
  PYLADA_INLINE PyAtomObject* new_atom(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
    PYLADA_END( return (*(PyAtomObject*(*)(PyTypeObject*, PyObject*, PyObject*))
                      api_capsule[PYLADA_SLOT(crystal)])(_type, _args, _kwargs); )
#else
  api_capsule[PYLADA_SLOT(crystal)]
         = (void *)((PyAtomObject*(*)(PyTypeObject*, PyObject*, PyObject*))new_atom);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a deepcopy of atom.
  PYLADA_INLINE PyAtomObject *copy_atom(PyAtomObject* _self, PyObject *_memo=NULL)
    PYLADA_END( return (*(PyAtomObject*(*)(PyAtomObject*, PyObject*))
                      api_capsule[PYLADA_SLOT(crystal)])(_self, _memo); )
#else
  api_capsule[3] = (void *)copy_atom;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Checks type of an object.
  inline bool check_atom(PyObject *_self)
    { return PyObject_TypeCheck(_self, atom_type()); }
  //! Checks type of an object.
  inline bool checkexact_atom(PyObject *_self)
    { return _self->ob_type ==  atom_type(); }
  //! Checks type of an object.
  inline bool check_atom(PyAtomObject *_self)
    { return PyObject_TypeCheck(_self, atom_type()); }
  //! Checks type of an object.
  inline bool checkexact_atom(PyAtomObject *_self)
    { return _self->ob_type ==  atom_type(); }

  } // anonymous namespace

  //! \brief Holds a reference to the underlying python atom
  //! \details This wrapper makes it easy to interact with the python object.
  //!          It should be used throughout any c++ code. Mainly, it makes
  //!          sure Py_INCREF and Py_DECREF are called accordingly.
  class Atom : public python::Object
  {
    public: 
      //! Constructor
      Atom() : Object() { object_ = (PyObject*)new_atom(); }
      //! Acquires ownership of an atom.
      Atom(Atom const &_c ) : Object(_c) {}
      //! Shallow copy Constructor
      Atom(PyAtomObject *_data ) : Object((PyObject*)_data) {}
      //! Full Initialization.
      Atom(PyObject *_args, PyObject *_kwargs) : Object()
        { object_ = (PyObject*)new_atom(atom_type(), _args, _kwargs); }

      //! Swaps data of two atoms.
      void swap(Atom &_in) { return std::swap(object_, _in.object_); }
      //! \brief Acquires another atom. 
      //! \throws error::TypeError, both c++ and python, when _in is not an
      //!         Atom or subtype.
      void reset(PyObject *_in) 
      {
        if(_in != NULL and not check_atom(_in))
        {
          PYLADA_PYTHROW(TypeError, "Cannot acquire object which is not an Atom or subclass.");
        }
        PyObject *dummy = (PyObject*)object_;
        object_ = (PyObject*)_in;
        Py_XINCREF(object_);
        Py_XDECREF(dummy);
      }
      //! True if the atom is valid.
      bool is_valid() const { return object_ != NULL; }
      //! \brief Return a (deep) copy of this atom.
      //! \details Return does not share data with this atom. 
      //!          Use constructor to obtain that behavior.
      //!          The user should check that the atom is valid.
      Atom copy() const { return Atom(copy_atom((PyAtomObject*)object_)); } 

      //! Returns borrowed reference to dictionary.
      PyObject* dict() const { return ((PyAtomObject*)object_)->pydict; }

      //! Points to data.
      PyAtomObject const* operator->() const { return (PyAtomObject*)object_; }
      //! Points to data.
      PyAtomObject* operator->() { return (PyAtomObject*)object_; }

      //! Returns const reference to pos.
      math::rVector3d const & pos() const { return ((PyAtomObject*)object_)->pos; }
      //! Returns reference to pos.
      math::rVector3d & pos() { return ((PyAtomObject*)object_)->pos; }
      //! Returns const reference to pos.
      math::rVector3d::Scalar const & pos(size_t i) const { return ((PyAtomObject*)object_)->pos(i); }
      //! Returns reference to pos.
      math::rVector3d::Scalar & pos(size_t i) { return ((PyAtomObject*)object_)->pos(i); }

      //! Returns type as a python object.
      python::Object type() const { return python::Object::acquire(((PyAtomObject*)object_)->type); }
      //! Returns type as a python object.
      PyObject* pytype() const { return ((PyAtomObject*)object_)->type; }
      //! Sets type to python object.
      void type(python::Object const &_in)
      { 
        PyObject *dummy = ((PyAtomObject*)object_)->type;
        ((PyAtomObject*)object_)->type = _in.new_ref(); 
        Py_XDECREF(dummy);
      }


      //! Check if instance is an atom.
      static bool check(PyObject *_atom) { return check_atom(_atom); }
      //! Check if instance is an atom.
      static bool check_exact(PyObject *_atom) { return checkexact_atom(_atom); }
      //! \brief Acquires new reference to an object.
      //! \details incref's reference first, unless null.
      //!          If non-null checks that it is a subtype of Atom.
      //! \returns a valid or invalid Atom object.
      static Atom acquire(PyObject *_atom) 
      {
        if(_atom == NULL) return Atom((PyAtomObject*)_atom);
        if(not Atom::check(_atom))
        {
          PYLADA_PYERROR_FORMAT( TypeError,
                               "Expected an Atom or subtype, not %.200s",
                               _atom->ob_type->tp_name );
          return Atom();
        }
        Py_INCREF(_atom);
        return Atom((PyAtomObject*)_atom);
      }
      //! \brief Acquires new reference to an object.
      //! \details incref's reference first, unless null.
      //!          Does not check object is Atom or subtype, and does not
      //!          throw.
      static Atom acquire_(PyObject *_atom) 
        { Py_XINCREF(_atom); return Atom((PyAtomObject*)_atom); }
  };
#endif
