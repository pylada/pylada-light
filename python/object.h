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

#if PYLADA_PYTHON_MODULE != 1
  //! \brief Thin wrapper around a python refence.
  //! \details In general, steals a reference which it decref's on destruction, unless it
  //!          is null. The destructor is not virtual, hence it is safer to
  //!          keep members of all derived class non-virtual as well.
  //!          When creating this object, the second argument can be false,
  //!          in which case the reference is not owned and never increfed or
  //!          decrefed (eg, borrowed). 
  class Object 
  {
    public:
      //! Steals a python reference.
      Object(PyObject* _in = NULL) : object_(_in) {}
      //! Copies a python reference.
      Object(Object const &_in) { Py_XINCREF(_in.object_); object_ = _in.object_; }
      //! Decrefs a python reference on destruction.
      ~Object() { PyObject* dummy = object_; object_ = NULL; Py_XDECREF(dummy);}
      //! Assignment operator. Gotta let go of current object.
      void operator=(Object const &_in) { Object::reset(_in.borrowed()); }
      //! Assignment operator. Gotta let go of current object.
      void operator=(PyObject *_in) { Object::reset(_in); }
      //! Casts to bool to check validity of reference.
      operator bool() const { return object_ != NULL; }
      //! True if reference is valid.
      bool is_valid () const { return object_ != NULL; }
      //! \brief Resets the wrapped refence.
      //! \details Decrefs the current object if needed. Incref's the input object.
      void reset(PyObject *_in = NULL) { object_reset(object_, _in); }
      //! \brief Resets the wrapped refence.
      //! \details Decrefs the current object if needed.
      void reset(Object const &_in) { reset(_in.object_); }
      
      //! \brief Releases an the reference.
      //! \details After this call, the reference is not owned by this object
      //!          anymore. The reference should be stolen by the caller.
      PyObject* release() { PyObject *result(object_); object_ = NULL; return result; }
      //! Returns a new reference to object.
      PyObject* new_ref() const
      { 
        if(object_ == NULL) return NULL; 
        Py_INCREF(object_);
        return object_; 
      }
      //! Returns a borrowed reference to object.
      PyObject* borrowed() const { return object_; }

      //! \brief Acquires a new reference.
      //! \details First incref's the reference (unless null).
      static Object acquire(PyObject *_in) { Py_XINCREF(_in); return Object(_in); }

      bool hasattr(std::string const &_name) const
        { return PyObject_HasAttrString(object_, _name.c_str()); }

      //! \brief Returns a new reference to a python attribute. 
      //! \details A python exception is set if attribute does not exist, and the
      //!          function returns null.
      inline Object pyattr(std::string const &_name) const
        { return Object(PyObject_GetAttrString((PyObject*)object_, _name.c_str())); }
      //! \brief Returns a new reference to a python attribute. 
      //! \details A python exception is set if attribute does not exist, and the
      //!          function returns null.
      inline Object pyattr(PyObject* _name) const
        { return Object(PyObject_GetAttr((PyObject*)object_, _name)); }
      //! \brief Sets/Deletes attribute.
      inline bool pyattr(std::string const& _name, PyObject *_in)
        { return PyObject_SetAttrString(object_, _name.c_str(), _in) == 0; }
      //! \brief Sets/Deletes attribute.
      inline bool pyattr(std::string const& _name, python::Object const &_in)
        { return PyObject_SetAttrString(object_, _name.c_str(), _in.borrowed()) == 0; }
      //! \brief Sets/Deletes attribute.
      inline bool pyattr(PyObject* _name, PyObject *_in)
        { return PyObject_SetAttr(object_, _name, _in) == 0; }

      //! \brief Compares two objects for equality.
      //! \details Looks for __eq__ in a. If not found, throws c++
      //! exception.
      bool operator==(PyObject *_b) const { return operator==(acquire(_b)); }
      //! \brief Compares two objects for equality.
      //! \details Looks for __eq__ in a. If not found, throws c++
      //! exception.
      bool operator==(Object const &_b) const
        { return object_equality_op(*this, _b); }
      //! \brief Compares two objects for equality.
      //! \details Looks for __eq__ in a. If not found, throws c++
      //! exception.
      bool operator!=(Object const &_b) const { return not operator==(_b); }
      //! \brief Compares two objects for equality.
      //! \details Looks for __eq__ in a. If not found, throws c++
      //! exception.
      bool operator!=(PyObject *_b) const { return operator!=(acquire(_b)); }
    protected:
      //! Python reference.
      PyObject* object_;
  };
#endif
