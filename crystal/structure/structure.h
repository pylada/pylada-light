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

#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Describes basic structure type. 
  //! \details Instances of this object are exactly those that are seen
  //!          within the python interface. C++, however, defines a
  //!          secondary Structure object which wrapps around a python
  //!          refence to instances of this object. Structure provides some
  //!          syntactic sugar for handling in c++. 
  struct PyStructureObject
  {
    PyObject_HEAD 
    //! Holds list of weak pointers.
    PyObject *weakreflist;
    //! Holds python attribute dictionary.
    PyObject *pydict;
    //! Holds python attribute dictionary.
    PyObject *scale;
    //! The unit-cell of the structure in cartesian coordinate.
    math::rMatrix3d cell;
    //! Vector of atom wrappers.
    std::vector<Atom> atoms;
  };

  namespace
  {
#endif 

#if PYLADA_CRYSTAL_MODULE != 1
  // Returns pointer to structure type.
  PYLADA_INLINE PyTypeObject* structure_type()
    PYLADA_END(return (PyTypeObject*)api_capsule[PYLADA_SLOT(crystal)];)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)structure_type();
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Returns address of structure iterator type object.
  PYLADA_INLINE PyTypeObject* structureiterator_type()
   PYLADA_END(return (PyTypeObject*)api_capsule[PYLADA_SLOT(crystal)];)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)structureiterator_type();
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a new structure.
  PYLADA_INLINE PyStructureObject* new_structure()
    PYLADA_END(return (PyStructureObject*)api_capsule[PYLADA_SLOT(crystal)];)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)((PyStructureObject*(*)())new_structure);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a new structure with a given type, also calling initialization.
  PYLADA_INLINE PyStructureObject* new_structure( PyTypeObject* _type, 
                                                PyObject *_args, 
                                                PyObject *_kwargs )
    PYLADA_END(return (*(PyStructureObject*(*)( PyTypeObject*, 
                                              PyObject*, 
                                              PyObject* ))
                     api_capsule[PYLADA_SLOT(crystal)])
                    (_type, _args, _kwargs);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)((PyStructureObject*(*)(PyTypeObject*, PyObject*, PyObject*))new_structure);
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a deepcopy of structure.
  PYLADA_INLINE PyStructureObject *copy_structure( PyStructureObject* _self,
                                                 PyObject *_memo=NULL)
    PYLADA_END(return (*(PyStructureObject*(*)(PyStructureObject*, PyObject*))
                    api_capsule[PYLADA_SLOT(crystal)])(_self, _memo);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)copy_structure;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Transforms a structure in-place, according to symop.
  PYLADA_INLINE void itransform_structure( PyStructureObject* _self,
                                         Eigen::Matrix<types::t_real, 4, 3> const &_op )
    PYLADA_END(return (*(void(*)( PyStructureObject*, 
                                Eigen::Matrix<types::t_real, 4, 3> const & ))
                     api_capsule[PYLADA_SLOT(crystal)])(_self, _op);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)itransform_structure;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Checks type of an object.
  inline bool check_structure(PyObject *_self)
    { return PyObject_TypeCheck(_self, structure_type()); }
  //! Checks type of an object.
  inline bool checkexact_structure(PyObject *_self)
    { return _self->ob_type ==  structure_type(); }
  //! Checks type of an object.
  inline bool check_structure(PyStructureObject *_self)
    { return PyObject_TypeCheck(_self, structure_type()); }
  //! Checks type of an object.
  inline bool checkexact_structure(PyStructureObject *_self)
    { return _self->ob_type ==  structure_type(); }

  } // anonymous namespace
  
  //! Wraps a python structure. 
  class Structure : public python::Object
  {
    public:
      //! \typedef The type of the collection of atoms. 
      typedef Atom t_Atom;
      //! \typedef The type of the collection of atoms. 
      typedef std::vector<t_Atom> t_Atoms;
      //! \typedef Type of the iterator.
      typedef t_Atoms::iterator iterator;
      //! \typedef Type of the constant iterator.
      typedef t_Atoms::const_iterator const_iterator;
      //! \typedef Type of the reverse iterator.
      typedef t_Atoms::reverse_iterator reverse_iterator;
      //! \typedef Type of the reverse constant iterator.
      typedef t_Atoms::const_reverse_iterator const_reverse_iterator;
      //! \typedef Type of the atoms.
      typedef t_Atoms::value_type value_type;
      //! \typedef Type of the reference to the atoms.
      typedef t_Atoms::reference reference;
      //! \typedef Type of the constant reference to the atoms.
      typedef t_Atoms::const_reference const_reference;
      //! \typedef Type of the size.
      typedef t_Atoms::size_type size_type;
      //! \typedef Type of the difference.
      typedef t_Atoms::difference_type difference_type;
      //! \typedef Type of the pointer to the atoms.
      typedef t_Atoms::pointer pointer;
      //! \typedef Type of the allocator used for the atoms.
      typedef t_Atoms::allocator_type allocator_type;
  
      //! Constructor
      Structure() : Object() { object_ = (PyObject*) new_structure(); }
      //! Cloning Constructor
      Structure(const Structure &_c) : Object(_c) {}
      //! Shallow copy Constructor
      Structure(PyStructureObject *_data ) : Object((PyObject*)_data) {}
      //! Full Initialization.
      Structure(PyObject *_args, PyObject *_kwargs) : Object()
        { object_ = (PyObject*)new_structure(structure_type(), _args, _kwargs); }
  
      //! Returns const reference to cell.
      math::rMatrix3d const & cell() const { return ((PyStructureObject*)object_)->cell; }
      //! Returns reference to cell.
      math::rMatrix3d & cell() { return ((PyStructureObject*)object_)->cell; }
      //! Returns const reference to cell.
      math::rMatrix3d::Scalar const & cell(size_t i, size_t j) const
        { return ((PyStructureObject*)object_)->cell(i, j); }
      //! Returns reference to cell.
      math::rMatrix3d::Scalar & cell(size_t i, size_t j)
        { return ((PyStructureObject*)object_)->cell(i, j); }
      //! Returns scale as real number in current units.
      types::t_real scale() const
        { return python::get_quantity(((PyStructureObject*)object_)->scale); }
      //! Returns scale as real number in given units.
      types::t_real scale(std::string const &_units) const
        { return python::get_quantity(((PyStructureObject*)object_)->scale, _units); }
  
      //! Deep copy of a structure.
      Structure copy() const { return Structure(copy_structure((PyStructureObject*)object_)); }
      //! Swaps content of two structures.
      void swap(Structure &_other) { std::swap(object_, _other.object_); }
      //! Points to data.
      PyStructureObject const* operator->() const { return (PyStructureObject*)object_; }
      //! Points to data.
      PyStructureObject* operator->() { return (PyStructureObject*)object_; }
  
      //! Iterator to the atoms.
      iterator begin() { return ((PyStructureObject*)object_)->atoms.begin(); }
      //! Iterator to the atoms.
      iterator end() { return ((PyStructureObject*)object_)->atoms.end(); }
      //! Iterator to the atoms.
      reverse_iterator rbegin() { return ((PyStructureObject*)object_)->atoms.rbegin(); }
      //! Iterator to the atoms.
      reverse_iterator rend() { return ((PyStructureObject*)object_)->atoms.rend(); }
      //! Iterator to the atoms.
      const_iterator begin() const { return ((PyStructureObject*)object_)->atoms.begin(); }
      //! Iterator to the atoms.
      const_iterator end() const { return ((PyStructureObject*)object_)->atoms.end(); }
      //! Iterator to the atoms.
      const_reverse_iterator rbegin() const { return ((PyStructureObject*)object_)->atoms.rbegin(); }
      //! Iterator to the atoms.
      const_reverse_iterator rend() const { return ((PyStructureObject*)object_)->atoms.rend(); }
      //! Number of atoms.
      size_type size() const { return ((PyStructureObject*)object_)->atoms.size(); }
      //! Maximum number of atoms.
      size_type max_size() const { return ((PyStructureObject*)object_)->atoms.max_size(); }
      //! Number of atoms.
      void resize(size_type _n) { ((PyStructureObject*)object_)->atoms.resize(_n); }
      //! Number of atoms.
      void resize(size_type _n, const_reference _obj) { ((PyStructureObject*)object_)->atoms.resize(_n, _obj); }
      //! Attempts to reserve memory for atoms.
      void reserve(size_type _n) { ((PyStructureObject*)object_)->atoms.reserve(_n); }
      //! Reserved memory.
      size_type capacity() const { return ((PyStructureObject*)object_)->atoms.capacity(); }
      //! Whether any atoms are in the structure.
      bool empty() const { return ((PyStructureObject*)object_)->atoms.empty(); }
      //! Returns nth atom.
      reference operator[](size_type _n) { return ((PyStructureObject*)object_)->atoms[_n]; }
      //! Returns nth atom.
      const_reference operator[](size_type _n) const { return ((PyStructureObject*)object_)->atoms[_n]; }
      //! Returns nth atom.
      reference at(size_type _n) { return ((PyStructureObject*)object_)->atoms.at(_n); }
      //! Returns nth atom.
      const_reference at(size_type _n) const { return ((PyStructureObject*)object_)->atoms.at(_n); }
      //! Returns nth atom.
      reference front() { return ((PyStructureObject*)object_)->atoms.front(); }
      //! Returns nth atom.
      const_reference front() const { return ((PyStructureObject*)object_)->atoms.front(); }
      //! Returns nth atom.
      reference back() { return ((PyStructureObject*)object_)->atoms.back(); }
      //! Returns nth atom.
      const_reference back() const { return ((PyStructureObject*)object_)->atoms.back(); }
      //! Replaces content of the container.
      template <class InputIterator>
        void assign(InputIterator _first, InputIterator _last)
        { ((PyStructureObject*)object_)->atoms.assign(_first, _last); }
      //! Replaces content of the container.
      void assign(size_type _n, const_reference _u) { ((PyStructureObject*)object_)->atoms.assign(_n, _u); }
      //! Adds atom at end of container.
      void push_back(const_reference _u) { ((PyStructureObject*)object_)->atoms.push_back(_u); }
      //! Deletes last element. 
      void pop_back() { ((PyStructureObject*)object_)->atoms.pop_back(); }
      //! Inserts atoms in container at given position.
      template <class InputIterator>
        void insert(iterator _pos, InputIterator _first, InputIterator _last)
        { ((PyStructureObject*)object_)->atoms.insert(_pos, _first, _last); }
      //! Inserts one atom at given position.
      iterator insert(iterator _pos, const_reference x) { return ((PyStructureObject*)object_)->atoms.insert(_pos, x); }
      //! Inserts n atom at given position.
      void insert(iterator _pos, size_type _n, const_reference x)
        { ((PyStructureObject*)object_)->atoms.insert(_pos, _n, x); }
      //! Erases atom at given positions.
      iterator erase(iterator _pos) { return ((PyStructureObject*)object_)->atoms.erase(_pos); }
      //! Erases range of atoms at given positions.
      iterator erase(iterator _first, iterator _last)
        { return ((PyStructureObject*)object_)->atoms.erase(_first, _last); }
      //! Clears all atoms from structure.
      void clear() { ((PyStructureObject*)object_)->atoms.clear(); }
      //! Returns allocator object used to construct atom container.
      allocator_type get_allocator() const { return ((PyStructureObject*)object_)->atoms.get_allocator(); }
  
      //! \brief True if both structures refer to the same object in memory.
      //! \details Does not compare values, just memory objects.
      bool is_same(Structure const &_in) { return object_ == _in.object_; }
  
      //! Returns structure volume in current units.
      types::t_real volume() const
        { return std::abs(((PyStructureObject*)object_)->cell.determinant()) 
                 * std::pow(python::get_quantity(((PyStructureObject*)object_)->scale), 3); }
  
      //! Transforms a structure according to an affine transformation.
      void transform(Eigen::Matrix<types::t_real, 4, 3> const &_affine)
        { itransform_structure((PyStructureObject*)object_, _affine); }
  
      //! Returns borrowed reference to dictionary.
      PyObject* dict() const { return ((PyStructureObject*)object_)->pydict; }
  
      //! Check if instance is an atom.
      static bool check(PyObject *_str) { return check_structure(_str); }
      //! Check if instance is an atom.
      static bool check_exact(PyObject *_str) { return checkexact_structure(_str); }
      //! \brief Acquires new reference to an object.
      //! \details incref's reference first, unless null.
      //!          If non-null checks that it is a subtype of Structure.
      //! \returns A valid or invalid Structure object.
      static Structure acquire(PyObject *_str) 
      {
        if(_str == NULL) return Structure((PyStructureObject*)_str);
        if(not Structure::check(_str))
        {
          PYLADA_PYERROR_FORMAT( TypeError,
                               "Expected an Structure or subtype, not %.200s",
                               _str->ob_type->tp_name );
          return Structure();
        }
        Py_INCREF(_str);
        return Structure((PyStructureObject*)_str);
      }
      //! \brief Acquires new reference to an object.
      //! \details incref's reference first, unless null.
      //!          Does not check object is Structure or subtype, and does not
      //!          throw.
      static Structure acquire_(PyObject *_str) 
        { Py_XINCREF(_str); return Structure((PyStructureObject*)_str); }
  };
#endif
