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
  //! \brief Holds data of a hf transform
  //! \details Instances of this object are exactly those that are seen
  //!          within the python interface. C++, however, defines a
  //!          secondary HFTransfirm object which wrapps around a python
  //!          refence to instances of this object. HFTransfrom provides some
  //!          syntactic sugar for handling in c++. 
  struct PyHFTObject
  {
    PyObject_HEAD 
    //! The transform to go the hf normal form.
    math::rMatrix3d transform;
    //! Vector of atom wrappers.
    math::iVector3d quotient;
  };

  namespace 
  {
#endif

#if PYLADA_CRYSTAL_MODULE != 1
  // Returns pointer to hftransform type.
  PYLADA_INLINE PyTypeObject* hftransform_type()
    PYLADA_END(return (PyTypeObject*)api_capsule[PYLADA_SLOT(crystal)];)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)hftransform_type();
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a new hftransform with a given type, also calling initialization.
  PYLADA_INLINE PyHFTObject* new_hftransform( PyTypeObject* _type, 
                                            PyObject *_args, 
                                            PyObject *_kwargs )
    PYLADA_END(return (*(PyHFTObject*(*)( PyTypeObject*, 
                                        PyObject*, 
                                        PyObject* ))
                     api_capsule[PYLADA_SLOT(crystal)])
                    (_type, _args, _kwargs);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)new_hftransform;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! Creates a deepcopy of hftransform.
  PYLADA_INLINE PyHFTObject *copy_hftransform(PyHFTObject* _self, PyObject *_memo = NULL)
    PYLADA_END(return (*(PyHFTObject*(*)(PyHFTObject*, PyObject*))
                     api_capsule[PYLADA_SLOT(crystal)])(_self, _memo);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)copy_hftransform;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1
  //! \brief Initializes a new hftransform from input lattice unit-cell and supercell.
  //! \details Performs initialization from c++ arguments.
  PYLADA_INLINE bool _init_hft( PyHFTObject* _self, 
                              math::rMatrix3d const &_lattice,
                              math::rMatrix3d const &_supercell )
    PYLADA_END(return (*(bool(*)( PyHFTObject*, math::rMatrix3d const&, 
                                math::rMatrix3d const &))
                     api_capsule[PYLADA_SLOT(crystal)])(_self, _lattice, _supercell);)
#else
  api_capsule[PYLADA_SLOT(crystal)] = (void *)_init_hft;
#endif
#define BOOST_PP_VALUE BOOST_PP_INC(PYLADA_SLOT(crystal))
#include PYLADA_ASSIGN_SLOT(crystal)

#if PYLADA_CRYSTAL_MODULE != 1

  } // anonymous namespace

  //! Convenience wrapper around the smuth transform.
  class HFTransform : public python::Object
  {
      //! \brief Initializer constructor.
      //! \details private so it cannot be constructed without a call throw
      //! hf_transform.
      HFTransform(PyObject *_in) : python::Object(_in) {}
      //! \brief Initializer constructor.
      //! \details private so it cannot be constructed without a call throw
      //! hf_transform.
      HFTransform() : python::Object() {}
  
    public:
      //! Copy constructor.
      HFTransform(const HFTransform &_in) : python::Object(_in) {}
      //! Initialization constructor.
      template<class T0, class T1> 
        HFTransform( Eigen::DenseBase<T0> const &_lattice,
                        Eigen::DenseBase<T1> const &_supercell )
          { init_(_lattice, _supercell); }
      //! Initialization constructor.
      HFTransform(Structure const &_lattice, Structure const &_supercell)
        { init_(_lattice->cell, _supercell->cell); }
  
      //! Returns constant reference to transform object.
      math::rMatrix3d const & transform() const 
        { return ((PyHFTObject* const)object_)->transform; }
      //! Returns reference to transform object.
      math::rMatrix3d & transform() 
        { return ((PyHFTObject*)object_)->transform; }
      //! Returns constant reference to quotient object.
      math::iVector3d const & quotient() const 
        { return ((PyHFTObject* const)object_)->quotient; }
      //! Returns reference to quotient object.
      math::iVector3d & quotient() 
        { return ((PyHFTObject*)object_)->quotient; }
  
#     include "macro.hpp"
      //! Computes hf indices of position \a _pos.
      inline math::iVector3d indices(math::rVector3d const &_pos) const
      {
        PYLADA_HFTRANSFORM_SHARED1(quotient(), transform(), _pos, PYLADA_PYTHROW,);
        return vector_result;
      }
      //! \brief Computes linear hf index from non-linear hf index.
      inline size_t flat_index(math::iVector3d const &_index, int _site=-1)
      {
        PYLADA_HFTRANSFORM_SHARED0(quotient(), _index, _site);
        return flat_result;
      }
      //! Computes linear hf index of position \a _pos.
      inline size_t flat_index(math::rVector3d const &_pos, int _site=-1)
      {
        PYLADA_HFTRANSFORM_SHARED1(quotient(), transform(), _pos, PYLADA_PYTHROW,);
        PYLADA_HFTRANSFORM_SHARED0(quotient(), vector_result, _site);
        return flat_result;
      }
      //! Number of unit-cells in the supercell.
      size_t size() const { return PYLADA_HFTRANSFORM_SHARED2(quotient()); }
#     include "macro.hpp"
    private:
      //! creates a hf transform from scratch.
      template<class T0, class T1> 
        void init_(Eigen::DenseBase<T0> const &_lattice, Eigen::DenseBase<T1> const &_supercell);
  };
      
  template<class T0, class T1> 
    void HFTransform::init_( Eigen::DenseBase<T0> const &_lattice, 
                             Eigen::DenseBase<T1> const &_supercell )
    {
      BOOST_STATIC_ASSERT((boost::is_same<typename Eigen::DenseBase<T0>::Scalar,
                                          types::t_real>::value));
      BOOST_STATIC_ASSERT((boost::is_same<typename Eigen::DenseBase<T1>::Scalar,
                                          types::t_real>::value));
      python::Object dummy(object_);
      object_ = hftransform_type()->tp_alloc(hftransform_type(), 0);
      if(not object_) return;
      if(not _init_hft((PyHFTObject*)object_, _lattice, _supercell)) release();
    }
#endif 
