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

#include "PyladaConfig.h"

#include <Python.h>
#include <structmember.h>
#define PY_ARRAY_UNIQUE_SYMBOL pylada_enum_ARRAY_API
#define NO_IMPORT_ARRAY
#include <python/include_numpy.h>

#include <vector>
#include <algorithm>


#define PYLADA_NO_IMPORT
#include <python/python.h>
#include <errors/exceptions.h>

#include "ndimiterator.h"

namespace Pylada
{
  namespace enumeration
  {
    //! Creates a new ndimiterator.
    NDimIterator* PyNDimIterator_New()
    {
      NDimIterator* result = (NDimIterator*) ndimiterator_type()->tp_alloc(ndimiterator_type(), 0);
      if(not result) return NULL;
      result->yielded = NULL;
      new(&result->ends) std::vector<t_ndim>;
      new(&result->counter) std::vector<t_ndim>;
      return result;
    }
    //! Creates a new ndimiterator with a given type.
    NDimIterator* PyNDimIterator_NewWithArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
    {
      NDimIterator* result = (NDimIterator*)_type->tp_alloc(_type, 0);
      if(not result) return NULL;
      result->yielded = NULL;
      new(&result->ends) std::vector<t_ndim>;
      new(&result->counter) std::vector<t_ndim>;
      return result;
    }

    // Creates a new ndimiterator with a given type, also calling initialization.
    NDimIterator* PyNDimIterator_NewFromArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
    {
      NDimIterator* result = PyNDimIterator_NewWithArgs(_type, _args, _kwargs);
      if(result == NULL) return NULL;
      if(_type->tp_init((PyObject*)result, _args, _kwargs) < 0) {Py_DECREF(result); return NULL; }
      return result;
    }

    static int gcclear(NDimIterator *self)
    { 
      Py_CLEAR(self->yielded);
      return 0;
    }


    // Function to deallocate a string atom.
    static void ndimiterator_dealloc(NDimIterator *_self)
    {
      gcclear(_self);
  
      // calls destructor explicitely.
      PyTypeObject* ob_type = _self->ob_type;
      _self->~NDimIterator();

      ob_type->tp_free((PyObject*)_self);
    }
  
    // Function to initialize an atom.
    static int ndimiterator_init(NDimIterator* _self, PyObject* _args, PyObject *_kwargs)
    {
      if(_args == NULL)
      {
        PYLADA_PYERROR(TypeError, "NDimIterator expects at least one argument.");
        return -1;
      }
      Py_ssize_t const N = PyTuple_Size(_args);
      if(N == 0)
      {
        PYLADA_PYERROR(TypeError, "NDimIterator expects at least one argument.");
        return -1;
      }
      if(_kwargs != NULL and PyDict_Size(_kwargs))
      {
        PYLADA_PYERROR(TypeError, "NDimIterator does not expect keyword arguments.");
        return -1;
      }
      _self->ends.clear();
      _self->ends.reserve(N);
      for(Py_ssize_t i(0); i < N; ++i)
      {
        PyObject *item = PyTuple_GET_ITEM(_args, i);
        if(PyLong_Check(item)) _self->ends.push_back( PyLong_AsLong(item));
        else if(PyInt_Check(item)) _self->ends.push_back( PyInt_AS_LONG(item));
        else
        {
          PYLADA_PYERROR(TypeError, "Unknown type in NDimIterator.");
          return -1;
        }
        if(_self->ends.back() <= 0)
        {
          PYLADA_PYERROR(ValueError, "NDimIterator does not expect negative or null arguments.");
          return -1;
        }
      } 
      _self->counter.resize(_self->ends.size());
      std::fill(_self->counter.begin(), _self->counter.end()-1, 1);
      _self->counter.back() = 0;
      if(_self->yielded != NULL)
      {
        PyArrayObject* dummy = _self->yielded;
        _self->yielded = NULL;
        Py_DECREF(dummy);
      }
      typedef python::numpy::type<t_ndim> t_type;
      npy_intp d[1] = {(npy_intp)N};
      _self->yielded = (PyArrayObject*)
          PyArray_SimpleNewFromData(1, d, t_type::value, &_self->counter[0]);
      if(not _self->yielded) return -1;
      PyArray_CLEARFLAGS(_self->yielded,  NPY_ARRAY_WRITEABLE);
      PyArray_ENABLEFLAGS(_self->yielded,  NPY_ARRAY_C_CONTIGUOUS);
      Py_INCREF(_self);
      PyArray_SetBaseObject(_self->yielded, (PyObject*)_self);
      return 0;
    }
  
    static int traverse(NDimIterator *self, visitproc visit, void *arg)
    {
      Py_VISIT(self->yielded);
      return 0;
    }
  
    static PyObject* self(PyObject* _self)
    {
      Py_INCREF(_self);
      return _self;
    }

    static PyObject* next(NDimIterator* _self)
    {
#     ifdef PYLADA_DEBUG
        if(_self->ends.size() != _self->counter.size())
        {
          PYLADA_PYERROR(internal, "Counter and Ends have different size.");
          return NULL;
        }
#     endif
      std::vector<t_ndim>::reverse_iterator i_first = _self->counter.rbegin();
      std::vector<t_ndim>::reverse_iterator const i_last = _self->counter.rend();
      std::vector<t_ndim>::const_reverse_iterator i_end = _self->ends.rbegin();
      for(; i_first != i_last; ++i_first, ++i_end)
        if(*i_first == *i_end) *i_first = 1;
        else
        {
          ++(*i_first);
#         ifdef PYLADA_DEBUG
            if(_self->yielded == NULL)
            {
              PYLADA_PYERROR(internal, "Yielded was not initialized.");
              return NULL;
            }
            if(PyArray_DATA(_self->yielded) != (void*)&_self->counter[0])
            {
              PYLADA_PYERROR(internal, "Yielded does not reference counter.");
              return NULL;
            }
#         endif
          Py_INCREF(_self->yielded);
          return (PyObject*)_self->yielded;
        }
      PyArrayObject* dummy = _self->yielded;
      _self->yielded = NULL;
      Py_DECREF(dummy);
      PyErr_SetNone(PyExc_StopIteration);
      return NULL;
    }


    // Returns pointer to ndimiterator type.
    PyTypeObject* ndimiterator_type()
    {
      static PyTypeObject dummy = {
          PyObject_HEAD_INIT(NULL)
          0,                                 /*ob_size*/
          "pylada.enum.cppwrappers.NDimIterator",   /*tp_name*/
          sizeof(NDimIterator),              /*tp_basicsize*/
          0,                                 /*tp_itemsize*/
          (destructor)ndimiterator_dealloc,  /*tp_dealloc*/
          0,                                 /*tp_print*/
          0,                                 /*tp_getattr*/
          0,                                 /*tp_setattr*/
          0,                                 /*tp_compare*/
          0,                                 /*tp_repr*/
          0,                                 /*tp_as_number*/
          0,                                 /*tp_as_sequence*/
          0,                                 /*tp_as_mapping*/
          0,                                 /*tp_hash */
          0,                                 /*tp_call*/
          0,                                 /*tp_str*/
          0,                                 /*tp_getattro*/
          0,                                 /*tp_setattro*/
          0,                                 /*tp_as_buffer*/
          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_ITER, /*tp_flags*/
          "Defines an N-dimensional iterator.\n\n"
          "This iterator holds a numpy array where each element is incremented\n"
          "between 1 and a specified value. Once an element (starting from the\n"
          "last) reaches the specified value, it is reset to 1 and the next is\n"
          "element is incremented. It looks something like this:\n\n"
          "  >>> for u in NDimIterator(2, 3): print u\n"
          "  [1 1]\n"
          "  [1 2]\n"
          "  [1 3]\n"
          "  [2 1]\n"
          "  [2 2]\n"
          "  [2 3]\n\n"
          "The last item is incremented up to 3, and the first item up to 2, as\n"
          "per the input. NDimIterator accepts any number of arguments, as long\n"
          "as they are integers. The yielded result ``u`` is a read-only numpy\n"
          "array.\n\n"
          ".. warning::\n\n"
          "   The yielded array *always* points to the same address in memory:\n\n"
          "   .. code-block:: python\n"
          "      iterator = NDimIterator(2, 3)\n"
          "      a = iterator.next()\n"
          "      assert all(a == [1, 1])\n"
          "      b = iterator.next()\n"
          "      assert all(b == [1, 2])\n"
          "      assert all(a == b)\n"
          "      assert a is b\n\n"
          "   If you need to keep track of a prior value, use numpy's copy_\n"
          "   function.\n\n"
          "   The other effect is that :py:class:`NDimIterator` cannot be used\n"
          "   with zip_ and similar functions:\n\n"
          "   .. code-block:: python\n\n"
          "     for u, v in zip( NDimIterator(5, 5, 5), \n"
          "                      product(range(1, 6), repeat=3) ):\n"
          "       assert all(u == 1)\n\n"
          ".. copy_: http://docs.scipy.org/doc/numpy/reference/generated/numpy.copy.html\n"
          ".. zip_: http://docs.python.org/library/functions.html#zip\n",
          (traverseproc)traverse,            /* tp_traverse */
          (inquiry)gcclear,                  /* tp_clear */
          0,		                             /* tp_richcompare */
          0,                                 /* tp_weaklistoffset */
          (getiterfunc)self,                 /* tp_iter */
          (iternextfunc)next,                /* tp_iternext */
          0,                                 /* tp_methods */
          0,                                 /* tp_members */
          0,                                 /* tp_getset */
          0,                                 /* tp_base */
          0,                                 /* tp_dict */
          0,                                 /* tp_descr_get */
          0,                                 /* tp_descr_set */
          0,                                 /* tp_dictoffset */
          (initproc)ndimiterator_init,       /* tp_init */
          0,                                 /* tp_alloc */
          (newfunc)PyNDimIterator_NewWithArgs,  /* tp_new */
      };
      return &dummy;
    }

  } // namespace Crystal
} // namespace Pylada
