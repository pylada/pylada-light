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
#define PY_ARRAY_UNIQUE_SYMBOL pylada_enum_ARRAY_API
#define NO_IMPORT_ARRAY
#include <python/include_numpy.h>

#include <vector>
#include <algorithm>


#define PYLADA_NO_IMPORT
#include <errors/exceptions.h>
#include <python/python.h>

#include "manipulations.h"

namespace Pylada
{
  namespace enumeration
  {
    //! Creates a new manipulations.
    Manipulations* PyManipulations_New()
    {
      Manipulations* result = (Manipulations*) manipulations_type()->tp_alloc(manipulations_type(), 0);
      if(not result) return NULL;
      new(&result->substituted) std::vector<t_ndim>;
      new(&result->counter) std::vector<t_ndim>;
      new(&result->substitutions) std::vector< std::vector<t_ndim> >;
      new(&result->i_first) std::vector< std::vector<t_ndim> > :: iterator;
      new(&result->i_end) std::vector< std::vector<t_ndim> > :: iterator;
      result->arrayin = NULL;
      result->arrayout = NULL;
      return result;
    }
    //! Creates a new manipulations with a given type.
    Manipulations* PyManipulations_NewWithArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
    {
      Manipulations* result = (Manipulations*)_type->tp_alloc(_type, 0);
      if(not result) return NULL;
      new(&result->substituted) std::vector<t_ndim>;
      new(&result->counter) std::vector<t_ndim>;
      new(&result->substitutions) std::vector<t_ndim>;
      new(&result->i_first) std::vector< std::vector<t_ndim> > :: iterator;
      new(&result->i_end) std::vector< std::vector<t_ndim> > :: iterator;
      result->arrayin = NULL;
      result->arrayout = NULL;
      return result;
    }

    // Creates a new manipulations with a given type, also calling initialization.
    Manipulations* PyManipulations_NewFromArgs(PyTypeObject* _type, PyObject *_args, PyObject *_kwargs)
    {
      Manipulations* result = PyManipulations_NewWithArgs(_type, _args, _kwargs);
      if(result == NULL) return NULL;
      if(_type->tp_init((PyObject*)result, _args, _kwargs) < 0) {Py_DECREF(result); return NULL; }
      return result;
    }

    static int gcclear(Manipulations *self)
    { 
      Py_CLEAR(self->arrayout);
      Py_CLEAR(self->arrayin);
      return 0;
    }


    // Function to deallocate a string atom.
    static void dealloc(Manipulations *_self)
    {
      gcclear(_self);
  
      // calls destructor explicitely.
      PyTypeObject* ob_type = _self->ob_type;
      _self->~Manipulations();

      ob_type->tp_free((PyObject*)_self);
    }

    static npy_intp check_substitutions(PyArrayObject* _subs)
    {
      if(not PyArray_Check(_subs))
      {
        PYLADA_PYERROR(TypeError, "first argument should be a numpy array.");
        return 0;
      }
      npy_intp const ndim = PyArray_NDIM(_subs);
      if(not PyArray_ISINTEGER(_subs))
      {
        PYLADA_PYERROR(TypeError, "Non-integral permutation array.");
        return 0;
      }
      if(ndim != 2)
      {
        PYLADA_PYERROR(TypeError, "Permutation array should be 2d.");
        return 0;
      }
      return PyArray_DIM(_subs, 0);
    }
  
    // Function to initialize an atom.
    static int init(Manipulations* _self, PyObject* _args, PyObject *_kwargs)
    {
      if(_kwargs != NULL and PyDict_Size(_kwargs))
      {
        PYLADA_PYERROR(TypeError, "Manipulations does not expect keyword arguments.");
        return -1;
      }
      int length = 0;
      PyArrayObject *substitutions;
      if(not PyArg_ParseTuple(_args, "O", (PyObject**)&substitutions))
        return -1;
      npy_intp nd = check_substitutions((PyArrayObject*)substitutions);
      if(nd == 0) return 0; 
      npy_intp size = PyArray_DIM(substitutions, 1);
      _self->substitutions.clear();

      python::Object iterator = PyArray_IterNew((PyObject*)substitutions);
      int const type_num = PyArray_TYPE(substitutions);
      _self->substitutions.resize(nd);
      std::vector< std::vector<t_ndim> >::iterator i_first = _self->substitutions.begin();
      while( PyArray_ITER_NOTDONE(iterator.borrowed()) )
      {
        for(npy_intp i(0); i < size; ++i)
        {
          t_ndim value 
              = python::numpy::to_type<t_ndim>(iterator.borrowed(), type_num); 
          if(value < 0) value += size;
          if(value < 0 or value >= size)
          {
            PYLADA_PYERROR(IndexError, "Invalid permutation index.");
            return -1;
          }
          i_first->push_back(value);
          PyArray_ITER_NEXT(iterator.borrowed());
        }
        ++i_first;
      }
      _self->i_first = _self->substitutions.begin();
      _self->i_end = _self->substitutions.end();
      _self->is_first = true;
      if(_self->arrayin != NULL)
      {
        PyObject* dummy = (PyObject*) _self->arrayin;
        _self->arrayin = NULL;
        Py_DECREF(dummy);
      }
      Py_INCREF(Py_None);
      _self->arrayin = Py_None;
      typedef python::numpy::type<t_ndim> t_type;
      _self->counter.resize(size);
      std::fill(_self->counter.begin(), _self->counter.end(), 0);
      npy_intp d[1] = {size};
      _self->arrayout = (PyArrayObject*)
          PyArray_SimpleNewFromData(1, d, t_type::value, &_self->counter[0]);
      if(not _self->arrayout) return -1;

      PyArray_CLEARFLAGS(_self->arrayout,  NPY_ARRAY_WRITEABLE);
      PyArray_ENABLEFLAGS(_self->arrayout,  NPY_ARRAY_C_CONTIGUOUS);
      Py_INCREF(_self);
      PyArray_SetBaseObject(_self->arrayout, (PyObject*)_self);
      return 0;
    }

    static PyObject* __call__(Manipulations *_self, PyObject *_in)
    {
      if(_self->substitutions.size() == 0) 
      {
        Py_INCREF(_self);
        return (PyObject*)_self;
      }
      if(not PyArray_Check(_in))
      {
        PYLADA_PYERROR(TypeError, "second argument should be a numpy array "
                                "yielded by an NDimIterator.");
        return NULL;
      }
      if( PyArray_TYPE((PyArrayObject*)_in) != python::numpy::type<t_ndim>::value ) 
      {
        PYLADA_PYERROR(TypeError, "second argument should be a numpy array "
                                "yielded by an NDimIterator.");
        return NULL;
      }
      if(PyArray_NDIM((PyArrayObject*)_in) != 1)
      {
        PYLADA_PYERROR(TypeError, "second argument should be a numpy array "
                                "with ndim == 1.");
        return NULL;
      }
      if(PyArray_DIM((PyArrayObject*)_in, 0) != _self->counter.size())
      {
        PYLADA_PYERROR(TypeError, "first argument should be a numpy array "
                                "of length > 0.");
        return NULL;
      }
#     ifdef PYLADA_DEBUG
        if(not (PyArray_FLAGS((PyArrayObject*)_in) && NPY_ARRAY_C_CONTIGUOUS) )
        {
          PYLADA_PYERROR( TypeError, 
                        "second argument should be a c-contiguous numpy array." );
          return NULL;
        }
        if(PyArray_STRIDE((PyArrayObject*)_in, 0) != sizeof(t_ndim) / sizeof(char))
        {
          PYLADA_PYERROR(TypeError, "Wrong stride for first argument.");
          return NULL;
        }
#     endif
      PyObject* dummy = _self->arrayin;
      _self->arrayin = _in;
      Py_INCREF(_in);
      Py_DECREF(dummy);
      Py_INCREF(_self);
      return (PyObject*)_self;
    } 
    static PyObject* call(PyObject *_self, PyObject *_in, PyObject* _kwargs)
    {
      if(_in == NULL)
      {
        PYLADA_PYERROR(TypeError, "The manipulations object expects one argument.");
        return NULL;
      }
      if(PyTuple_Size(_in) != 1)
      {
        PYLADA_PYERROR(TypeError, "The manipulations object expects only one argument.");
        return NULL;
      }
      return __call__((Manipulations*)_self, PyTuple_GET_ITEM(_in, 0));
    };
  
    static int traverse(Manipulations *self, visitproc visit, void *arg)
    {
      Py_VISIT(self->arrayout);
      Py_VISIT(self->arrayin);
      return 0;
    }
  
    static PyObject* self(PyObject* _self)
    {
      Py_INCREF(_self);
      return _self;
    }

    static PyObject* next(Manipulations* _self)
    {
      if(_self->substitutions.size() == 0)
      {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
      }
      if(_self->is_first) _self->is_first = false; 
      else if((++_self->i_first) == _self->i_end)
      {
        _self->i_first = _self->substitutions.begin();
        _self->is_first = true;
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
      }
      std::vector<t_ndim>::iterator i_out = _self->counter.begin();
      std::vector<t_ndim>::iterator const i_end = _self->counter.end();
      std::vector<t_ndim>::iterator i_manip = _self->i_first->begin();
      t_ndim * const i_data = (t_ndim*) PyArray_DATA((PyArrayObject *)_self->arrayin); 
      for(; i_out != i_end; ++i_out, ++i_manip)
        *i_out = *(i_data + *i_manip);
      Py_INCREF(_self->arrayout);
      return (PyObject*)_self->arrayout;
    }

    // Returns pointer to manipulations type.
    PyTypeObject* manipulations_type()
    {
#     ifdef PYLADA_DECLARE
#       error PYLADA_DECLARE already declared
#     endif
#     define PYLADA_DECLARE(name, func, args, doc) \
        {#name, (PyCFunction)func, METH_ ## args, doc} 
      static PyMethodDef methods[] = {
        PYLADA_DECLARE(__call__, __call__, NOARGS, "Sets array to manipulate."),
          {NULL}  /* Sentinel */
      };
#     undef PYLADA_DECLARE
      static PyTypeObject dummy = {
          PyObject_HEAD_INIT(NULL)
          0,                                 /*ob_size*/
          "pylada.enum.cppwrappers.Manipulations",   /*tp_name*/
          sizeof(Manipulations),              /*tp_basicsize*/
          0,                                 /*tp_itemsize*/
          (destructor)dealloc,               /*tp_dealloc*/
          0,                                 /*tp_print*/
          0,                                 /*tp_getattr*/
          0,                                 /*tp_setattr*/
          0,                                 /*tp_compare*/
          0,                                 /*tp_repr*/
          0,                                 /*tp_as_number*/
          0,                                 /*tp_as_sequence*/
          0,                                 /*tp_as_mapping*/
          0,                                 /*tp_hash */
          call,                              /*tp_call*/
          0,                                 /*tp_str*/
          0,                                 /*tp_getattro*/
          0,                                 /*tp_setattro*/
          0,                                 /*tp_as_buffer*/
          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_ITER, /*tp_flags*/
          "Defines manipulations iterator.\n\n"
          "Given a series of vector manipulations and an input vector,\n"
          "Performs a series of manipulations:\n\n"
          "  >>> print x\n"
          "  array([5, 6 ,7], dtype='int16')\n"
          "  >>> manips = Manipulations([[0, 1, 2], [2, 1, 0]])\n"
          "  >>> for out in manips(x): print out\n"
          "  [5, 6 , 7]\n"
          "  [7, 6 , 5]\n\n"
          "The first manipulation above leaves x unchanged, wheras the second\n"
          "puts item 2 in position 0 (that's the 2 in column 0, row 1) and\n"
          "item 0 in position 2 (the 0 in column 2, row 1).\n\n"
          ".. warning:: The type of x must be int16.\n",
          (traverseproc)traverse,            /* tp_traverse */
          (inquiry)gcclear,                  /* tp_clear */
          0,		                     /* tp_richcompare */
          0,                                 /* tp_weaklistoffset */
          (getiterfunc)self,                 /* tp_iter */
          (iternextfunc)next,                /* tp_iternext */
          methods,                           /* tp_methods */
          0,                                 /* tp_members */
          0,                                 /* tp_getset */
          0,                                 /* tp_base */
          0,                                 /* tp_dict */
          0,                                 /* tp_descr_get */
          0,                                 /* tp_descr_set */
          0,                                 /* tp_dictoffset */
          (initproc)init,                    /* tp_init */
          0,                                 /* tp_alloc */
          (newfunc)PyManipulations_NewWithArgs,  /* tp_new */
      };
      return &dummy;
    }

  } // namespace Crystal
} // namespace Pylada

