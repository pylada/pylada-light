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
#include "FCMangle.h"

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL pylada_ewald_ARRAY_API
#define NO_IMPORT_ARRAY
#include <python/include_numpy.h>

#define PYLADA_NO_IMPORT
#include <python/python.h>

#include "ewald.h"

extern "C" void FC_GLOBAL( ewaldf, EWALDF )
                (
                  const int *const,    // verbosity
                  double * const,      // Energy
                  double * const,      // forces (reduced)
                  double * const,      // forces (cartesian)
                  const double *const, // stress
                  const int *const,    // number of atoms
                  const double *const, // reduced atomic coordinates.
                  const double *const, // atomic charges
                  const double *const, // real space cutoff
                  const double *const, // cell vectors
                  const int *const,    // dimension of arrays.
                  int * const          // ERROR
                );
namespace Pylada
{
  namespace pcm
  {
    PyObject* ewald(PyObject *_module, PyObject* _args, PyObject* _kwargs)  
    {
      static char *kwlist[] = { const_cast<char*>("cell"),
                                const_cast<char*>("positions"),
                                const_cast<char*>("charges"),
                                const_cast<char*>("cutoff"), NULL };
      PyObject *cell = NULL;
      PyObject *positions = NULL;
      PyObject *charges = NULL;
      double cutoff = 0;
      if(not PyArg_ParseTupleAndKeywords( _args, _kwargs, "OOOd:ewals",
                                          kwlist, &cell, &positions,
                                          &charges, &cutoff) )
        return NULL;

      const int verbosity(0);
      double energy(0);
      int error;
      npy_intp dims[3] = {PyArray_DIM((PyArrayObject*)positions, 0), 3, 6};
      int const nptype = python::numpy::type<double>::value;
      python::Object forces = PyArray_ZEROS(2, dims, nptype, 0);
      if(not forces) return NULL;
      python::Object cforces = PyArray_ZEROS(2, dims, nptype, 0);
      if(not cforces) return NULL;
      python::Object stress = PyArray_ZEROS(1, &dims[2], nptype, 0);
      if(not stress) return NULL;
      int const n0 = dims[0];

      FC_GLOBAL( ewaldf, EWALDF )
      (
        &verbosity,                                                 // verbosity
        &energy,                                                    // Energy
        (double*) PyArray_DATA((PyArrayObject*)forces.borrowed()),  // forces (reduced)
        (double*) PyArray_DATA((PyArrayObject*)cforces.borrowed()), // forces (cartesian)
        (double*) PyArray_DATA((PyArrayObject*)stress.borrowed()),  // stress
        &n0,                                                        // number of atoms
        (double*) PyArray_DATA((PyArrayObject*)positions),          // reduced atomic coordinates.
        (double*) PyArray_DATA((PyArrayObject*)charges),            // atomic charges
        &cutoff,                                                    // g-space cutoff in Ry.
        (double*) PyArray_DATA((PyArrayObject*)cell),               // cell vectors
        &n0,                                                        // dimension of arrays.
        &error
      );
      if(error == 1)
      {
        PYLADA_PYERROR(internal, "Could not find optimal alpha for ewald summation.");
        return NULL;
      }
      python::Object result = PyTuple_New(4);
      if(not result) return NULL;
      PyObject *pyenergy = PyFloat_FromDouble(energy);
      if(not pyenergy) return NULL;
      PyTuple_SET_ITEM(result.borrowed(), 0, pyenergy);
      PyTuple_SET_ITEM(result.borrowed(), 1, forces.release());
      PyTuple_SET_ITEM(result.borrowed(), 2, cforces.release());
      PyTuple_SET_ITEM(result.borrowed(), 3, stress.release());
      return result.release();
    }
  } // namespace pcm
} // namespace Pylada
