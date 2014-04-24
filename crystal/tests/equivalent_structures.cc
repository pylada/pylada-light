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


#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>

#include <math/smith_normal_form.h>

#define PYLADA_DOASSERT(a,b) \
        { \
          if((not (a)))\
          { \
            std::cerr << __FILE__ << ", line: " << __LINE__ << "\n" << b; \
            throw 0;\
          }\
        }
#include "../supercell.h"
#include "../space_group.h"
#include "../equivalent_structures.h"

using namespace Pylada;
using namespace Pylada::crystal;
using namespace Pylada::math;
typedef Structure< PYLADA_TYPE > t_Str;
#define PYLADA_RAND(s) types::t_real(rand()%s)/types::t_real(s)
#if PYLADA_INCREMENT == 0
#  define PYLADA_INIT "Si"
#  define PYLADA_DOASSERT_TYPE0 PYLADA_DOASSERT(c[0]->type == "Si", "Wrong first type.\n"); 
#  define PYLADA_DOASSERT_TYPE1 PYLADA_DOASSERT(c[1]->type == "Ge", "Wrong first type.\n"); 
#elif PYLADA_INCREMENT == 1
#  define PYLADA_INIT "Si", "Ge"
#  define PYLADA_DOASSERT_TYPE0 \
     PYLADA_DOASSERT(c[0]->type.size() == 1, "Wrong number of types.\n");\
     PYLADA_DOASSERT(c[0]->type[0] == "Si", "Wrong first type.\n");
#  define PYLADA_DOASSERT_TYPE1 \
     PYLADA_DOASSERT(c[1]->type.size() == 2, "Wrong number of types.\n");\
     PYLADA_DOASSERT(c[1]->type[0] == "Ge", "Wrong first type.\n");\
     PYLADA_DOASSERT(c[1]->type[1] == "Si", "Wrong first type.\n");
#elif PYLADA_INCREMENT == 2
#  define PYLADA_INIT "Si", "Ge"
#  define PYLADA_DOASSERT_TYPE0 \
    { PYLADA_TYPE cmp; cmp.insert("Si"); \
      PYLADA_DOASSERT(c[0]->type.size() == 1, "Wrong number of types.\n");\
      PYLADA_DOASSERT(c[0]->type == cmp, "Wrong first type.\n"); }
#  define PYLADA_DOASSERT_TYPE1 \
    { PYLADA_TYPE cmp; cmp.insert("Si"); cmp.insert("Ge");                \
      PYLADA_DOASSERT(c[1]->type.size() == 2, "Wrong number of types.\n");\
      PYLADA_DOASSERT(c[1]->type == cmp, "Wrong first type.\n"); }
#endif

void scale(t_Str const &_A, t_Str const &_B)
{
  t_Str B = _B.copy();
  PYLADA_DOASSERT(equivalent(_A, B, true, 1e-5), "A is not equivalent to B.\n");
  B.scale() = 3.0;
  PYLADA_DOASSERT(not equivalent(_A, B, true, 1e-5), "A is equivalent to B.\n");
  PYLADA_DOASSERT(equivalent(_A, B, false, 1e-5), "A is not equivalent to B.\n");
  B.cell() *= 0.5;
  foreach(t_Str::reference b, B) b->pos *= 0.5;
  PYLADA_DOASSERT(not equivalent(_A, B, true, 1e-5), "A is equivalent to B.\n");
  PYLADA_DOASSERT(equivalent(_A, B, false, 1e-5), "A is not equivalent to B.\n");
}
void motif(t_Str const &_A, t_Str const &_B)
{
  boost::shared_ptr<t_SpaceGroup> sg = cell_invariants(_A.cell());
  t_SpaceGroup::const_iterator i_first = sg->begin();
  t_SpaceGroup::const_iterator const i_end = sg->begin();
  for(; i_first != i_end; ++i_first)
  {
    t_Str B = _B.copy();
    foreach(t_Str::reference b, B)
    {
      b->pos = i_first->linear() * b->pos;
      scale(_A, B);
    }
  }
}
void basis(t_Str const &_A, t_Str const &_B)
{
  math::Affine3d affine(math::AngleAxis(0.5*math::pi, math::rVector3d::UnitX()));
  t_Str B = _B.transform(affine);
  motif(_A, B);
  affine = math::AngleAxis(-math::pi, math::rVector3d::UnitX());
  B = _B.transform(affine);
  motif(_A, B);
  affine = math::AngleAxis(-0.13*math::pi, math::rVector3d::UnitX());
  B = _B.transform(affine);
  motif(_A, B);
  affine = math::Translation(0.25, 0.25, 0.25);
  B = _B.transform(affine);
  motif(_A, B);

  affine =   math::AngleAxis(PYLADA_RAND(100)*2.0*math::pi, math::rVector3d::UnitX())
           * math::Translation(PYLADA_RAND(100)-0.5, PYLADA_RAND(100)-0.5, PYLADA_RAND(100)-0.5);
  B = _B.transform(affine);
  motif(_A, B);
}
void decoration(t_Str const &_A, t_Str const &_B, t_Str const _latt)
{
  // map A's atoms with linear smith index.
  t_SmithTransform st = smith_transform(_latt.cell(), _A.cell());
  std::vector<size_t> indices(_A.size());
  t_Str::const_iterator i_atom = _A.begin();
  t_Str::const_iterator const i_end = _A.end();
  for(size_t i(0); i_atom != i_end; ++i_atom, ++i)
    indices[linear_smith_index(st, i_atom->site(), i_atom->pos() - _latt[i_atom->site()]->pos)] = i;

  // loop over decoration translations
  for(i_atom = _A.begin(); i_atom != i_end; ++i_atom)
  {
    if(i_atom->site() != _A[0]->site) continue; // only primitive lattice translations.
    math::rVector3d const trans = i_atom->pos() - _A[0]->pos;
    t_Str B = _A.copy();
    std::vector<size_t>::const_iterator i_ind = indices.begin();
    std::vector<size_t>::const_iterator const i_ind_end = indices.end();
    for(; i_ind != i_ind_end; ++i_ind)
    {
      math::rVector3d const vec = _A[*i_ind]->pos + trans - _latt[_A[*i_ind]->site]->pos;
      if(not is_integer(_latt.cell().inverse() * vec))
      {
        std::cout << ~(_latt.cell().inverse() * vec) << "\n";
        PYLADA_DOASSERT(false, "Not on lattice.\n" )
      }
      B[ indices[linear_smith_index(st, _A[*i_ind]->site, vec)] ] = _A[*i_ind];
    }
    basis(_A, B);
  }
}

void clear(std::string const &) {}
void clear(std::vector<std::string> &_in) { _in.clear(); }
void clear(std::set<std::string> &_in) { _in.clear(); }
void push_back(std::string &_in, std::string const &_type)
  { _in = _type; }
void push_back(std::vector<std::string> &_in, std::string const &_type)
  { _in.push_back(_type); }
void push_back(std::set<std::string> &_in, std::string const &_type)
  { _in.insert(_type); }
void set(std::string &_in, std::string const &_type)
  { _in = _type; }
void set(std::vector<std::string> &_in, std::string const &_type)
  { _in[0] = _type; }
void set(std::set<std::string> &_in, std::string const &_type)
  { _in.clear(); _in.insert(_type); }

int main()
{
  types::t_int const seed = time(NULL); // 1317785207; 
  std::cout << seed << "\n";
  srand(seed);

  t_Str A;
  A.set_cell(0,0.5,0.5)
            (0.5,0,0.5)
            (0.5,0.5,0);
  A.add_atom(0,0,0, "Si")
            (0.25,0.25,0.25, PYLADA_INIT);
  basis(A, A);

  clear(A[0]->type); push_back(A[0]->type, "Si");
  clear(A[1]->type); push_back(A[1]->type, "Si");
  t_Str latt = A.copy();
  rMatrix3d cell;
  cell << 3, 0, 0, 0, 0.5, -0.5, 0, 0.5, 0.5;
  A = supercell(latt, cell);
  set(A[0]->type, "Ge");
  set(A[1]->type, "Ge");
  set(A[3]->type, "Ge");

  decoration(A, A, latt);

  cell << 2, 0, 0, 0, 2, 0, 0, 0, 2;
  A = supercell(latt, cell);
  for(size_t i(0); i < 5; ++i)
  {
    t_Str::iterator i_atom = A.begin();
    t_Str::iterator const i_end = A.end();
    types::t_real const x = PYLADA_RAND(100) * 0.5;
    for(; i_atom != i_end; ++i_atom) set(i_atom->type(), PYLADA_RAND(100) > x ? "Si": "Ge");
    decoration(A, A, latt);
  }
 
  for(size_t i(0); i < 10; ++i)
  {
    do
    {
      cell << rand()%10-5, rand()%10-5, rand()%10-5,
              rand()%10-5, rand()%10-5, rand()%10-5,
              rand()%10-5, rand()%10-5, rand()%10-5;
    } while( is_null(cell.determinant()) );
    if(cell.determinant() < 0) cell.col(0).swap(cell.col(1));
//   if(i < 4) continue;
    A = supercell(latt, latt.cell() * cell);
    t_Str::iterator i_atom = A.begin();
    t_Str::iterator const i_end = A.end();
    types::t_real const x = PYLADA_RAND(100) * 0.5;
    for(; i_atom != i_end; ++i_atom) set(i_atom->type(), PYLADA_RAND(100) > x ? "Si": "Ge");
    decoration(A, A, latt);
  }

  return 0;
}
