#ifndef PYLADA_SMITH_NORMAL_FORM_H
#define PYLADA_SMITH_NORMAL_FORM_H

#include "crystal/types.h"
#include <Eigen/Core>

namespace pylada {
void smith_normal_form(types::t_int *_S, types::t_int *_L, types::t_int *_R);
}
#endif
