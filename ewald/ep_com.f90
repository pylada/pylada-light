module ep_param
  implicit none

  integer, parameter :: dbl=selected_real_kind(14,200)
  
  integer                     :: nspec_tot
  integer       , allocatable :: id(:)
  real(kind=dbl), allocatable :: rad_ion(:), charge_ion(:)
  integer       , allocatable :: ibond(:,:)
  real(kind=dbl), allocatable :: epslon(:), rsigma(:)
  real(kind=dbl)              :: rcut_const0, RCUT_ewaldA
  real(kind=dbl)              :: PEGS
  
end module ep_param