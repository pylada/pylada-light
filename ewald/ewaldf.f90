subroutine ewaldf(ipr,eewald,fewa,fewac,stress, natot,rc,zz,ecut,avec,mxdnat, error)
  ! computes the ewald energy, forces and stresses.
  !
  ! adapted from jlm plane-wave program
  ! march 12, 1996. nb
  ! the g-sum is order n^2
  use ep_param    ! xz
  implicit none
  
  integer, parameter :: dble = 8
  
  
  integer, intent(in)          :: ipr              ! controls printing.
  real(kind=dble), intent(out) :: eewald           ! ewald energy on ouput.
  real(kind=dble), intent(out) :: fewa(3, mxdnat)  ! forces in lattice coordinates, in ry/au
  real(kind=dble), intent(out) :: fewac(3, mxdnat) ! forces in cartesian coordinates, in ry/au
  real(kind=dble), intent(out) :: stress(6)        ! stress tensor (ry)
  integer, intent(in)          :: natot            ! number of atoms.
  real(kind=dble), intent(in)  :: rc(3, mxdnat)    ! atomic positions in direct space, in au.
  real(kind=dble), intent(in)  :: zz(mxdnat)       ! charges of each atom in au.
  ! plane wave cutoff, not energy cutoff.          
  real(kind=dble), intent(in)  :: ecut             ! g-space cutoff in rydbergs.
  real(kind=dble), intent(in)  :: avec(3, 3)       ! unit cell in direct space, in au.
  integer, intent(in)          :: mxdnat           ! array dimension for atoms.
  integer, intent(out)         :: error            ! error: 0 => good, other -> failure


  real(kind=dble), parameter :: pi = 3.141592653589793e0
  real(kind=dble), parameter :: twopi = 2e0 * pi
  real(kind=dble), parameter :: zero = 0e0
  real(kind=dble), parameter :: um = 1e0
  real(kind=dble), parameter :: dois = 2e0
  real(kind=dble), parameter :: tres = 3e0
  real(kind=dble), parameter :: quatro = 4e0

  real(kind=dble), parameter :: small = 1e-12
  
  real(kind=dble) :: adot(3,3),bdot(3,3),bvec(3,3) 
  real(kind=dble) :: tau(3),ir(3),ig(3),rp(3)
  real(kind=dble) :: ssumg(6),ssumr(6),fsub(3),ssub(6),ssum0(6)  

  real(kind=dble) :: fsumr(3,mxdnat),fsumg(9,mxdnat)
  real(kind=dble), external :: boost_erfc
  real(kind=dble) :: dummy, arg, cosg, enorm, alpha, esub, esum0, esumg, exp1
  real(kind=dble) :: exp2, expg, expgi, expgr, gcut, esumr, factor, gdt
  real(kind=dble) :: gmod2, qpv, rmod, sepi, seps, sfac2, sfaci, sfacr, sing
  real(kind=dble) :: tot, vcell, r1cc, r2cc, r3cc
  real(kind=dble) :: dnatot, tot_charge_squared, tot_squared_charge, upperbound
  integer :: i, j, k, ii, jj, kk, idum, im2, imx, imk, jmk, ir3cc
  integer :: ir1cc, ir2cc, jm2, jmx, km2, kmk, kmx, l, m, n, n1, im
   
  error = 0
  tot_squared_charge = sum(zz(1:natot)*zz(1:natot))
  tot_charge_squared = sum(zz(1:natot))**2
  dnatot = natot

  alpha = 3.0d0
  upperbound = 1e0
  do while (alpha .gt. 0d0 .and. upperbound .gt. 1d-7 .and. abs(ecut) .ge. 1e-12) 
    alpha = alpha - 0.1d0
    upperbound = 2.d0 * tot_charge_squared * sqrt(alpha/pi) * boost_erfc( sqrt(ecut/4e0/alpha) )
  enddo
  if (upperbound .ge. 1e-7 .or. alpha .le. 0d0) then
    error = 1
  endif

  seps = sqrt(alpha)
  sepi = dois*seps/sqrt(pi)
 
 
  ! bdot(i,j)   metric in reciprocal space
   
  ! compute the lattice wave-vectors, the cell volume and
  ! the inner products.
 
  bvec(1,1) = avec(2,2)*avec(3,3) - avec(3,2)*avec(2,3)
  bvec(2,1) = avec(3,2)*avec(1,3) - avec(1,2)*avec(3,3)
  bvec(3,1) = avec(1,2)*avec(2,3) - avec(2,2)*avec(1,3)
  bvec(1,2) = avec(2,3)*avec(3,1) - avec(3,3)*avec(2,1)
  bvec(2,2) = avec(3,3)*avec(1,1) - avec(1,3)*avec(3,1)
  bvec(3,2) = avec(1,3)*avec(2,1) - avec(2,3)*avec(1,1)
  bvec(1,3) = avec(2,1)*avec(3,2) - avec(3,1)*avec(2,2)
  bvec(2,3) = avec(3,1)*avec(1,2) - avec(1,1)*avec(3,2)
  bvec(3,3) = avec(1,1)*avec(2,2) - avec(2,1)*avec(1,2)
 
  ! cell volume
 
  vcell = bvec(1,1)*avec(1,1) + bvec(2,1)*avec(2,1) + &
          bvec(3,1)*avec(3,1)

  if (abs(vcell) .lt. small) then
    error = 2
    return
  endif

 
  do 10 j=1,3
    bvec(1,j) = twopi*bvec(1,j)/vcell
    bvec(2,j) = twopi*bvec(2,j)/vcell
    bvec(3,j) = twopi*bvec(3,j)/vcell
 10    continue
       vcell = abs(vcell) 
       qpv = quatro*pi/vcell
 
!c     compute metric bdot(i,j)
 
       bdot(1,1) = bvec(1,1)*bvec(1,1) + bvec(2,1)*bvec(2,1) + &
                   bvec(3,1)*bvec(3,1)
       bdot(2,2) = bvec(1,2)*bvec(1,2) + bvec(2,2)*bvec(2,2) + &
                   bvec(3,2)*bvec(3,2)
       bdot(3,3) = bvec(1,3)*bvec(1,3) + bvec(2,3)*bvec(2,3) + &
                   bvec(3,3)*bvec(3,3)
       bdot(1,2) = bvec(1,1)*bvec(1,2) + bvec(2,1)*bvec(2,2) + &
                   bvec(3,1)*bvec(3,2)
       bdot(1,3) = bvec(1,1)*bvec(1,3) + bvec(2,1)*bvec(2,3) + &
                   bvec(3,1)*bvec(3,3)
       bdot(2,3) = bvec(1,2)*bvec(1,3) + bvec(2,2)*bvec(2,3) + &
                   bvec(3,2)*bvec(3,3)
       bdot(2,1) = bdot(1,2)
       bdot(3,1) = bdot(1,3)
       bdot(3,2) = bdot(2,3)
 
!c     compute metric in real space
 
       factor = (vcell/(twopi*twopi))*(vcell/(twopi*twopi))
       adot(1,1) = factor*(bdot(2,2)*bdot(3,3) - bdot(2,3)*bdot(2,3))
       adot(2,2) = factor*(bdot(3,3)*bdot(1,1) - bdot(3,1)*bdot(3,1))
       adot(3,3) = factor*(bdot(1,1)*bdot(2,2) - bdot(1,2)*bdot(1,2))
       adot(1,2) = factor*(bdot(1,3)*bdot(3,2) - bdot(1,2)*bdot(3,3))
       adot(1,3) = factor*(bdot(1,2)*bdot(2,3) - bdot(1,3)*bdot(2,2))
       adot(2,3) = factor*(bdot(2,1)*bdot(1,3) - bdot(2,3)*bdot(1,1))
       adot(2,1) = adot(1,2)
       adot(3,1) = adot(1,3)
       adot(3,2) = adot(2,3)
 
  ! limits g-sum to less than ecut.
  if( abs(ecut) .ge. 1e-12) then
    imk = int(sqrt(ecut/bdot(1,1))) + 1
    jmk = int(sqrt(ecut/bdot(2,2))) + 1
    kmk = int(sqrt(ecut/bdot(3,3))) + 1
  endif
  ! means r-sum never goes beyond erfc(15)
  imx = int(15e0/sqrt(alpha * adot(1,1))) + 1
  jmx = int(15e0/sqrt(alpha * adot(2,2))) + 1
  kmx = int(15e0/sqrt(alpha * adot(3,3))) + 1
 
!c     initialize sums : esum(g,r)   -   energy
!c                       fsum(g,r)   -   force
!c                       ssum(g,r)   -   stress
 
!c     note that the sums are in units of basis vectors
!c     (g) in units of bi's, and (r) in units of ai's
       esumg = zero
       esumr = zero
       ssumg = zero
       ssumr = zero
       fsumg = zero
       fsumr = zero
 
!c     start sum in g space
 
       im2 = 2*imk+1
       jm2 = 2*jmk+1
       km2 = 2*kmk+1
       do 30 i=1,im2
       ig(1) = i-imk-1
       do 30 j=1,jm2
       ig(2) = j-jmk-1
       do 30 k=1,km2
       ig(3) = k-kmk-1
         gmod2 = zero
         do 18 l=1,3
         do 18 m=1,3
           gmod2 = gmod2 + real(ig(l))*bdot(l,m)*real(ig(m))
 18      continue
         if (gmod2 .ne. zero) then
           arg = gmod2/(quatro*alpha)       
           expg = exp(-arg)/gmod2                      
           sfacr = zero     
           sfaci = zero 
           do 20 n1= 1,natot        
             gdt = twopi*(real(ig(1))*rc(1,n1) +  &
                          real(ig(2))*rc(2,n1) +  &
                          real(ig(3))*rc(3,n1))
             cosg = zz(n1)*cos(gdt)  
             sing = zz(n1)*sin(gdt)
             sfacr = sfacr + cosg  
             sfaci = sfaci + sing
             fsumg(4,n1) = - real(ig(1))*cosg 
             fsumg(5,n1) = - real(ig(2))*cosg 
             fsumg(6,n1) = - real(ig(3))*cosg 
             fsumg(7,n1) = real(ig(1))*sing
             fsumg(8,n1) = real(ig(2))*sing
             fsumg(9,n1) = real(ig(3))*sing
 20        continue
           sfac2 = sfacr*sfacr + sfaci*sfaci 
           exp1 = sfac2*expg      
           esumg = esumg + exp1
           exp2 = - (um/(alpha*dois) + dois/gmod2) * exp1
           expgr = sfacr*expg
           expgi = sfaci*expg
           do 26 n1=1,natot   
             fsumg(1,n1) = fsumg(1,n1) + fsumg(7,n1)*expgr +  &
                           fsumg(4,n1)*expgi
             fsumg(2,n1) = fsumg(2,n1) + fsumg(8,n1)*expgr +  &
                           fsumg(5,n1)*expgi
             fsumg(3,n1) = fsumg(3,n1) + fsumg(9,n1)*expgr +  &
                           fsumg(6,n1)*expgi
 26        continue 
           ssumg(1) = ssumg(1) + exp2 * real(ig(1)*ig(1))
           ssumg(2) = ssumg(2) + exp2 * real(ig(2)*ig(2))
           ssumg(3) = ssumg(3) + exp2 * real(ig(3)*ig(3))
           ssumg(4) = ssumg(4) + exp2 * real(ig(1)*ig(2))
           ssumg(5) = ssumg(5) + exp2 * real(ig(2)*ig(3))
           ssumg(6) = ssumg(6) + exp2 * real(ig(3)*ig(1))
         endif
 30    continue
!c                                     
       esumg = esumg - tot_charge_squared * 0.25d0 / alpha
       esumg = qpv*esumg
       ssumg = qpv*ssumg
       fsumg(:3,:natot) = dois*qpv*fsumg(:3,:natot)
 
!c     start sum in r space
 
       im2 = 2*imx+1
       jm2 = 2*jmx+1
       km2 = 2*kmx+1
       esum0 = zero
       do 38 i = 1,6
         ssum0(i) = zero
 38    continue
       do 40 i=1,im2
       ir(1) = i-imx-1
       do 40 j=1,jm2
       ir(2) = j-jmx-1
       do 40 k=1,km2
       ir(3) = k-kmx-1
         rmod = zero
         do 39 l=1,3
         do 39 m=1,3
           rmod = rmod + real(ir(l))*adot(l,m)*real(ir(m))
 39      continue
         if (rmod .ne. zero) then
           rmod = sqrt(rmod)
           arg = seps*rmod
           if (arg .lt. 25.0) then
             exp1 = boost_erfc(arg) / rmod
             exp2 = (exp1 + sepi*exp(-arg*arg))/(rmod*rmod)
             esum0 = esum0 + exp1
             ssum0(1) = ssum0(1) + exp2 * real(ir(1)*ir(1))
             ssum0(2) = ssum0(2) + exp2 * real(ir(2)*ir(2))
             ssum0(3) = ssum0(3) + exp2 * real(ir(3)*ir(3))
             ssum0(4) = ssum0(4) + exp2 * real(ir(1)*ir(2))
             ssum0(5) = ssum0(5) + exp2 * real(ir(2)*ir(3))
             ssum0(6) = ssum0(6) + exp2 * real(ir(3)*ir(1))
           endif
         endif
 40    continue
       esum0 = esum0 - sepi
 
!c     start loop over atoms in cell
 
       do 52 i=1,natot
 
!c       term with a=b
 
         esumr = esumr + zz(i)*zz(i)*esum0
         do 42 j=1,6
           ssumr(j) = ssumr(j) + zz(i)*zz(i) * ssum0(j)
 42      continue
         im = i-1
         if (im .ne. 0) then
 
!c         terms with a#b
 
           do 50 j=1,im
 
!c           loop over lattice points
!c                                   
!c           atoms not in the unit cell are sent back
 
             r1cc = rc(1,i) - rc(1,j)    
             r2cc = rc(2,i) - rc(2,j)
             r3cc = rc(3,i) - rc(3,j)
             ir1cc = int(r1cc)
             if(r1cc .lt. zero)  ir1cc = ir1cc - 1
             ir2cc = int(r2cc)
             if(r2cc .lt. zero) ir2cc = ir2cc - 1
             ir3cc = int(r3cc)
             if(r3cc .lt. zero) ir3cc = ir3cc - 1
             r1cc = r1cc - real(ir1cc)
             r2cc = r2cc - real(ir2cc)
             r3cc = r3cc - real(ir3cc)
 
             esub = zero
             do 43 k=1,3
               fsub(k) = zero
               ssub(k+3) = zero
               ssub(k) = zero
 43          continue
             do 46 ii=1,im2
             ir(1) = ii-imx-1
             do 46 jj=1,jm2
             ir(2) = jj-jmx-1
             do 46 kk=1,km2
             ir(3) = kk-kmx-1
               rp(1) = real(ir(1)) + r1cc
               rp(2) = real(ir(2)) + r2cc
               rp(3) = real(ir(3)) + r3cc
               rmod = zero
               do 44 l=1,3
               do 44 m=1,3
                 rmod = rmod + rp(l)*adot(l,m)*rp(m)
 44            continue
               rmod = sqrt(rmod)
               arg = seps*rmod
               if (arg .lt. 25.0) then
                 exp1 = boost_erfc(arg) / rmod
                 exp2 = (exp1 + sepi*exp(-arg*arg))/(rmod*rmod)
                 esub = esub + exp1
                 fsub(1) = fsub(1) + rp(1) * exp2
                 fsub(2) = fsub(2) + rp(2) * exp2
                 fsub(3) = fsub(3) + rp(3) * exp2
                 ssub(1) = ssub(1) + rp(1) * exp2 * rp(1)
                 ssub(2) = ssub(2) + rp(2) * exp2 * rp(2)
                 ssub(3) = ssub(3) + rp(3) * exp2 * rp(3)
                 ssub(4) = ssub(4) + rp(1) * exp2 * rp(2)
                 ssub(5) = ssub(5) + rp(2) * exp2 * rp(3)
                 ssub(6) = ssub(6) + rp(3) * exp2 * rp(1)
               endif
 46          continue
             esumr = esumr + dois*zz(i)*zz(j)*esub
             do 48 k=1,6
               ssumr(k) = ssumr(k) + dois*zz(i)*zz(j)*ssub(k)
 48          continue
             do 49 k=1,3
               fsumr(k,i) = fsumr(k,i) + dois*zz(i)*zz(j)*fsub(k)
               fsumr(k,j) = fsumr(k,j) - dois*zz(i)*zz(j)*fsub(k)
 49          continue
 50        continue
         endif
 52    continue
 
!c     end r sum
 
       eewald = esumg + esumr
 
!c     force
!c     note - returned force in units of lattice vectors (fewa)
!c            and cartesian coordinates (fewac)
!c            printed force in cartesian coordinates
 
       do 62 i=1,natot
         do 60 k=1,3
           fewa(k,i) = fsumr(k,i)
           do 58 l=1,3
             fewa(k,i) = fewa(k,i) + bdot(k,l)*fsumg(l,i)/twopi
 58        continue
 60      continue
 62    continue
 
!c     stress
!c     note - both returned and printed stress are
!c            in cartesian coordinates
 
       do 70 i=1,6
         j = i
         k = i
         if (i .gt. 3) j = i - 3
         if (i .gt. 3) k = j + 1
         if (k .gt. 3) k = 1
         stress(i) = &
           bvec(j,1)*ssumg(1)*bvec(k,1) + avec(j,1)*ssumr(1)*avec(k,1) &
         + bvec(j,2)*ssumg(2)*bvec(k,2) + avec(j,2)*ssumr(2)*avec(k,2) &
         + bvec(j,3)*ssumg(3)*bvec(k,3) + avec(j,3)*ssumr(3)*avec(k,3) &
         + bvec(j,1)*ssumg(4)*bvec(k,2) + avec(j,1)*ssumr(4)*avec(k,2) &
         + bvec(j,2)*ssumg(5)*bvec(k,3) + avec(j,2)*ssumr(5)*avec(k,3) &
         + bvec(j,3)*ssumg(6)*bvec(k,1) + avec(j,3)*ssumr(6)*avec(k,1) &
         + bvec(k,1)*ssumg(4)*bvec(j,2) + avec(k,1)*ssumr(4)*avec(j,2) &
         + bvec(k,2)*ssumg(5)*bvec(j,3) + avec(k,2)*ssumr(5)*avec(j,3) &
         + bvec(k,3)*ssumg(6)*bvec(j,1) + avec(k,3)*ssumr(6)*avec(j,1)
         if (i .le. 3) then
           stress(i) = stress(i) + esumg 
         endif
 70    continue
!c          
!c     forces in cartesian coordinates
 
       do 82 i=1,natot
         do 80 k=1,3
           fewac(k,i) = avec(k,1)*fewa(1,i) &
                    + avec(k,2)*fewa(2,i) &
                    + avec(k,3)*fewa(3,i)
 80      continue
 82    continue
!      
!c     printout
       enorm = eewald*vcell**(um/tres)
!c      write(6,100) enorm
       if(ipr .ne. 0) then 
         write(6,101) eewald,enorm 
          do 84 n=1,natot
            write(6,102) n,(rc(k,n),k=1,3),(fewa(k,n),k=1,3), (fewac(k,n),k=1,3)
 84       continue
 
         write(6,103) (stress(i),i=1,6)
         write(6,103) (stress(i)/vcell,i=1,6)
       endif
! change stress
       return
 100   format(/,26h normalized ewald energy =,f16.10)
 101   format(/,' ewald analysis',/,1x,14('*'),/,50x,'1/3',/, &
       ' energy :',12x,'energy (ry)',16x, &
       '*v    (ry*a.u.)',/,16x,f16.10,14x,f16.9,// &
       ' forces :',3x,'n',11x,'coord',19x,'force (ry/a.u.)',/ &
       19x,'a1',4x,'a2',4x,'a3',11x,'-x-',7x,'-y-',7x,'-z-')
 102   format(10x,i3,3x,3f6.3,5x,3f14.6,5x,3f14.6)
 103   format(/,' stress :',25x,'sigma * v (ry)',/, &
       14x,'-xx-',6x,'-yy-',6x,'-zz-',6x,'-xy-',6x,'-yz-',6x,'-zx-',/, &
       9x,6f10.6)

end subroutine ewaldf
