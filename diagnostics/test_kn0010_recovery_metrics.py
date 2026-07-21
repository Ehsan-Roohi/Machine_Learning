#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np

from analyze_kn0010_recovery_variant import global_metrics
from patch_r13_kn0010_recovery_variants import VARIANTS


def main() -> None:
    x=np.linspace(0.0,1.0,5)
    y=np.linspace(0.0,1.0,4)
    mach=0.2
    gamma=5.0/3.0
    speed=math.sqrt(gamma)*mach

    # D must use only the top wall and Rana's reduced-stress normalization.
    sigma=np.full((y.size,x.size),1234.0)
    sigma[-1,:]=-2.0
    u=np.asarray([[X+2.0*Y for X in x] for Y in y])
    got=global_metrics(sigma,u,x,y,mach,speed,gamma)
    expected_sigma_over_p0=-2.0*gamma*mach**2
    expected_D=math.sqrt(2.0)*expected_sigma_over_p0/speed
    integrate=getattr(np,'trapezoid',np.trapz)
    expected_G=float(integrate(np.abs(0.5+2.0*y),y))

    assert math.isclose(got['D_sigma_over_p0_signed'],expected_sigma_over_p0,abs_tol=1e-14)
    assert math.isclose(got['D_reduced_signed'],expected_D,abs_tol=1e-14)
    assert math.isclose(got['D_reduced_abs'],abs(expected_D),abs_tol=1e-14)
    assert math.isclose(got['G'],expected_G,abs_tol=1e-14)

    sigma2=sigma.copy(); sigma2[0,:]=-9.9e8
    got2=global_metrics(sigma2,u,x,y,mach,speed,gamma)
    assert got2['D_reduced_signed']==got['D_reduced_signed']

    try:
        global_metrics(sigma,u,x,y,mach,speed*1.01,gamma)
    except ValueError as exc:
        assert 'inconsistent nondimensionalization' in str(exc)
    else:
        raise AssertionError('inconsistent Mach/speed metadata was accepted')

    cfg=VARIANTS['owner_nomass_extrap']
    assert cfg==dict(
        owner=True,mass=False,extrap=True,
        zero_top_corner=False,alternating=False,
    ),cfg
    print('KN0010_RECOVERY_STATIC_TESTS_PASS')


if __name__=='__main__':
    main()
