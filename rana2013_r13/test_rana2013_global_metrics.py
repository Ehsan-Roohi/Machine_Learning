#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np

from analyze_rana2013_case import global_metrics


def main() -> None:
    x=np.linspace(0.0,1.0,5)
    y=np.linspace(0.0,1.0,4)
    mach=0.2
    gamma=5.0/3.0
    speed=math.sqrt(gamma)*mach

    # Only the y=max row is allowed to contribute to moving-wall drag.
    sigma=np.full((y.size,x.size),1234.0)
    sigma[-1,:]=-2.0

    # u=x+2y makes the exactly interpolated x=0.5 centerline analytic.
    u=np.asarray([[X+2.0*Y for X in x] for Y in y])
    got=global_metrics(sigma,u,x,y,mach,speed,gamma)

    const2=gamma*mach**2
    literal=-2.0*const2
    reduced=literal/(math.sqrt(2.0)*speed)
    expected_G=float(np.trapezoid(np.abs(0.5+2.0*y),y))

    assert math.isclose(got['D_sigma_over_p0_signed'],literal,rel_tol=0,abs_tol=1e-14),got
    assert math.isclose(got['D_signed'],reduced,rel_tol=0,abs_tol=1e-14),got
    assert math.isclose(got['D_abs'],abs(reduced),rel_tol=0,abs_tol=1e-14),got
    assert math.isclose(got['G'],expected_G,rel_tol=0,abs_tol=1e-14),got

    # Sentinel: changing a non-top row must not change D.
    sigma2=sigma.copy(); sigma2[0,:]=-9.9e8
    got2=global_metrics(sigma2,u,x,y,mach,speed,gamma)
    assert got2['D_signed']==got['D_signed'],(got,got2)

    # Reject inconsistent Mach/speed-ratio metadata rather than silently fitting D.
    try:
        global_metrics(sigma,u,x,y,mach,speed*1.01,gamma)
    except ValueError as exc:
        assert 'inconsistent nondimensionalization' in str(exc)
    else:
        raise AssertionError('inconsistent metadata was accepted')

    print('RANA_REDUCED_D_ORACLE_STATIC_TESTS_PASS')


if __name__=='__main__':
    main()
