from helpers import get_empennage_lift, get_cdi_wing
import numpy as np

def induced_drag(flaps, thrust, z_eng, z_cg, weight, x_cg, mac, l_t, q, sref, ar_emp, h_v_stab, b_h_stab):

    '''
    Computes total induced drag of airplane (wing induced + equilibrium drag). Only for in flight 
    performance as forst assumes equiulibrium to determine wing lift and wing aoa. then fetches 
    wing induced drag from calculated wing angle of attack. 
    
    :param flaps: deflection of flaps (0,15,30)
    :type flaps: int
    :param thrust: engine thrust
    :param z_eng: height of engine [ft]
    :param z_cg: height of center of gravity [ft]
    :param weight: weight of the aircraft [lbf]
    :param x_cg: position of the center of gravity [%MAC]
    :param mac: mean aerodynamic chord [ft]
    :param l_t: tail arm [ft]
    :param q: dynamic pressure [psi]
    :param sref: reference surface (wing) [ft^2]
    :param ar_emp: aspect ratio of horizontal stabilizer [-]
    :param h_v_stab: 
    :param b_h_stab: span of horizontal stabilizer [ft]
    '''

    assert flaps in [0,15,30], 'ERROR: flaps position 0, 15 or 30'
    
    wing_lift, stab_lift, wing_aoa = get_empennage_lift(flaps, thrust, z_eng, z_cg, weight, x_cg, mac, l_t, q, sref)
    
    dar = 1.9* ar_emp * h_v_stab/b_h_stab 
    ar_emp += dar
    e_emp = 1.78*(1 - 0.045 * ar_emp**0.68) - 0.64
    cl_emp = stab_lift/ (q * sref)
    cdi_emp = cl_emp**2 / (e_emp * np.pi * ar_emp)

    cdi_wing = get_cdi_wing(wing_aoa, flaps)

    di_wing = cdi_wing * q *sref
    di_emp = cdi_emp * q *sref

    return cdi_wing, di_wing, cdi_emp, di_emp, wing_aoa


