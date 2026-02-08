import numpy as np
import matplotlib.pyplot as plt

def get_cl_0(flaps, folder_name = 'wing_polar'):
    
    '''
    Uses linear interpolation to find CL_0 value
    
    :param flaps: angle of flaps deflection
    :type flaps: int
    :param file: folder where all the polars are found
    :type file: str
    '''

    assert flaps in [0,15,30], 'ERROR: flaps position 0, 15 or 30'

    cl_filename = folder_name + f'/CL_f{flaps}.npy'
    alpha_filename = folder_name + f'/alphas_f{flaps}.npy'

    try:
        cl_vect = np.load(cl_filename)
        alpha_vect = np.load(alpha_filename)

    except FileNotFoundError as e:
        print(f"Error: {e}, fournir le bon dossier et s'assurer que polaire de forme CL_f30.npy exemple")
    
    cl_o_value = np.interp(0,alpha_vect,cl_vect)

    return cl_o_value


def get_cl_alpha(flaps, folder_name = 'wing_polar'):

    '''
    Computes Cl alpha value for givent polar, gives averaged value for first 15 values 
    of cl as to reject non linear phases
    
    :param flaps: angle of flaps deflection
    :type flaps: int
    :param file: folder where all the polars are found
    :type file: str
    '''

    assert flaps in [0,15,30], 'ERROR: flaps position 0, 15 or 30'

    cl_filename = folder_name + f'/CL_f{flaps}.npy'
    alpha_filename = folder_name + f'/alphas_f{flaps}.npy'

    try:
        cl_vect = np.load(cl_filename)
        alpha_vect = np.load(alpha_filename)

    except FileNotFoundError as e:
        print(f"Error: {e}, fournir le bon dossier et s'assurer que polaire de forme CL_f30.npy exemple")

    cl_alpha_vect = cl_vect/alpha_vect
    cl_alpha_value = np.mean(cl_alpha_vect)
    
    return cl_alpha_value

def get_wing_cl_cd_from_aoa(aoa, flaps, folder_name = 'wing_polar'):
    
    '''
    Gets a CL value for wing polar when given a CL value, uses linear interpolation to find the 
    best fit
    
    :param aoa: angle of attack of wing
    :type aoa: float
    :param flaps: angle of flaps deflection
    :type flaps: int
    :param file: folder where all the polars are found
    :type file: str

    returns CL, CD (total) wing values
    '''

    assert flaps in [0,15,30], 'ERROR: flaps position 0, 15 or 30'

    cl_filename = folder_name + f'/CL_f{flaps}.npy'
    cd_filename = folder_name + f'/CD_f{flaps}.npy'
    alpha_filename = folder_name + f'/alphas_f{flaps}.npy'

    try:
        cl_vect = np.load(cl_filename)
        cd_vect = np.load(cd_filename)
        alpha_vect = np.load(alpha_filename)

    except FileNotFoundError as e:
        print(f"Error: {e}, fournir le bon dossier et s'assurer que polaire de forme CL_f30.npy exemple")
    
    cl_value = np.interp(aoa,alpha_vect,cl_vect)
    cd_value = np.interp(aoa,alpha_vect,cd_vect)

    return cl_value, cd_value

def get_cdi_wing(aoa, flaps, folder_name = 'wing_polar'):

    '''
    returns Cdi of wing based on given aoa and polar files
    
    :param aoa: angle of attack of wing
    :type aoa: float
    :param flaps: angle of flaps deflection
    :type flaps: int
    :param file: folder where all the polars are found
    :type file: str
    '''

    assert flaps in [0,15,30], 'ERROR: flaps position 0, 15 or 30'

    cdi_filename = folder_name + f'/CD_f{flaps}.npy'
    alpha_filename = folder_name + f'/alphas_f{flaps}.npy'

    try:
        cdi_vect = np.load(cdi_filename)
        alpha_vect = np.load(alpha_filename)

    except FileNotFoundError as e:
        print(f"Error: {e}, fournir le bon dossier et s'assurer que polaire de forme CL_f30.npy exemple")
    
    cdi_value = np.interp(aoa,alpha_vect,cdi_vect)

    return cdi_value

def get_alpha_from_cl(cl_value, flaps, folder_name = 'wing_polar'):
    
    '''
    Computes a wing angle of attack from a given wing CL
    
    :param cl_value: value of cl 
    :param flaps: angle of flaps deflection (0,15,30)
    :type flaps: int
    :param file: folder where all the polars are found
    :type file: str

    returns: angle of attack [deg]
    '''

    assert flaps in [0,15,30], 'ERROR: flaps position 0, 15 or 30'

    cl_filename = folder_name + f'/CL_f{flaps}.npy'
    alpha_filename = folder_name + f'/alphas_f{flaps}.npy'

    try:
        cl_vect = np.load(cl_filename)
        alpha_vect = np.load(alpha_filename)

    except FileNotFoundError as e:
        print(f"Error: {e}, fournir le bon dossier et s'assurer que polaire de forme CL_f30.npy exemple")
    
    alpha_value = np.interp(cl_value,cl_vect,alpha_vect)

    return alpha_value

def get_empennage_lift(flaps, thrust, z_eng, z_cg, weight, x_cg, mac, l_t, q, sref):

    '''
    Calculates empennage equilibrium lift with sum of moments and forces. Solve 2 dimension problem
    for lift of wing and lift of empennage. Takes into account positionning of engine and stablizing
    effect on plane. Assumes x_cp constant at 0.25% of Mean Aero Chord.
    
    :param flaps: angle of flaps deflection (0,15,30)
    :type flaps: int
    :param thrust: engine thrust [lbf]
    :param z_eng: height of the engine [ft]
    :param z_cg: height of the center of gravity [ft]
    :param weight: weight of the aircraft [lbf]
    :param x_cg: position of center of gravity [%MAC]
    :param mac: mean aero chord [ft]
    :param l_t: tail arm [ft]
    :param q: dynamic pressure [psi]
    :param sref: reference surface (wing) [ft^2]

    Returns wing lift horizontal stab lift and wing angle of attack which directly relates 
    to the angle of attack of the whole plane
    '''

    assert flaps in [0,15,30], 'ERROR: flaps position 0, 15 or 30'

    x_cp = 0.25
    B = np.array([thrust*(z_eng - z_cg), weight])
    A = np.array([-(x_cp - x_cg)*mac, -l_t],
                 [1, 1])
    
    C = np.linalg.solve(A,B)

    wing_lift = C[0]
    emp_lift = C[1]

    cl_value = wing_lift/(q*sref)
    wing_aoa = get_alpha_from_cl(cl_value,flaps)

    return wing_lift, emp_lift, wing_aoa
