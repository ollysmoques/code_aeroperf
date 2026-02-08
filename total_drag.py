from Aircraft_data import get_default_inputs, FlightConditions
from Cdmin import compute_parasite_drag
from helpers import get_cdi_wing, get_wing_cl_cd_from_aoa
from induced_equilibrium import induced_drag
from atmosphere import *


def drag_total(h_ft, dT_isa, V_kts, weight_lbf, cg_mac_current, thrust, config, alpha = 4):

    '''
    Computes total drag of aircraft depending on certain conditions. If on ground, 
    angle of attack (geometric) must be provided and no stabilizing lift will be
    taken into account. If in flight, equilibrium condition is assumed and therefore, 
    wing angle of attack is deduced from equilibrium. 
    
    :param h_ft: altitude based on pressure [ft]
    :param dT_isa: standard atmosphere temperature deviation [K, C]
    :param V_kts: True air speed [kts]
    :param weight_lbf: weight of aircraft [lbf]
    :param cg_mac_current: position of cg along wing [%MAC]
    :param thrust: thrust of aircraft [lbf]
    :param config: configuration 'take_off' , 'landing', 'cruise' if take_off => on_ground
    :type config: str
    :param alpha: angle of attack if in take_off mode [degree, 4 degrees]
    '''
    fc_default, geom, aero = get_default_inputs()
    
    if config == 'take_off':
        on_ground = True
        flap_defl_deg = 15
    elif config == 'landing':
        flap_defl_deg = 30
    else:
        assert config == 'cruise', "ERROR: config must be 'take_off', 'landing' or 'cruise'" 
        flap_defl_deg = 0  

    if hasattr(geom, "flaps") and geom.flaps is not None:
        geom.flaps.delta_f_deg = flap_defl_deg

    atm = atmosphere(
        h=h_ft,
        variable=dT_isa,
        mode="delta_isa",
    )

    speed = vitesses(
        atmosphere=atm,
        vitesse=V_kts,
        type="vraie",
        masse=weight_lbf,   
        Sref=geom.S_ref,
        MAC=geom.wing.c_root,
    )


    Mach  = speed["mach"]
    V_ft  = speed["vitesses avion ft"]["vitesse vraie"]   # ft/s
    q_psf = speed["pression dynamique"]          # déjà en lb/ft^2 (psf)

    rho   = atm["densite"]                        # slug/ft^3
    T_K   = atm["temperature_K"]                  # K
    T_R   = T_K * 9.0 / 5.0                       # Rankine
    
    fc = FlightConditions(
        rho=rho,
        V=V_ft,
        T_R=T_R,
        M_inf=Mach,
    )

    parasite_components, Cdf_parasite, d_parasite = compute_parasite_drag(
        fc=fc, geom=geom, aero=aero
    )

    if on_ground:
        #not necessarily at equilibrium, therefore resulting to given alpha
        cdi_wing = get_cdi_wing(alpha, flap_defl_deg)
        di_wing = cdi_wing * q_psf * geom.S_ref
        di_emp = 0
        cdi_emp = 0
        cl_tot, cd_wing = get_wing_cl_cd_from_aoa(alpha, flap_defl_deg)
        
    else:
        #in flight then necessarily at equilibirum
        cdi_wing, di_wing, cdi_emp, di_emp, wing_aoa = induced_drag(
            flaps= flap_defl_deg,
            thrust = thrust,
            z_eng = geom.z_eng,
            z_cg = geom.z_cg,
            weight = weight_lbf,
            x_cg= cg_mac_current,
            mac= aero.MAC,
            q= q_psf,
            sref = geom.S_ref,
            ar_emp = aero.AR_emp,
            h_v_stab   = aero.h_v_stab,
            b_h_stab   = aero.b_h_stab)
        cl_tot = weight_lbf/(q_psf * geom.S_ref)
        cl_wing, cd_wing = get_wing_cl_cd_from_aoa(wing_aoa, flap_defl_deg)

    cdi_total = cdi_wing + cdi_emp
    d_induced_total = di_wing + di_emp
    d_wing = cd_wing * q_psf * geom.S_ref
    d_total = d_parasite + d_wing
    
    print("=== PARASITE DRAG (friction + form + IF + gears + Krud) ===")
    print(f"{'Component':<18}  {'Cdf_tot':>10}  {'D_tot [lbf]':>12}")
    print("-" * 50)
    for c in parasite_components:
          print(f"{c.name:<18}  {c.Cdf_total:10.5e}  {c.D_total:12.4f}")
    print("-" * 50)
    print(f"{'TOTAL PARASITE':<18}  {Cdf_parasite:10.5e}  {d_parasite:12.4f}")

    print("\n=== INDUCED DRAG (wing + tail) ===")
    print(f"{'Surface':<18}  {'Cdi':>10}  {'Di [lbf]':>12}")
    print("-" * 50)
    print(f"{'Wing':<18}  {cdi_wing:10.5e}  {di_wing:12.4f}")
    print(f"{'Tail':<18}  {cdi_emp:10.5e}  {di_emp:12.4f}")
    print("-" * 50)
    print(f"{'TOTAL INDUCED':<18}  {cdi_total:10.5e}  {d_induced_total:12.4f}")

    print("\n=== TOTAL DRAG ===")
    print(f"{'C_D_parasite':<16} = {Cdf_parasite:.5e}")
    print(f"{'C_D_induced':<16} = {cdi_total:.5e}")
    print(f"{'C_D_total':<16} = {Cdf_parasite + cdi_total:.5e}")
    print(f"{'D_total [lbf]':<16} = {d_total:.4f}")

    
    print(f"{'CL_total':<16} = {cl_tot:.5f}")

    return cl_tot, d_total, q_psf





