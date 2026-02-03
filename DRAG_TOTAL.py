from Aircraft_data import get_default_inputs, FlightConditions
from Cdmin import compute_parasite_drag
from induced_drag import compute_induced_drag
from atmosphere import *
from cg_shift import *


def drag_total(h_ft, dT_isa, V_kts, weight_lbf, cg_mac_current, mode, config, alpha = 4):
    # ============================
    # 1) Géométrie et paramètres aero
    # ============================
    fc_default, geom, aero = get_default_inputs()
    
    # --- 1.a Choix CL_max et polaire selon la config ---
    if config == 'TO':
        CL_max     = aero.CL_max_TO
        polar_file = 'flaps_15_polar.polar'
        flap_defl_deg = 15.0   # <<--- déflexion volets pour décollage
    elif config == 'LANDING':
        CL_max = aero.CL_max_LANDING
        flap_defl_deg = 30.0 # <<--- flaps landing
    else:  # 'clean'
        CL_max     = aero.CL_max_clean
        polar_file = 'horizontal_only.polar'
        flap_defl_deg = 0.0    # <<--- volets rentrés

    # --- 1.b Imposer la déflexion de volets dans la géométrie ---
    # (utilisée par compute_flap_drag_increment via geom.flaps.delta_f_deg)
    if hasattr(geom, "flaps") and geom.flaps is not None:
        geom.flaps.delta_f_deg = flap_defl_deg

    
        
        
    # ============================
    # 2) Atmosphère & vitesses
    # ============================
    atm = atmosphere(
        h=h_ft,
        variable=dT_isa,
        mode="delta_isa",
    )

    speed = vitesses(
        atmosphere=atm,
        vitesse=V_kts,
        type="vraie",
        masse=weight_lbf,      # tu utilises déjà le poids en lbf dans CL
        Sref=geom.S_ref,
        MAC=geom.wing.c_root,
    )

    # Récup des grandeurs utiles

    Mach  = speed["mach"]
    V_ft  = speed["vitesses avion ft"]["vitesse vraie"]   # ft/s
    q_psf = speed["pression dynamique"]          # déjà en lb/ft^2 (psf)
    Cl_tot = get_cl_value_helper(polar_file,mode, speed, alpha)

    rho   = atm["densite"]                        # slug/ft^3
    T_K   = atm["temperature_K"]                  # K
    T_R   = T_K * 9.0 / 5.0                       # Rankine
    
    # ============================
    # 3) FlightConditions cohérentes
    # ============================
    fc = FlightConditions(
        rho=rho,
        V=V_ft,
        T_R=T_R,
        M_inf=Mach,
    )

    # ============================
    # 4) Traînée parasite (Cdmin)
    # ============================
    parasite_components, Cdf_parasite, D_parasite = compute_parasite_drag(
        fc=fc, geom=geom, aero=aero
    )

    # ============================
    # 5) Traînée induite (wing + tail)
    # ============================


    Cdi_wing, Di_wing, Cdi_emp, Di_emp = compute_induced_drag(
    Cl_tot     = Cl_tot,
    Cm_emp     = aero.Cm_emp,
    cg_mac     = cg_mac_current,
    cp_mac     = aero.cp_mac,
    lt         = aero.lt,
    MAC        = aero.MAC,
    AR_wing    = aero.AR_wing,
    AR_emp     = aero.AR_emp,
    h_v_stab   = aero.h_v_stab,
    b_h_stab   = aero.b_h_stab,
    fc         = fc,
    geom       = geom,
    aero       = aero,
    )

    Cdi_total = Cdi_wing + Cdi_emp
    D_induced_total = Di_wing + Di_emp
    D_total = D_parasite + D_induced_total

     # ============================
     # 6) Tableau : traînée parasite
     # ============================
    print("=== PARASITE DRAG (friction + form + IF + gears + Krud) ===")
    print(f"{'Component':<18}  {'Cdf_tot':>10}  {'D_tot [lbf]':>12}")
    print("-" * 50)
    for c in parasite_components:
          print(f"{c.name:<18}  {c.Cdf_total:10.5e}  {c.D_total:12.4f}")
    print("-" * 50)
    print(f"{'TOTAL PARASITE':<18}  {Cdf_parasite:10.5e}  {D_parasite:12.4f}")

      # ============================
      # 7) Tableau : traînée induite
      # ============================
    print("\n=== INDUCED DRAG (wing + tail) ===")
    print(f"{'Surface':<18}  {'Cdi':>10}  {'Di [lbf]':>12}")
    print("-" * 50)
    print(f"{'Wing':<18}  {Cdi_wing:10.5e}  {Di_wing:12.4f}")
    print(f"{'Tail':<18}  {Cdi_emp:10.5e}  {Di_emp:12.4f}")
    print("-" * 50)
    print(f"{'TOTAL INDUCED':<18}  {Cdi_total:10.5e}  {D_induced_total:12.4f}")

    # ============================
      # 8) Total global
      # ============================
    
    print("\n=== TOTAL DRAG ===")
    print(f"{'C_D_parasite':<16} = {Cdf_parasite:.5e}")
    print(f"{'C_D_induced':<16} = {Cdi_total:.5e}")
    print(f"{'C_D_total':<16} = {Cdf_parasite + Cdi_total:.5e}")
    print(f"{'D_total [lbf]':<16} = {D_total:.4f}")

    
    print(f"{'CL_total':<16} = {Cl_tot:.5f}")

    return Cl_tot, D_total, q_psf





