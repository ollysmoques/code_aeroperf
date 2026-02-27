from Aircraft_data import get_default_inputs, FlightConditions
from Cdmin import compute_parasite_drag
from helpers import get_cdi_wing, get_wing_cl_cd_from_aoa
from induced_equilibrium import induced_drag
from atmosphere import *


def drag_total(h_ft, dT_isa, V_kts, weight_lbf, cg_mac_current, thrust, config,
               alpha=4, return_breakdown=False):
    """
    Computes total drag of aircraft depending on certain conditions.

    If on ground (take_off/landing):
      - angle of attack (geometric) must be provided
      - no stabilizing tail lift taken into account (tail induced set to 0)

    If in flight (cruise):
      - equilibrium condition assumed
      - wing AoA deduced from equilibrium

    If return_breakdown=True:
      returns an additional dict with drag breakdown per component for pie charts.
    """
    fc_default, geom, aero = get_default_inputs()

    if config == 'take_off':
        on_ground = True
        flap_defl_deg = 15
    elif config == 'landing':
        on_ground = True
        flap_defl_deg = 30
    else:
        assert config == 'cruise', "ERROR: config must be 'take_off', 'landing' or 'cruise'"
        on_ground = False
        flap_defl_deg = 0

    if hasattr(geom, "flaps") and geom.flaps is not None:
        geom.flaps.delta_f_deg = flap_defl_deg

    atm = atmosphere(h=h_ft, variable=dT_isa, mode="delta_isa")

    speed = vitesses(
        atmosphere=atm,
        vitesse=V_kts,
        type="vraie",
        masse=weight_lbf,
        Sref=geom.S_ref,
        MAC=geom.wing.c_root,
    )

    Mach  = speed["mach"]
    V_ft  = speed["vitesses avion ft"]["vitesse vraie"]  # ft/s
    q_psf = speed["pression dynamique"]                  # psf (lb/ft^2)

    rho   = atm["densite"]        # slug/ft^3
    T_K   = atm["temperature_K"]  # K
    T_R   = T_K * 9.0 / 5.0       # Rankine

    fc = FlightConditions(rho=rho, V=V_ft, T_R=T_R, M_inf=Mach)

    parasite_components, Cdf_parasite, d_parasite = compute_parasite_drag(
        fc=fc, geom=geom, aero=aero
    )

    if on_ground:
        cdi_wing = get_cdi_wing(alpha, flap_defl_deg)
        di_wing = cdi_wing * q_psf * geom.S_ref  # not used in total (wing polar already has induced)
        di_emp = 0.0
        cdi_emp = 0.0
        cl_tot, cd_wing = get_wing_cl_cd_from_aoa(alpha, flap_defl_deg)
    else:
        cdi_wing, di_wing, cdi_emp, di_emp, wing_aoa = induced_drag(
            flaps=flap_defl_deg,
            thrust=thrust,
            z_eng=geom.wing.z_eng,
            z_cg=geom.wing.z_cg,
            weight=weight_lbf,
            x_cg=cg_mac_current,
            mac=aero.MAC,
            l_t=aero.lt,
            q=q_psf,
            sref=geom.S_ref,
            ar_emp=aero.AR_emp,
            h_v_stab=aero.h_v_stab,
            b_h_stab=aero.b_h_stab
        )
        cl_tot = weight_lbf / (q_psf * geom.S_ref)
        print('============= FLIGHT angle of attack ==============')
        print("angle d'attaque de l'aile:   ", wing_aoa, " degrees")

        cl_wing, cd_wing = get_wing_cl_cd_from_aoa(wing_aoa, flap_defl_deg)
        alpha = wing_aoa

    # Totals
    cdi_total = cdi_wing + cdi_emp
    d_induced_total = di_wing + di_emp

    # Wing drag from polar (includes induced for wing, per your statement)
    d_wing = cd_wing * q_psf * geom.S_ref

    # Total drag model used
    d_total = d_parasite + d_wing

    # ---------------------- PRINTS (unchanged) ----------------------
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

    # ---------------------- BREAKDOWN ----------------------
    breakdown = None
    if return_breakdown:
        # Names that mean "wing" inside compute_parasite_drag (exclude them to avoid double count)
        WING_NAMES = {"wing", "aile", "main wing", "wing+flaps", "aile+volets"}

        pie_components = {}
        parasite_by_component = {}

        for c in parasite_components:
            name = str(c.name)
            parasite_by_component[name] = float(c.D_total)

            # exclude wing-like names from pie parasite, since wing comes from polar
            if name.strip().lower() in WING_NAMES:
                continue
            pie_components[name] = float(c.D_total)

        # Add wing from polar (includes induced)
        pie_components["Wing (polar total)"] = float(d_wing)

        # Optional: if you want the tail induced as its own slice (cruise only)
        if abs(di_emp) > 1e-9:
            pie_components["Tail induced"] = float(di_emp)

        breakdown = {
            "config": config,
            "flaps_deg": float(flap_defl_deg),
            "alpha_deg": float(alpha),
            "q_psf": float(q_psf),
            "CL_total": float(cl_tot),

            "D_total": float(d_total),
            "D_parasite_total": float(d_parasite),
            "D_wing_polar_total": float(d_wing),

            "parasite_components": parasite_by_component,  # raw parasite as computed
            "pie_components": pie_components,              # ready for pie charts
            "pie_total": float(sum(pie_components.values())),

            # Keeping these for debugging/consistency checks:
            "induced_tail": float(di_emp),
            "induced_wing_from_equil": float(di_wing),
        }

    if return_breakdown:
        return cl_tot, d_total, q_psf, alpha, breakdown
    else:
        return cl_tot, d_total, q_psf, alpha