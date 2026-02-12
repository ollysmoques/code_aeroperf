import numpy as np
import matplotlib.pyplot as plt

from Aircraft_data import get_default_inputs, FlightConditions
from Cdmin import compute_parasite_drag
from induced_equilibrium import induced_drag
from helpers import get_wing_cl_cd_from_aoa
from atmosphere import *


# ---------------------- PIE PLOT ---------------------- #
def plot_single_pie(title, components_dict):
    """
    components_dict: dict {name: {"D_lbf":..., "CD":...}, ...}
    """
    labels = [k for k, v in components_dict.items() if abs(v["D_lbf"]) > 1e-9]
    values = np.array([components_dict[k]["D_lbf"] for k in labels], dtype=float)

    total = values.sum()
    pct = values / total * 100.0

    explode = [0.10 if "Wing" in l else 0.03 for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(values, startangle=90, explode=explode, labels=None)

    # Pourcentages à distance variable (lisible)
    for i, w in enumerate(wedges):
        ang = 0.5 * (w.theta1 + w.theta2)
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))

        if pct[i] > 20:
            r = 0.70
        elif pct[i] > 5:
            r = 0.90
        else:
            r = 1.05

        ax.text(
            r * x,
            r * y,
            f"{pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=15)

    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=11,
        title="Components",
        title_fontsize=12,
    )

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# ---------------------- ANALYSIS CORE ---------------------- #
def run_case_equilibrium(h_ft, dT_isa, V_kts, weight_lbf, cg_mac_current, thrust, flap_defl_deg):
    """
    Returns a results dict with:
      - alpha_eq_deg, V_kts, h_ft, q_psf
      - D_total_lbf, CD_total
      - components: dict {component: {"D_lbf","CD"}}
    """
    _, geom, aero = get_default_inputs()
    Sref = geom.S_ref

    # Atmos + speed
    atm = atmosphere(h=h_ft, variable=dT_isa, mode="delta_isa")
    speed = vitesses(
        atmosphere=atm,
        vitesse=V_kts,
        type="vraie",
        masse=weight_lbf,
        Sref=Sref,
        MAC=geom.wing.c_root,
    )

    Mach  = speed["mach"]
    V_ft  = speed["vitesses avion ft"]["vitesse vraie"]
    q_psf = speed["pression dynamique"]

    rho = atm["densite"]
    T_R = atm["temperature_K"] * 9.0 / 5.0
    fc = FlightConditions(rho=rho, V=V_ft, T_R=T_R, M_inf=Mach)

    # Parasite drag
    parasite_components, Cdf_parasite, d_parasite = compute_parasite_drag(fc=fc, geom=geom, aero=aero)

    # Equilibrium with imposed flaps
    cdi_wing, di_wing, cdi_emp, di_emp, alpha_eq = induced_drag(
        flaps=flap_defl_deg,
        thrust=thrust,
        z_eng=geom.wing.z_eng,
        z_cg=geom.wing.z_cg,
        weight=weight_lbf,
        x_cg=cg_mac_current,
        mac=aero.MAC,
        l_t=aero.lt,
        q=q_psf,
        sref=Sref,
        ar_emp=aero.AR_emp,
        h_v_stab=aero.h_v_stab,
        b_h_stab=aero.b_h_stab,
    )

    # Wing CD from polar at alpha_eq (your polar includes induced for the wing)
    cl_w, cd_w = get_wing_cl_cd_from_aoa(alpha_eq, flap_defl_deg)
    d_wing = cd_w * q_psf * Sref

    # Total (same model as your drag_total: parasite + wing polar)
    d_total = d_parasite + d_wing
    cd_total = d_total / (q_psf * Sref)

    # Per-component D and CD
    WING_NAMES = {"wing", "aile", "main wing", "wing+flaps", "aile+volets"}
    components = {}

    # Parasite components (except wing to avoid double count)
    for c in parasite_components:
        name = str(c.name)
        if name.strip().lower() in WING_NAMES:
            continue
        D_i = float(c.D_total)
        components[name] = {"D_lbf": D_i, "CD": D_i / (q_psf * Sref)}

    # Wing from polar
    components["Wing (polar total)"] = {"D_lbf": float(d_wing), "CD": float(cd_w)}

    # Optional tail induced (info slice)
    if abs(di_emp) > 1e-9:
        components["Tail induced"] = {"D_lbf": float(di_emp), "CD": float(di_emp / (q_psf * Sref))}

    results = {
        "h_ft": float(h_ft),
        "V_kts": float(V_kts),
        "q_psf": float(q_psf),
        "alpha_eq_deg": float(alpha_eq),
        "flaps_deg": float(flap_defl_deg),
        "D_total_lbf": float(d_total),
        "CD_total": float(cd_total),
        "components": components,
    }
    return results


def print_summary(name, res):
    print(f"\n===== {name} =====")
    print(f"Alt [ft]      : {res['h_ft']:.0f}")
    print(f"V [kts]       : {res['V_kts']:.1f}")
    print(f"alpha_eq [deg]: {res['alpha_eq_deg']:.2f}")
    print(f"D_total [lbf] : {res['D_total_lbf']:.3f}")
    print(f"CD_total      : {res['CD_total']:.6f}")
    print("\nComponent breakdown (D and CD):")
    print(f"{'Component':<24} {'D [lbf]':>12} {'CD':>12}")
    print("-" * 52)
    for comp, vals in sorted(res["components"].items(), key=lambda kv: -kv[1]["D_lbf"]):
        print(f"{comp:<24} {vals['D_lbf']:12.4f} {vals['CD']:12.6f}")


# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    W = 400
    cg = 0.33
    thrust = 89
    dT_isa = 0.0

    res_to = run_case_equilibrium(
        h_ft=0, dT_isa=dT_isa, V_kts=75,
        weight_lbf=W, cg_mac_current=cg, thrust=thrust,
        flap_defl_deg=15
    )

    res_cr = run_case_equilibrium(
        h_ft=10000, dT_isa=dT_isa, V_kts=100,
        weight_lbf=W, cg_mac_current=cg, thrust=thrust,
        flap_defl_deg=0
    )

    res_ld = run_case_equilibrium(
        h_ft=0, dT_isa=dT_isa, V_kts=70,
        weight_lbf=W, cg_mac_current=cg, thrust=thrust,
        flap_defl_deg=30
    )

    # Print requested info
    print_summary("TAKE-OFF (equilibrium, flaps 15)", res_to)
    print_summary("CRUISE (equilibrium, flaps 0)", res_cr)
    print_summary("LANDING (equilibrium, flaps 30)", res_ld)

    # 3 figures séparées (pie charts)
    plot_single_pie("Take-off drag breakdown (equilibrium, flaps 15)", res_to["components"])
    plot_single_pie("Cruise drag breakdown (equilibrium, flaps 0)", res_cr["components"])
    plot_single_pie("Landing drag breakdown (equilibrium, flaps 30)", res_ld["components"])