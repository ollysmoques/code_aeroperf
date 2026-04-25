import numpy as np
import matplotlib.pyplot as plt
import os
import json 

from Aircraft_data import get_default_inputs, FlightConditions
from Cdmin import compute_parasite_drag
from induced_equilibrium import induced_drag
from helpers import get_wing_cl_cd_from_aoa
from atmosphere import *

def load_launcher_config():
    """
    Charge les paramètres écrits par le launcher.
    Si le fichier n'existe pas, retourne un dictionnaire vide.
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_launcher_config.json")

    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

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

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, _ = ax.pie(
        values,
        startangle=90,
        explode=explode,
        labels=None,
        wedgeprops=dict(edgecolor="white", linewidth=1.0)
    )

    min_dy = 0.08
    outside_threshold = 2.0
    outside_labels = {"left": [], "right": []}

    for i, w in enumerate(wedges):
        ang = 0.5 * (w.theta1 + w.theta2)
        ang_rad = np.deg2rad(ang)

        x = np.cos(ang_rad)
        y = np.sin(ang_rad)

        color = w.get_facecolor()

        # Grosses parts
        if pct[i] >= 8:
            r = 0.72
            ax.text(
                r * x,
                r * y,
                f"{pct[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="black"
            )

        # Parts moyennes
        elif pct[i] >= outside_threshold:
            r = 0.88
            ax.text(
                r * x,
                r * y,
                f"{pct[i]:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec=color,
                    lw=1.2,
                    alpha=0.9
                )
            )

        # Très petites parts
        else:
            side = "right" if x >= 0 else "left"
            outside_labels[side].append({
                "i": i,
                "x": x,
                "y": y,
                "pct": pct[i],
                "color": color
            })

    def spread_y_positions(items, min_dy):
        if not items:
            return []

        items = sorted(items, key=lambda item: item["y"])
        y_positions = [item["y"] * 1.08 for item in items]

        for j in range(1, len(y_positions)):
            if y_positions[j] - y_positions[j - 1] < min_dy:
                y_positions[j] = y_positions[j - 1] + min_dy

        y_min, y_max = -1.05, 1.05

        if y_positions[-1] > y_max:
            shift = y_positions[-1] - y_max
            y_positions = [y - shift for y in y_positions]

        if y_positions[0] < y_min:
            shift = y_min - y_positions[0]
            y_positions = [y + shift for y in y_positions]

        return y_positions

    # Annotations extérieures
    for side, items in outside_labels.items():
        items = sorted(items, key=lambda item: item["y"])
        y_positions = spread_y_positions(items, min_dy)

        for k, (item, y_text) in enumerate(zip(items, y_positions)):
            x = item["x"]
            y = item["y"]
            color = item["color"]

            x_text = 1.08 if side == "right" else -1.08
            ha = "left" if side == "right" else "right"

            # petite variation de courbure pour distinguer les lignes
            rad = 0.15 + 0.05 * (k % 3)

            ax.annotate(
                f"{item['pct']:.1f}%",
                xy=(1.01 * x, 1.01 * y),
                xytext=(x_text, y_text),
                ha=ha,
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec=color,
                    lw=1.2,
                    alpha=0.95
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    lw=1.8,
                    color=color,
                    shrinkA=0,
                    shrinkB=0,
                    connectionstyle=f"arc3,rad={rad if side == 'right' else -rad}"
                )
            )

    ax.set_title(title, fontsize=15)

    ax.legend(
        wedges,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
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
        type="calibree",
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

        # Valeurs provenant de la polaire de l'aile
        "CL_wing": float(cl_w),
        "CD_wing": float(cd_w),

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

    if "CL_wing" in res:
        print(f"CL_wing       : {res['CL_wing']:.6f}")

    if "CD_wing" in res:
        print(f"CD_wing       : {res['CD_wing']:.6f}")

    print(f"D_total [lbf] : {res['D_total_lbf']:.3f}")
    print(f"CD_total      : {res['CD_total']:.6f}")

    print("\nComponent breakdown (D and CD):")
    print(f"{'Component':<24} {'D [lbf]':>12} {'CD':>12}")
    print("-" * 52)

    for comp, vals in sorted(res["components"].items(), key=lambda kv: -kv[1]["D_lbf"]):
        print(f"{comp:<24} {vals['D_lbf']:12.4f} {vals['CD']:12.6f}")
# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    cfg = load_launcher_config()

    OEW = cfg.get("OEW", 218)
    fuel = cfg.get("FUEL_LOAD", 75.0)
    payload = cfg.get("PAYLOAD", 170)

    W = OEW + fuel + payload

    cg = 0.33
    thrust = 89

    dT_isa = cfg.get("dT_Isa", 0.0)

    h_airport = cfg.get("h_airport", 0)
    h_cruise = cfg.get("MISSION_HEIGHT_FT", 10000)

    V_cruise = cfg.get("V_cruise_CAS", 108)

    print("\n===== PARAMÈTRES LAUNCHER UTILISÉS =====")
    print(f"Poids total [lbf]       : {W:.1f}")
    print(f"Altitude aéroport [ft]  : {h_airport:.0f}")
    print(f"Altitude croisière [ft] : {h_cruise:.0f}")
    print(f"Delta ISA [°C]          : {dT_isa:.1f}")
    print(f"V croisière [kts CAS]   : {V_cruise:.1f}")

    res_to = run_case_equilibrium(
        h_ft=h_airport, dT_isa=dT_isa, V_kts=51.4,
        weight_lbf=W, cg_mac_current=cg, thrust=thrust,
        flap_defl_deg=15
    )

    res_cr = run_case_equilibrium(
        h_ft=h_cruise, dT_isa=dT_isa, V_kts=V_cruise,
        weight_lbf=W, cg_mac_current=cg, thrust=thrust,
        flap_defl_deg=0
    )

    res_ld = run_case_equilibrium(
        h_ft=h_airport, dT_isa=dT_isa, V_kts=75,
        weight_lbf=W, cg_mac_current=cg, thrust=thrust,
        flap_defl_deg=30
    )

    print_summary("TAKE-OFF ", res_to)
    print_summary("CRUISE ", res_cr)
    print_summary("LANDING ", res_ld)

    plot_single_pie("Take-off drag breakdown", res_to["components"])
    plot_single_pie("Cruise drag breakdown", res_cr["components"])
    plot_single_pie("Landing drag breakdown", res_ld["components"])