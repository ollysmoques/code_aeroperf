from dataclasses import dataclass

from skin_friction import (
    WingFrictionResult,
    FuselageFrictionResult,
    compute_wing_skin_friction,
    compute_horizontal_tail_skin_friction,
    compute_vertical_tail_skin_friction,
    compute_fuselage_skin_friction,
    compute_engine_skin_friction,
    compute_pylon_skin_friction,      
)
from Form_factor import (
    FormFactorInputs,
    FuselageFFInputs,
    compute_form_factor,
    compute_fuselage_form_factor,
    get_interference_factor,
    fineness_ratio_from_L_and_d,
    ff_engine_nacelle
)
from Aircraft_data import (
    FlightConditions,
    AircraftGeometry,
    AeroParams,
    get_default_inputs,
)
from Drag_Increment import (
    compute_main_gear_drag_increment,
    compute_tail_gear_drag_increment,
    compute_flap_drag_increment,
)



# ==============================
# 0) Crud factor global
# ==============================
CRUD_FACTOR = 1.25


# ==============================
# 1) Dataclass résultat global
# ==============================

@dataclass
class ComponentDrag:
    name: str

    Cf_eq: float
    FF: float
    IF_: float

    Cdf_base: float
    Cdf_total: float

    D_base: float
    D_total: float


# ==============================
# 2) Helper commun : combine Cf, FF, IF
# ==============================

def build_component_drag(
    name: str,
    Cf_eq: float,
    Cdf_base: float,
    D_base: float,
    FF: float,
    IF_: float,
) -> ComponentDrag:
    # Drag “aéro propre” avec FF & IF
    Cdf_clean = Cdf_base * FF * IF_
    D_clean   = D_base   * FF * IF_

    # Application du crud factor (+25 %)
    Cdf_total = Cdf_clean * CRUD_FACTOR
    D_total   = D_clean   * CRUD_FACTOR

    return ComponentDrag(
        name=name,
        Cf_eq=Cf_eq,
        FF=FF,
        IF_=IF_,
        Cdf_base=Cdf_base,
        Cdf_total=Cdf_total,
        D_base=D_base,
        D_total=D_total,
    )


# ==============================
# 3) Fonction principale de calcul
# ==============================

def compute_all_components(
    fc: FlightConditions | None = None,
    geom: AircraftGeometry | None = None,
    aero: AeroParams | None = None,
) -> list[ComponentDrag]:
    """
    Si aucun argument n'est passé, utilise les inputs par défaut.
    Tu peux aussi passer d'autres inputs (autre cas de vol, autre géométrie).
    """
    if fc is None or geom is None or aero is None:
        fc, geom, aero = get_default_inputs()

    # Fineness ratios via L/d
    f_fus    = fineness_ratio_from_L_and_d(geom.fus.length,    geom.fus.d_max)
    f_moteur = fineness_ratio_from_L_and_d(geom.nacelle.length, geom.nacelle.d_max)

    # Interference factors
    IF_wing     = get_interference_factor(aero.IF_wing_key)
    IF_tail     = get_interference_factor(aero.IF_tail_key)
    IF_fuselage = aero.IF_fuselage
    IF_moteur   = get_interference_factor(aero.IF_moteur_key)
    IF_pylon    = 1.0   # pour l’instant, pas de facteur d’interférence spécifique

    S_ref = geom.S_ref


    # --- EMPENNAGE HORIZONTAL ---
    res_ht: WingFrictionResult = compute_horizontal_tail_skin_friction(
        rho=fc.rho,
        V=fc.V,
        T=fc.T_R,
        M_inf=fc.M_inf,
        c_root=geom.ht.c_root,
        c_tip=geom.ht.c_tip,
        S_wet=geom.ht.S_wet,
        S_ref=S_ref,
        surface_type="smooth_paint",
        regime="subsonic",
        units="imperial",
        model="turbulent",
        xtr_root_upper=0.45,
        xtr_root_lower=0.45,
        xtr_tip_upper=0.60,
        xtr_tip_lower=0.50,
    )

    Cf_ht_eq    = res_ht.Cf_wing_avg
    Cdf_ht_base = res_ht.Cdf_wing
    D_ht_base   = res_ht.Df_wing

    FF_ht = compute_form_factor(
        FormFactorInputs(
            method="jenkinson_tail",
            t_over_c=aero.t_over_c_tail,
            sweep_c2_deg=aero.sweep_c2_tail_deg,
        )
    )

    # --- EMPENNAGE VERTICAL (haut + bas) ---
    res_vt_upper: WingFrictionResult = compute_vertical_tail_skin_friction(
        rho=fc.rho,
        V=fc.V,
        T=fc.T_R,
        M_inf=fc.M_inf,
        c_root=geom.vt.c_root,
        c_tip=geom.vt.c_tip,
        S_wet=geom.vt.S_wet_upper,
        S_ref=S_ref,
        surface_type="smooth_paint",
        regime="subsonic",
        units="imperial",
        model="turbulent",
        xtr_root_upper=0.45,
        xtr_root_lower=0.45,
        xtr_tip_upper=0.60,
        xtr_tip_lower=0.50,
    )

    res_vt_lower: WingFrictionResult = compute_vertical_tail_skin_friction(
        rho=fc.rho,
        V=fc.V,
        T=fc.T_R,
        M_inf=fc.M_inf,
        c_root=geom.vt.c_root,
        c_tip=geom.vt.c_tip,
        S_wet=geom.vt.S_wet_lower,
        S_ref=S_ref,
        surface_type="smooth_paint",
        regime="subsonic",
        units="imperial",
        model="turbulent",
        xtr_root_upper=0.45,
        xtr_root_lower=0.45,
        xtr_tip_upper=0.60,
        xtr_tip_lower=0.50,
    )

    S_wet_vt_tot = geom.vt.S_wet_upper + geom.vt.S_wet_lower

    Cf_vt_eq = (
        res_vt_upper.Cf_wing_avg * geom.vt.S_wet_upper
        + res_vt_lower.Cf_wing_avg * geom.vt.S_wet_lower
    ) / S_wet_vt_tot

    Cdf_vt_base = res_vt_upper.Cdf_wing + res_vt_lower.Cdf_wing
    D_vt_base   = res_vt_upper.Df_wing  + res_vt_lower.Df_wing

    FF_vt = FF_ht  # même famille de profils que le HT

    # --- FUSELAGE ---
    res_fuselage: FuselageFrictionResult = compute_fuselage_skin_friction(
        rho=fc.rho,
        V=fc.V,
        T=fc.T_R,
        M_inf=fc.M_inf,
        c_ref=geom.fus.length,
        S_wet=geom.fus.S_wet,
        S_ref=S_ref,
        surface_type="smooth_paint",
        regime="subsonic",
        units="imperial",
        model="turbulent",
    )

    Cf_fus_eq    = res_fuselage.Cf_avg
    Cdf_fus_base = res_fuselage.Cdf
    D_fus_base   = res_fuselage.Df

    _, FF_fus = compute_fuselage_form_factor(
        FuselageFFInputs(
            method="nicolai",
            f=f_fus,
        )
    )

    # --- MOTEUR / NACELLE ---
    res_moteur: FuselageFrictionResult = compute_engine_skin_friction(
        rho=fc.rho,
        V=fc.V,
        T=fc.T_R,
        M_inf=fc.M_inf,
        c_ref=geom.nacelle.length,
        S_wet=geom.nacelle.S_wet,
        S_ref=S_ref,
        surface_type="smooth_paint",
        regime="subsonic",
        units="imperial",
        model="turbulent",
    )

    Cf_moteur_eq    = res_moteur.Cf_avg
    Cdf_moteur_base = res_moteur.Cdf
    D_moteur_base   = res_moteur.Df

    FF_moteur = ff_engine_nacelle(f_moteur)

    # --- PYLON (NACA 0020) ---
    res_pylon: WingFrictionResult = compute_pylon_skin_friction(
        rho=fc.rho,
        V=fc.V,
        T=fc.T_R,
        M_inf=fc.M_inf,
        c_root=geom.pylon.c_root,
        c_tip=geom.pylon.c_tip,
        S_wet=geom.pylon.S_wet,
        S_ref=S_ref,
        surface_type="smooth_paint",
        regime="subsonic",
        units="imperial",
        model="turbulent",
        xtr_root_upper=0.45,
        xtr_root_lower=0.45,
        xtr_tip_upper=0.60,
        xtr_tip_lower=0.50,
    )

    Cf_pylon_eq    = res_pylon.Cf_wing_avg
    Cdf_pylon_base = res_pylon.Cdf_wing
    D_pylon_base   = res_pylon.Df_wing

    FF_pylon = compute_form_factor(
        FormFactorInputs(
            method="jenkinson_tail",   # profil d’empennage → bon pour NACA0020
            t_over_c=0.20,             # NACA 0020 → t/c = 0.20
            sweep_c2_deg=0.0,
        )
    )
    
    flap_res = compute_flap_drag_increment(
    fc=fc,
    geom=geom,
    aero=aero,
  
)

    # ==============================
    # Assemblage Cf–FF–IF par composante
    # ==============================

    ht_drag = build_component_drag(
        name="Horizontal Tail",
        Cf_eq=Cf_ht_eq,
        Cdf_base=Cdf_ht_base,
        D_base=D_ht_base,
        FF=FF_ht,
        IF_=IF_tail,
    )

    vt_drag = build_component_drag(
        name="Vertical Tail",
        Cf_eq=Cf_vt_eq,
        Cdf_base=Cdf_vt_base,
        D_base=D_vt_base,
        FF=FF_vt,
        IF_=IF_tail,
    )

    fus_drag = build_component_drag(
        name="Fuselage",
        Cf_eq=Cf_fus_eq,
        Cdf_base=Cdf_fus_base,
        D_base=D_fus_base,
        FF=FF_fus,
        IF_=IF_fuselage,
    )

    moteur_drag = build_component_drag(
        name="Engine/Nacelle",
        Cf_eq=Cf_moteur_eq,
        Cdf_base=Cdf_moteur_base,
        D_base=D_moteur_base,
        FF=FF_moteur,
        IF_=IF_moteur,
    )

    pylon_drag = build_component_drag(
        name="Pylon",
        Cf_eq=Cf_pylon_eq,
        Cdf_base=Cdf_pylon_base,
        D_base=D_pylon_base,
        FF=FF_pylon,
        IF_=IF_pylon,
    )
    
    flap_drag = ComponentDrag(
    name="Flaps",
    Cf_eq=0.0,
    FF=1.0,
    IF_=1.0,
    Cdf_base=flap_res.delta_CD,
    Cdf_total=flap_res.delta_CD,
    D_base=flap_res.D,
    D_total=flap_res.D,
)

    return [ ht_drag, vt_drag, fus_drag, moteur_drag, pylon_drag, flap_drag]

# ==============================
# 4) Résumé imprimable
# ==============================

def main():
    # Utiliser les mêmes inputs partout
    fc, geom, aero = get_default_inputs()

    # 1) Traînée de frottement + FF + IF pour ailes, empennages, fuselage, nacelle
    components = compute_all_components(fc=fc, geom=geom, aero=aero)

    # 2) Incrément de traînée du train principal (Gudmundsson Eq. 16-155)
    gear_res = compute_main_gear_drag_increment(
        fc=fc,
        geom=geom,
        aero=aero,
        config_key="J1",   # à ajuster au besoin
    )

    main_gear_drag = build_component_drag(
        name="Main Gear",
        Cf_eq=0.0,
        Cdf_base=gear_res.delta_CD,
        D_base=gear_res.D,
        FF=1.0,
        IF_=1.0,
    )

    # 3) Incrément de traînée du tail gear (taildragger)
    tail_res = compute_tail_gear_drag_increment(
        fc=fc,
        geom=geom,
        aero=aero,
    )

    tail_gear_drag = build_component_drag(
        name="Tail Gear",
        Cf_eq=0.0,
        Cdf_base=tail_res.delta_CD,
        D_base=tail_res.D,
        FF=1.0,
        IF_=1.0,
    )

    # On ajoute les deux trains à la liste des composantes
    components_with_gear = components + [main_gear_drag, tail_gear_drag]

    # 4) Impression détaillée
    print(f"{'Component':<18}  {'Cf_eq':>10}  {'FF':>8}  {'IF':>8}  {'Cdf_tot':>10}  {'D_tot [lbf]':>12}")
    print("-"*70)
    for comp in components_with_gear:
        print(
            f"{comp.name:<18}  "
            f"{comp.Cf_eq:10.5e}  "
            f"{comp.FF:8.3f}  "
            f"{comp.IF_:8.3f}  "
            f"{comp.Cdf_total:10.5e}  "
            f"{comp.D_total:12.4f}"
        )

    # 5) Totaux incluant les landing gears
    D_total_aircraft   = sum(c.D_total for c in components_with_gear)
    Cdf_total_aircraft = sum(c.Cdf_total for c in components_with_gear)

    print("\n=== TOTAL AIRCRAFT  ===")
    print(f"Cdf_total = {Cdf_total_aircraft:.5e}")
    print(f"D_total   = {D_total_aircraft:.4f} lbf")


if __name__ == "__main__":
    main()
    
    
@dataclass
class ComponentDrag:
    name: str
    Cf_eq: float
    FF: float
    IF_: float
    Cdf_base: float
    Cdf_total: float
    D_base: float
    D_total: float

# ... all your existing helper + compute_all_components, including:
# - pylon friction
# - main gear drag
# - tail gear drag
# - 25% crud factor applied ONLY to the parasite components

def compute_parasite_drag(fc=None, geom=None, aero=None):
    """
    Returns:
        components_with_gears: list[ComponentDrag]
        Cdf_total_parasite: float
        D_total_parasite:   float [lbf]
    """
    if fc is None or geom is None or aero is None:
        fc, geom, aero = get_default_inputs()

    # 1) friction+form+IF (+ pylon, + Krud)
    components = compute_all_components(fc=fc, geom=geom, aero=aero)

    # 2) main gear
    gear_res = compute_main_gear_drag_increment(fc=fc, geom=geom, aero=aero, config_key="J1")
    main_gear_drag = build_component_drag(
        name="Main Gear",
        Cf_eq=0.0,
        Cdf_base=gear_res.delta_CD,
        D_base=gear_res.D,
        FF=1.0,
        IF_=1.0,
    )

    # 3) tail gear
    tail_res = compute_tail_gear_drag_increment(fc=fc, geom=geom, aero=aero)
    tail_gear_drag = build_component_drag(
        name="Tail Gear",
        Cf_eq=0.0,
        Cdf_base=tail_res.delta_CD,
        D_base=tail_res.D,
        FF=1.0,
        IF_=1.0,
    )

    components_with_gears = components + [main_gear_drag, tail_gear_drag]

    D_total = sum(c.D_total for c in components_with_gears)
    Cdf_total = sum(c.Cdf_total for c in components_with_gears)

    return components_with_gears, Cdf_total, D_total