# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:46:50 2025

@author: edoua
"""

from dataclasses import dataclass

from Aircraft_data import (
    FlightConditions,
    AircraftGeometry,
    AeroParams,
    get_default_inputs,
)


@dataclass
class LandingGearConfig:
    key: str           # ex: "A", "B", "C1", ...
    description: str   # description textuelle
    delta_CDs: float   # valeur tabulée ΔC_Ds (référencée à A = d×w)


# Dictionnaire minimal – à compléter si tu veux toutes les lignes
# Les valeurs ci-dessous viennent directement de la Table 16-15
# (NACA R-485). Tu peux ajouter d’autres entrées au besoin.
LANDING_GEAR_CONFIGS: dict[str, LandingGearConfig] = {
    "A": LandingGearConfig(
        key="A",
        description="8.50-10 tire (type A configuration)",
        delta_CDs=1.112,
    ),
    "B": LandingGearConfig(
        key="B",
        description="8.50-10 tire (type B configuration)",
        delta_CDs=1.204,
    ),
    "C1": LandingGearConfig(
        key="C1",
        description="8.50-10 + streamline wire",
        delta_CDs=1.151,
    ),
    "J1": LandingGearConfig(
        key="J1",
        description="8.50-10 + streamline wire",
        delta_CDs=0.615,
    ),
    # ...
    # Tu peux compléter avec C2, C3, D1, etc.
    # "C2": LandingGearConfig(...),
}


# ============================================================
# 2) Résultat de l’incrément de traînée du train
# ============================================================

@dataclass
class GearDragResult:
    config_key: str
    delta_CDs: float       # valeur tabulée (Table 16-15)
    delta_CD: float        # incrément de traînée global avion
    D: float               # force de traînée correspondante [lbf]


# ============================================================
# 3) Fonction principale : ΔCD du train principal
# ============================================================

def compute_main_gear_drag_increment(
    fc: FlightConditions | None = None,
    geom: AircraftGeometry | None = None,
    aero: AeroParams | None = None,     # pas utilisé ici mais gardé pour homogénéité
    *,
    config_key: str | None = None,
    delta_CDs_manual: float | None = None,
) -> GearDragResult:
    """
    Calcule l'incrément de traînée dû au train principal selon Gudmundsson Eq. (16-155) :

        ΔCD_fixed = (d * w / S_ref) * ΔC_Ds

    - Si fc / geom / aero ne sont pas fournis, on prend get_default_inputs().
    - Tu peux soit fournir :
        * config_key → va chercher ΔC_Ds dans LANDING_GEAR_CONFIGS
        * delta_CDs_manual → tu fournis directement la valeur tabulée
    """

    # Inputs par défaut si nécessaire
    if fc is None or geom is None or aero is None:
        fc, geom, aero = get_default_inputs()

    # Récupération de ΔC_Ds
    if delta_CDs_manual is not None and config_key is not None:
        raise ValueError("Donne soit 'config_key', soit 'delta_CDs_manual', pas les deux.")

    if delta_CDs_manual is not None:
        delta_CDs = delta_CDs_manual
        used_key = "MANUAL"
    else:
        if config_key is None:
            raise ValueError("Spécifie 'config_key' ou 'delta_CDs_manual'.")
        if config_key not in LANDING_GEAR_CONFIGS:
            raise KeyError(f"Configuration de train inconnue: {config_key}")
        cfg = LANDING_GEAR_CONFIGS[config_key]
        delta_CDs = cfg.delta_CDs
        used_key = cfg.key

    # Dimensions du pneu (géométrie du train)
    d = geom.gear.tire_diameter   # ft
    w = geom.gear.tire_width      # ft
    S_ref = geom.S_ref            # ft^2

    # Eq. (16-155) : ΔCD_fixed = (d * w / S_ref) * ΔC_Ds
    delta_CD = (d * w / S_ref) * delta_CDs

    # Force de traînée correspondante
    D = fc.q * S_ref * delta_CD   # [lbf]

    return GearDragResult(
        config_key=used_key,
        delta_CDs=delta_CDs,
        delta_CD=delta_CD,
        D=D,
    )

def compute_tail_gear_drag_increment(
    fc: FlightConditions | None = None,
    geom: AircraftGeometry | None = None,
    aero: AeroParams | None = None,     # pas utilisé ici mais gardé pour homogénéité
    *,
    config_key: str | None = None,
    delta_CDs_manual: float | None = None,
) -> GearDragResult:
    delta_CDs = 0.42
    
    d = geom.tailgear.tire_diameter
    w = geom.tailgear.tire_width
    S_ref = geom.S_ref
    
    delta_CD = (d * w / S_ref) * delta_CDs
    D = fc.q * S_ref * delta_CD 
    
    return GearDragResult(
        config_key=None,
        delta_CDs=delta_CDs,
        delta_CD=delta_CD,
        D=D,
    )



# ============================================================
# 4) Incrément de traînée des volets (Plain flaps, t/c = 0.12)
#    ΔCD_flap = Δ1 · Δ2 · (S_flap / S_ref)
#    R_f = c_f / c   (rapport corde volet / corde locale)
#    δ_f en degrés
#    Formules tirées des Tables 16-16 et 16-17 de Gudmundsson
# ============================================================

@dataclass
class FlapDragResult:
    delta1: float       # fonction Δ1 (géométrie du volet)
    delta2: float       # fonction Δ2 (déflexion du volet)
    delta_CD: float     # incrément de traînée global ΔCD_flap
    D: float            # traînée correspondante [lbf]


def delta1_plain_tc012(R_f: float) -> float:
    # Table 16-16, plain flap t/c = 0.12
    return (
        -21.090 * R_f**3
        + 14.091 * R_f**2
        + 3.165  * R_f
        - 0.00103
    )

def delta2_plain_tc012(delta_f_deg: float) -> float:
    # Table 16-17, plain flap t/c = 0.12
    d = delta_f_deg
    return (
        -3.795e-7 * d**3
        + 5.387e-5 * d**2
        + 6.843e-4 * d
        - 1.4729e-3
    )

@dataclass
class FlapDragResult:
    delta_CD: float
    D: float


def compute_flap_drag_increment(
    fc: FlightConditions | None = None,
    geom: AircraftGeometry | None = None,
    aero: AeroParams | None = None,
) -> FlapDragResult:
    # Inputs par défaut si pas fournis
    if fc is None or geom is None or aero is None:
        fc, geom, aero = get_default_inputs()

    flap = geom.flaps

    # Sécurité : si pas de flaps ou déflexion nulle → pas d’incrément
    if flap is None or flap.S_flap <= 0.0 or abs(flap.delta_f_deg) < 1e-6:
        return FlapDragResult(delta_CD=0.0, D=0.0)

    # ---- 1) Calcul de R_f = c_flap / c ----
    # ici j’utilise la corde racine comme référence (comme Gudmundsson)
    R_f = flap.c_flap / geom.wing.c_root

    # ---- 2) Δ1(R_f) et Δ2(δ_f) ----
    d1 = delta1_plain_tc012(R_f)
    d2 = delta2_plain_tc012(flap.delta_f_deg)

    # ---- 3) Formule Gudmundsson : ΔCD_flap = Δ1·Δ2·(S_flap / S_ref) ----
    S_ratio      = flap.S_flap / geom.S_ref
    delta_CD_flap = d1 * d2 * S_ratio

    # Force de traînée correspondante
    D_flap = fc.q * geom.S_ref * delta_CD_flap

    return FlapDragResult(
        delta_CD=delta_CD_flap,
        D=D_flap,
    )