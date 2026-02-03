import math
import numpy as np
from dataclasses import dataclass

# ==============================
#  Form factors pour éléments d'aile
#  Basé sur Gudmundsson §16.3.5
# ==============================

def ff_hoerner_tc30(t_over_c: float) -> float:
    """
    Hoerner (Eq. 16-123)
    Pour profils avec (x/c)_max ≈ 0.30
    FF = 1 + 2(t/c) + 60(t/c)^4
    """
    tc = t_over_c
    return 1.0 + 2.0 * tc + 60.0 * tc**4


def ff_hoerner_tc40_50(t_over_c: float) -> float:
    """
    Hoerner (Eq. 16-124)
    Pour profils avec (x/c)_max ≈ 0.40–0.50
    FF = 1 + 1.2(t/c) + 70(t/c)^4
    """
    tc = t_over_c
    return 1.0 + 1.2 * tc + 70.0 * tc**4


def ff_torenbeek(t_over_c: float) -> float:
    """
    Torenbeek (Eq. 16-125)
    Valide pour t/c <= 0.21
    FF = 1 + 2.7(t/c) + 100(t/c)^4
    """
    tc = t_over_c
    return 1.0 + 2.7 * tc + 100.0 * tc**4


def ff_jenkinson_wing(t_over_c: float, sweep_c2_deg: float) -> float:
    """
    Jenkinson – surfaces portantes (Eq. 16-128)
    FF = [3.3(t/c) - 0.008(t/c)^2 + 27.0(t/c)^3] cos^2(Λ_c/2) + 1
    """
    tc = t_over_c
    lam_c2_rad = math.radians(sweep_c2_deg)
    cos2 = math.cos(lam_c2_rad)**2
    bracket = 3.3*tc - 0.008*tc**2 + 27.0*tc**3
    return bracket * cos2 + 1.0


def ff_jenkinson_tail(t_over_c: float, sweep_c2_deg: float) -> float:
    """
    Jenkinson – empennages (Eq. 16-129)
    FF = [3.52(t/c)] cos^2(Λ_c/2) + 1
    """
    tc = t_over_c
    lam_c2_rad = math.radians(sweep_c2_deg)
    cos2 = math.cos(lam_c2_rad)**2
    return (3.52 * tc) * cos2 + 1.0


# ==============================
#  Fineness ratio & form factor fuselage
# ==============================

def equivalent_diameter(A_max: float) -> float:
    """
    Diamètre équivalent d'un corps de révolution
    d_eq = 2 * sqrt(A_max / pi)
    """
    return 2.0 * math.sqrt(A_max / math.pi)


def fineness_ratio_from_L_and_Amax(L: float, A_max: float) -> float:
    d_eq = equivalent_diameter(A_max)
    return L / d_eq


def fineness_ratio_from_L_and_d(L: float, d: float) -> float:
    """fineness ratio direct si tu connais déjà le diamètre 'd'."""
    return L / d


def ff_fuselage_hoerner(f: float) -> float:
    """Hoerner (Eq. 16-133)"""
    return 1.0 + 1.5 * f**(-1.5) + 7.0 * f**(-3.0)


def ff_fuselage_torenbeek(f: float) -> float:
    """Torenbeek (Eq. 16-134)"""
    return 1.0 + 2.2 * f**(-1.5) + 3.8 * f**(-3.0)


def ff_fuselage_nicolai_raymer_roskam(f: float) -> float:
    """Nicolai / Raymer / Roskam (Eq. 16-135)"""
    return 1.0 + 60.0 / (f**3.0) + f / 400.0


def ff_fuselage_shevell(f: float) -> float:
    """Shevell (Eq. 16-136)"""
    return (
        2.939
        - 0.766 * f
        + 0.182 * f**2
        - 0.01704 * f**3
        + 3.275e-4 * f**4
    )


def ff_fuselage_jenkinson(f: float) -> float:
    """Jenkinson (Eq. 16-137)"""
    return 1.0 + 2.2 * f**(-1.5) - 0.9 * f**(-3.0)

def ff_engine_nacelle(f: float) -> float:
    return 1 + 0.35*np.sqrt(f)


# ==============================
#  Interference factors
# ==============================

@dataclass
class InterferenceFactor:
    name: str
    IF_nominal: float
    IF_min: float | None = None
    IF_max: float | None = None


INTERFERENCE_FACTOR_TABLE: dict[str, InterferenceFactor] = {
    # Nacelles / external stores
    "nacelle_under_fuselage_direct": InterferenceFactor(
        name="Nacelle or store directly under fuselage",
        IF_nominal=1.5
    ),
    "nacelle_under_fuselage_less_1D": InterferenceFactor(
        name="Nacelle or store under fuselage, < 1 diameter away",
        IF_nominal=1.3
    ),
    "nacelle_under_fuselage_more_1D": InterferenceFactor(
        name="Nacelle or store under fuselage, > 1 diameter away",
        IF_nominal=1.0
    ),
    "fuel_tank_on_wingtip": InterferenceFactor(
        name="Fuel tank on wingtip",
        IF_nominal=1.25
    ),

    # Wing / winglet
    "high_or_mid_wing_well_faired": InterferenceFactor(
        name="High or mid wing with carefully designed fairing",
        IF_nominal=1.0
    ),
    "low_wing_unfilleted": InterferenceFactor(
        name="Unfilleted low wing",
        IF_nominal=1.25,
        IF_min=1.1,
        IF_max=1.4
    ),
    "whitcomb_winglet": InterferenceFactor(
        name="Whitcomb winglet",
        IF_nominal=1.04
    ),
    "airbus_style_winglet": InterferenceFactor(
        name="Airbus-style winglet",
        IF_nominal=1.04
    ),
    "modern_blended_winglet": InterferenceFactor(
        name="Modern blended winglet",
        IF_nominal=1.005,
        IF_min=1.00,
        IF_max=1.01
    ),

    # Train principal / mats
    "main_gear_strut_one_end": InterferenceFactor(
        name="Main landing-gear strut entering wing OR fuselage",
        IF_nominal=1.10
    ),
    "main_gear_strut_both_ends": InterferenceFactor(
        name="Wing strut entering wing at one end and fuselage at the other",
        IF_nominal=1.10
    ),

    # Divers
    "boundary_layer_diverter": InterferenceFactor(
        name="Boundary-layer diverter",
        IF_nominal=1.0
    ),

    # Empennages
    "conventional_tail": InterferenceFactor(
        name="Conventional tail",
        IF_nominal=1.045,
        IF_min=1.04,
        IF_max=1.05
    ),
    "cruciform_tail": InterferenceFactor(
        name="Cruciform tail",
        IF_nominal=1.06
    ),
    "v_tail": InterferenceFactor(
        name="V-tail",
        IF_nominal=1.03
    ),
    "h_tail_b25_a10": InterferenceFactor(
        name="H-tail (B-25 / A-10 style)",
        IF_nominal=1.08
    ),
    "h_tail_lockheed_electra": InterferenceFactor(
        name="H-tail (Lockheed Electra style)",
        IF_nominal=1.13
    ),
    "h_tail_beech_d18": InterferenceFactor(
        name="H-tail (Beech D-18 style)",
        IF_nominal=1.06
    ),
    "triple_tail_constellation": InterferenceFactor(
        name="Triple-tail (Lockheed Constellation style)",
        IF_nominal=1.10
    ),
    "t_tail": InterferenceFactor(
        name="T-tail",
        IF_nominal=1.04
    ),
}


# ==============================
#  Dispatchers
# ==============================

@dataclass
class FormFactorInputs:
    method: str           # 'hoerner30', 'hoerner4050', 'torenbeek',
                          # 'jenkinson_wing', 'jenkinson_tail'
    t_over_c: float
    sweep_c2_deg: float | None = None


@dataclass
class FuselageFFInputs:
    method: str          # 'hoerner', 'torenbeek', 'nicolai', 'shevell', 'jenkinson'
    L: float | None = None
    A_max: float | None = None
    f: float | None = None       # fineness ratio si déjà connu


def compute_form_factor(inputs: FormFactorInputs) -> float:
    m  = inputs.method.lower()
    tc = inputs.t_over_c

    if m == "hoerner30":
        return ff_hoerner_tc30(tc)
    elif m == "hoerner4050":
        return ff_hoerner_tc40_50(tc)
    elif m == "torenbeek":
        return ff_torenbeek(tc)
    elif m == "jenkinson_wing":
        if inputs.sweep_c2_deg is None:
            raise ValueError("sweep_c2_deg must be provided for jenkinson_wing")
        return ff_jenkinson_wing(tc, inputs.sweep_c2_deg)
    elif m == "jenkinson_tail":
        if inputs.sweep_c2_deg is None:
            raise ValueError("sweep_c2_deg must be provided for jenkinson_tail")
        return ff_jenkinson_tail(tc, inputs.sweep_c2_deg)
    else:
        raise ValueError(
            "method must be one of "
            "'hoerner30', 'hoerner4050', 'torenbeek', "
            "'jenkinson_wing', 'jenkinson_tail'"
        )


def get_interference_factor(key: str, mode: str = "nominal") -> float:
    if key not in INTERFERENCE_FACTOR_TABLE:
        raise KeyError(f"Unknown interference factor key: {key}")

    data = INTERFERENCE_FACTOR_TABLE[key]
    m = mode.lower()

    if m == "nominal":
        return data.IF_nominal
    elif m == "min":
        return data.IF_min if data.IF_min is not None else data.IF_nominal
    elif m == "max":
        return data.IF_max if data.IF_max is not None else data.IF_nominal
    else:
        raise ValueError("mode must be 'nominal', 'min', or 'max'")


def compute_fuselage_form_factor(inputs: FuselageFFInputs) -> tuple[float, float]:
    """
    Retourne (f, FF)
    - si inputs.f est fourni -> utilisé
    - sinon, il faut fournir L et A_max
    """
    if inputs.f is not None:
        f = inputs.f
    else:
        if inputs.L is None or inputs.A_max is None:
            raise ValueError("Provide either f, or (L and A_max)")
        f = fineness_ratio_from_L_and_Amax(inputs.L, inputs.A_max)

    m = inputs.method.lower()

    if m == "hoerner":
        FF = ff_fuselage_hoerner(f)
    elif m == "torenbeek":
        FF = ff_fuselage_torenbeek(f)
    elif m in ("nicolai", "raymer", "roskam"):
        FF = ff_fuselage_nicolai_raymer_roskam(f)
    elif m == "shevell":
        FF = ff_fuselage_shevell(f)
    elif m == "jenkinson":
        FF = ff_fuselage_jenkinson(f)
    else:
        raise ValueError(
            "method must be one of "
            "'hoerner', 'torenbeek', 'nicolai', 'raymer', 'roskam', "
            "'shevell', 'jenkinson'"
        )

    return f, FF