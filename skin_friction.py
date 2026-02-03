import math
from dataclasses import dataclass

# ============================================================
# 1) Viscosité (loi de Sutherland)
# ============================================================

def mu_sutherland_SI(T_K: float) -> float:
    """
    Viscosité dynamique de l'air (loi de Sutherland, SI)
    T_K : température en Kelvin
    Retour : mu en N·s/m²
    """
    C1 = 1.458e-6
    S  = 110.4
    return C1 * T_K**1.5 / (T_K + S)


def mu_sutherland_imperial(T_R: float) -> float:
    """
    Viscosité dynamique (impérial) Eq. (16-25) Gudmundsson.
    T_R : température en Rankine
    Retour : mu en lb_f·s/ft²
    """
    return 3.170e-11 * (734.77 * T_R**1.5) / (216.0 + T_R)


# ============================================================
# 2) Reynolds
# ============================================================

def reynolds_number(rho: float, V: float, c: float, mu: float) -> float:
    """
    Reynolds basé sur la corde ou une longueur caractéristique.
    rho, V, c, mu : unités cohérentes entre elles (SI ou impérial).
    """
    return rho * V * c / mu


# ============================================================
# 3) Coefficients de skin friction de base
# ============================================================


def cf_laminar_avg(Re_c: float) -> float:
    """Eq. (16-40) : C_f,lam = 1.328 / sqrt(Re_c)"""
    if Re_c <= 1.0:
        return 0.0
    return 1.328 / math.sqrt(Re_c)


def cf_turbulent_PS_avg(Re_c: float) -> float:
    """Eq. (16-43) : C_f,turb = 0.455 / (log10(Re_c))^2.58"""
    if Re_c <= 1.0:
        return 0.0
    return 0.455 / (math.log10(Re_c) ** 2.58)


# ============================================================
# 4) Méthode Young : écoulement mixte laminaire–turbulent
# ============================================================

def x0_over_c_young(Re_c: float, xtr_over_c: float) -> float:
    """
    Eq. (16-69) :
    (x0/c) = 36.9 * (xtr/c)^0.625 * (1/Re_c)^0.375
    """
    return 36.9 * (xtr_over_c ** 0.625) * ((1.0 / Re_c) ** 0.375)


def cf_mixed_young(Re_c: float, xtr_over_c: float) -> float:
    """
    Eq. (16-70) :
    C_f,mixed = 0.074 / Re_c^0.2 * [ 1 - (xtr - x0)/c ]^0.8
    """
    x0c = x0_over_c_young(Re_c, xtr_over_c)
    factor = 1.0 - (xtr_over_c - x0c)
    factor = max(factor, 0.0)  # sécurité numérique
    return 0.074 / (Re_c ** 0.2) * (factor ** 0.8)


# ============================================================
# 5) Rugosité & Re_cutoff (Gudmundsson Table 16-2 / Eq. 16-31, 16-32)
# ============================================================

# k en pieds, tirés de Table 16-2
ROUGHNESS_FT = {
    "camouflage_paint": 3.33e-5,
    "smooth_paint":     2.08e-5,
    "production_sheet": 1.33e-5,
    "polished_sheet":   0.50e-5,
    "smooth_molded":    0.17e-5,
}

def roughness_height(surface_type: str, units_c: str = "ft") -> float:
    """
    Hauteur de rugosité k, dans les mêmes unités que la corde c.
    surface_type : clé du dict ROUGHNESS_FT
    units_c      : 'ft' ou 'm'
    """
    k_ft = ROUGHNESS_FT[surface_type]
    if units_c == "ft":
        return k_ft
    elif units_c == "m":
        return k_ft * 0.3048
    else:
        raise ValueError("units_c must be 'ft' or 'm'")


def reynolds_cutoff(c: float, k: float, M_inf: float,
                    regime: str = "subsonic") -> float:
    """
    Re_cutoff selon Gudmundsson :
      - Eq. (16-31) subsonique
      - Eq. (16-32) trans/supersonique
    c, k : mêmes unités (ft ou m)
    """
    ratio = c / k
    reg = regime.lower()
    if reg == "subsonic":
        return 38.21 * (ratio ** 1.053)
    elif reg in ("transonic", "supersonic"):
        return 44.62 * (ratio ** 1.053) * (M_inf ** 1.16)
    else:
        raise ValueError("regime must be 'subsonic', 'transonic' or 'supersonic'")


# ============================================================
# 6) Dataclasses résultats
# ============================================================

@dataclass
class WingFrictionResult:
    mu: float
    Re_root_geom: float
    Re_tip_geom: float
    Re_root_eff: float
    Re_tip_eff: float
    Re_root_cut: float
    Re_tip_cut: float

    Cf_root_upper: float
    Cf_root_lower: float
    Cf_root_avg: float

    Cf_tip_upper: float
    Cf_tip_lower: float
    Cf_tip_avg: float

    Cf_wing_avg: float
    Cdf_wing: float
    Df_wing: float


@dataclass
class FuselageFrictionResult:
    mu: float
    Re_geom: float
    Re_eff: float
    Re_cut: float

    Cf_upper: float
    Cf_lower: float
    Cf_avg: float

    Cdf: float
    Df: float


# ============================================================
# 7) Helper générique : choix du C_f en fonction du modèle
# ============================================================

def _cf_surface(model_local: str, Re: float, xtr: float | None) -> float:
    m = model_local.lower()
    if m == "laminar":
        return cf_laminar_avg(Re)
    elif m == "turbulent":
        return cf_turbulent_PS_avg(Re)
    elif m == "mixed":
        if xtr is None:
            raise ValueError("xtr must be provided for mixed model")
        return cf_mixed_young(Re, xtr)
    else:
        raise ValueError("model must be 'laminar', 'turbulent' or 'mixed'")


# ============================================================
# 8) BOÎTE NOIRE GÉNÉRIQUE : surface portante (aile / empennage)
# ============================================================

def compute_lifting_surface_skin_friction(
    rho: float,
    V: float,
    T: float,
    M_inf: float,
    c_root: float,
    c_tip: float,
    S_wet: float,
    S_ref: float,
    surface_type: str = "smooth_paint",
    regime: str = "subsonic",
    units: str = "imperial",
    model: str = "mixed",
    xtr_root_upper: float | None = None,
    xtr_root_lower: float | None = None,
    xtr_tip_upper: float | None = None,
    xtr_tip_lower: float | None = None,
    mu_override: float | None = None,
) -> WingFrictionResult:
    """
    Boîte noire générique pour une surface portante (aile, HT, VT).
    """

    # --- viscosité ---
    if mu_override is not None:
        mu = mu_override
    else:
        if units.lower() == "si":
            mu = mu_sutherland_SI(T)
        elif units.lower() == "imperial":
            mu = mu_sutherland_imperial(T)
        else:
            raise ValueError("units must be 'SI' or 'imperial'")

    # --- Reynolds géométriques ---
    Re_root_geom = reynolds_number(rho, V, c_root, mu)
    Re_tip_geom  = reynolds_number(rho, V, c_tip,  mu)

    # --- Re_cutoff ---
    units_c = "ft" if units.lower() == "imperial" else "m"
    k_root  = roughness_height(surface_type, units_c=units_c)
    k_tip   = k_root

    Re_root_cut = reynolds_cutoff(c_root, k_root, M_inf, regime)
    Re_tip_cut  = reynolds_cutoff(c_tip,  k_tip,  M_inf, regime)

    Re_root_eff = min(Re_root_geom, Re_root_cut)
    Re_tip_eff  = min(Re_tip_geom,  Re_tip_cut)

    # --- Cf root / tip ---
    Cf_root_upper = _cf_surface(model, Re_root_eff, xtr_root_upper)
    Cf_root_lower = _cf_surface(model, Re_root_eff, xtr_root_lower)
    Cf_root_avg   = 0.5 * (Cf_root_upper + Cf_root_lower)

    Cf_tip_upper = _cf_surface(model, Re_tip_eff, xtr_tip_upper)
    Cf_tip_lower = _cf_surface(model, Re_tip_eff, xtr_tip_lower)
    Cf_tip_avg   = 0.5 * (Cf_tip_upper + Cf_tip_lower)

    # --- moyenne root + tip ---
    Cf_wing_avg = 0.5 * (Cf_root_avg + Cf_tip_avg)

    # --- Cdf et Df ---
    Cdf_wing = Cf_wing_avg * (S_wet / S_ref)
    q        = 0.5 * rho * V**2
    Df_wing  = q * S_ref * Cdf_wing

    return WingFrictionResult(
        mu=mu,
        Re_root_geom=Re_root_geom,
        Re_tip_geom=Re_tip_geom,
        Re_root_eff=Re_root_eff,
        Re_tip_eff=Re_tip_eff,
        Re_root_cut=Re_root_cut,
        Re_tip_cut=Re_tip_cut,
        Cf_root_upper=Cf_root_upper,
        Cf_root_lower=Cf_root_lower,
        Cf_root_avg=Cf_root_avg,
        Cf_tip_upper=Cf_tip_upper,
        Cf_tip_lower=Cf_tip_lower,
        Cf_tip_avg=Cf_tip_avg,
        Cf_wing_avg=Cf_wing_avg,
        Cdf_wing=Cdf_wing,
        Df_wing=Df_wing,
    )


# --- Wrappers spécifiques : pour compatibilité avec ton code actuel ---

def compute_wing_skin_friction(**kwargs) -> WingFrictionResult:
    return compute_lifting_surface_skin_friction(**kwargs)


def compute_horizontal_tail_skin_friction(**kwargs) -> WingFrictionResult:
    return compute_lifting_surface_skin_friction(**kwargs)


def compute_vertical_tail_skin_friction(**kwargs) -> WingFrictionResult:
    return compute_lifting_surface_skin_friction(**kwargs)


# ============================================================
# 9) BOÎTE NOIRE GÉNÉRIQUE : fuselage, nacelle, etc.
# ============================================================

def compute_body_skin_friction(
    rho: float,
    V: float,
    T: float,
    M_inf: float,
    c_ref: float,          # longueur caractéristique (ex: longueur fuselage)
    S_wet: float,
    S_ref: float,
    surface_type: str = "smooth_paint",
    regime: str = "subsonic",
    units: str = "imperial",
    model: str = "turbulent",
    xtr_upper: float | None = None,
    xtr_lower: float | None = None,
    mu_override: float | None = None,
) -> FuselageFrictionResult:
    """
    Boîte noire générique pour un corps de révolution (fuselage, nacelle, etc.).
    """

    # --- viscosité ---
    if mu_override is not None:
        mu = mu_override
    else:
        if units.lower() == "si":
            mu = mu_sutherland_SI(T)
        elif units.lower() == "imperial":
            mu = mu_sutherland_imperial(T)
        else:
            raise ValueError("units must be 'SI' or 'imperial'")

    # --- Reynolds géométrique ---
    Re_geom = reynolds_number(rho, V, c_ref, mu)

    # --- Re_cutoff ---
    units_c = "ft" if units.lower() == "imperial" else "m"
    k       = roughness_height(surface_type, units_c=units_c)

    Re_cut = reynolds_cutoff(c_ref, k, M_inf, regime)
    Re_eff = min(Re_geom, Re_cut)

    # --- Cf haut / bas ---
    Cf_upper = _cf_surface(model, Re_eff, xtr_upper)
    Cf_lower = _cf_surface(model, Re_eff, xtr_lower)
    Cf_avg   = 0.5 * (Cf_upper + Cf_lower)

    # --- Cdf et Df ---
    Cdf = Cf_avg * (S_wet / S_ref)
    q   = 0.5 * rho * V**2
    Df  = q * S_ref * Cdf

    return FuselageFrictionResult(
        mu=mu,
        Re_geom=Re_geom,
        Re_eff=Re_eff,
        Re_cut=Re_cut,
        Cf_upper=Cf_upper,
        Cf_lower=Cf_lower,
        Cf_avg=Cf_avg,
        Cdf=Cdf,
        Df=Df,
    )


# Wrappers pour compatibilité

def compute_fuselage_skin_friction(**kwargs) -> FuselageFrictionResult:
    return compute_body_skin_friction(**kwargs)


def compute_engine_skin_friction(**kwargs) -> FuselageFrictionResult:
    return compute_body_skin_friction(**kwargs)


def compute_pylon_skin_friction(**kwargs) -> WingFrictionResult:
    """
    Wrapper dédié pour le pylône (utilise la même boîte noire que les surfaces portantes).
    """
    return compute_lifting_surface_skin_friction(**kwargs)