from dataclasses import dataclass
import math


# ==============================
# 1) Conditions de vol
# ==============================

@dataclass
class FlightConditions:
    rho: float     # slug/ft^3
    V: float       # ft/s
    T_R: float     # Rankine
    M_inf: float   # -

    @property
    def q(self) -> float:
        """Pression dynamique q = 0.5 rho V^2"""
        return 0.5 * self.rho * self.V**2


# ==============================
# 2) Géométrie
# ==============================

@dataclass
class WingGeometry:
    c_root: float
    c_tip: float
    S_ref: float
    S_wet: float


@dataclass
class TailGeometry:
    c_root: float
    c_tip: float
    S_wet: float


@dataclass
class VerticalTailGeometry:
    c_root: float
    c_tip: float
    S_wet_upper: float
    S_wet_lower: float


@dataclass
class FuselageGeometry:
    length: float
    S_wet: float
    d_max: float    # diamètre max (approx.)
    

@dataclass
class MainGear:
    tire_width: float
    tire_diameter: float
    strut_lenght: float    # longueur de jambe de train    
    
    
@dataclass   
class TailGear:
    tire_width: float
    tire_diameter: float
    e: float


@dataclass
class NacelleGeometry:
    length: float
    S_wet: float
    d_max: float    # diamètre max (approx.)
    

@dataclass
class PylonGeometry:
    c_root: float
    c_tip: float
    S_wet: float
    

@dataclass
class FlapGeometry:
    S_flap: float
    c_flap: float
    delta_f_deg: float  # déflexion en degrés
    t_over_c: float     # t/c, ici 0.12 pour les polynômes
    

@dataclass
class MotorParameter:
    Max_thrust_sl: float
    Fuelflow_power_setting: float
       

@dataclass
class AircraftGeometry:
    S_ref: float
    wing: WingGeometry
    ht: TailGeometry
    vt: VerticalTailGeometry
    fus: FuselageGeometry
    nacelle: NacelleGeometry
    pylon: PylonGeometry
    gear: MainGear
    tailgear: TailGear
    Motor: MotorParameter
    flaps: FlapGeometry | None = None
    

# ==============================
# 3) Paramètres aéro (non géométriques)
# ==============================

@dataclass
class AeroParams:
    # --- paramètres existants ---
    t_over_c_wing: float
    t_over_c_tail: float
    sweep_c2_wing_deg: float
    sweep_c2_tail_deg: float

    
# Et ajoutez ces valeurs par défaut dans get_default_inputs()

    IF_wing_key: str      # clé pour get_interference_factor
    IF_tail_key: str
    IF_fuselage: float    # souvent 1.0
    IF_moteur_key: str

    # --- paramètres pour l'induced drag / stabilité ---
    Cm_emp: float      # moment coeff (empennage)
    cg_mac: float      # position du CG en c/ MAC
    cp_mac: float      # position CP aile en c/ MAC
    lt: float          # bras de levier tail (ft)
    MAC: float         # Mean Aerodynamic Chord (ft)

    AR_wing: float     # aspect ratio aile
    AR_emp: float      # aspect ratio empennage horizontal
    h_v_stab: float    # hauteur du stab (ft)
    b_h_stab: float    # envergure du stab horiz. (ft)
    
    # --- paramètres de décollage / mission ---
    v_trans_kts: float     # vitesse de transition (kts)
    alpha_ini_deg: float
    alpha_trans_deg: float
    alpha_rot_deg: float
    thrust_TO_lbf: float   # poussée dispo au décollage [lbf]
    h_TO_ft: float         # altitude pour ce cas de TO [ft]
    dT_isa_TO: float       # delta ISA pour ce cas [°C]
    mu_TO: float           # coefficient de friction au sol
    CL_max_TO: float       # CL_max utilisé pour le TO
    CL_max_clean: float    # CL_max propre (clean)

    # --- masses ---
    OEW: float             # Operating Empty Weight [lb]
    FUEL_LOAD: float       # fuel total au décollage [lb]
    RESERVE: float         # fuel de réserve [lb]
    PAYLOAD: float = 0.0   # charge utile [lb]
    MAX_TO: float = 0.0    # sera calculé automatiquement si laissé à 0

    def __post_init__(self):
        """Calcul automatique du MTOW si non spécifié."""
        if self.MAX_TO == 0.0:
            # Si tu veux exclure la payload, enlève self.payload ici
            self.MAX_TO = self.OEW + self.FUEL_LOAD + self.RESERVE + self.PAYLOAD


# ==============================
# 4) Inputs par défaut
# ==============================

def get_default_inputs() -> tuple[FlightConditions, AircraftGeometry, AeroParams]:
    """
    Regroupe les valeurs "hard-codées" actuelles.
    Tu pourras plus tard :
    - les lire dans un JSON,
    - les modifier selon un cas de vol, etc.
    """
    # --- Conditions de vol ---
    rho   = 0.002378                  # slug/ft^3
    V     = 108.0 * 1.688             # ft/s
    T_R   = 518.67                    # Rankine (ISA SL)
    M_inf = 0.2                       # Mach approx.

    fc = FlightConditions(rho=rho, V=V, T_R=T_R, M_inf=M_inf)

    # --- Géométrie ---
    S_ref = 36.8                      # ft^2

    wing = WingGeometry(
        c_root=2.30,
        c_tip=2.22,
        S_ref=S_ref,
        S_wet=69.41,
    )

    ht = TailGeometry(
        c_root=1.10,
        c_tip=1.10,
        S_wet=10.70,
    )

    vt = VerticalTailGeometry(
        c_root=1.3075,
        c_tip=0.98,
        S_wet_upper=4.5856,
        S_wet_lower=2.754,
    )

    fus = FuselageGeometry(
        length=12.1,
        S_wet=71.59,
        d_max=2.20,
    )

    nacelle = NacelleGeometry(
        length=1.10,
        S_wet=3.04,
        d_max=1.05,
    )
    
    pylon = PylonGeometry(
        c_root=0.75,
        c_tip=0.75, 
        S_wet=1.125,
    )
    
    gear = MainGear(
        tire_width=0.23,
        tire_diameter=0.7,
        strut_lenght=0.15,
    )
    
    tail_gear = TailGear(
        tire_width=0.25,
        tire_diameter=0.61, 
        e=0.974,
    )
    
    flaps = FlapGeometry(
        S_flap      = 2.25,
        c_flap      = 0.5,
        delta_f_deg = 0.0,      # 0° ou 15° selon la config
        t_over_c    = 0.12,
    )
    
    Turbine = MotorParameter(
        Max_thrust_sl = 89.0,
        Fuelflow_power_setting = 0.023 * 100.0,
    )

    geom = AircraftGeometry(
        S_ref   = S_ref,
        wing    = wing,
        ht      = ht,
        vt      = vt,
        fus     = fus,
        nacelle = nacelle,
        pylon   = pylon,
        gear    = gear,
        tailgear= tail_gear,
        Motor   = Turbine,
        flaps   = flaps,
    )
        
    aero = AeroParams(
        t_over_c_wing      = 0.15,
        t_over_c_tail      = 0.12,
        sweep_c2_wing_deg  = 0.0,
        sweep_c2_tail_deg  = 0.0,
        IF_wing_key        = "low_wing_unfilleted",
        IF_tail_key        = "h_tail_lockheed_electra",
        IF_fuselage        = 1.0,
        IF_moteur_key      = "nacelle_under_fuselage_more_1D",

        # ---- paramètres induits / stabilité ----
        Cm_emp   = -0.05,
        cg_mac   = 0.20,
        cp_mac   = 0.25,
        lt       = 6.355,
        MAC      = 2.30,
        AR_wing  = 7.18,
        AR_emp   = 6.0,
        h_v_stab = 1.5,
        b_h_stab = 6.0,
        
        # ---- décollage / cas de mission ----
        v_trans_kts    = 20.0,
        alpha_ini_deg   = 6.0,
        alpha_trans_deg = 3.0,
        alpha_rot_deg   = 6.0,
        dT_isa_TO       = 0.0,
        mu_TO           = 0.04,
        CL_max_TO       = 1.4,
        CL_max_clean    = 1.2,

        # ---- masses ----
        OEW       = 126.0,
        FUEL_LOAD = 75.0,
        RESERVE   = 7.0,
        PAYLOAD   = 170,   # tu pourras changer ça plus tard
        # MAX_TO non fourni → sera calculé dans __post_init__
    )

    return fc, geom, aero