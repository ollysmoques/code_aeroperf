from Aircraft_data import get_default_inputs

# ============================================================
#  Définition du déplacement du CG avec le Fuel Burn
# ============================================================

def get_cg_parameters():
    """Récupère les paramètres nécessaires du fichier Aircraft_data."""
    _, _, aero = get_default_inputs()
    
    # Poids pour les bornes
    MAX_TO = aero.MAX_TO  # Poids Max au décollage (W_initial)
    WEIGHT_RESERVE = aero.OEW + aero.RESERVE + aero.PAYLOAD # Poids avec juste le fuel de réserve (W_final)
    
    # Positions du CG (en fraction de MAC) aux bornes
    # NOTE: Ces valeurs doivent être ajustées si les paramètres par défaut changent.
    # J'ai arbitrairement mis le CG arrière à la masse minimale fuel, ce qui est courant.
    CG_MAC_AT_MTOW = 0.30541  # Position du CG à MAX_TO (aero.cg_mac par défaut)
    CG_MAC_AT_RESERVE = 0.32507# Position du CG à WEIGHT_RESERVE (déplacé vers l'arrière)
    
    # Pour confirmation : 
    # Le carburant brûlé est 55.0 lb. S'il est stocké derrière le CG initial,
    # le CG se déplacera vers l'avant quand on le brûle.
    # Ici, nous supposons qu'il se déplace vers l'ARRIÈRE (de 0.20 à 0.30)
    # lorsque le poids DIMINUE, ce qui est courant pour un réservoir centré/avant.

    return MAX_TO, WEIGHT_RESERVE, CG_MAC_AT_MTOW, CG_MAC_AT_RESERVE


def compute_cg_mac(current_weight: float) -> float:
    """
    Calcule la position du CG (en fraction de MAC) basée sur le poids actuel.
    Interpolation linéaire entre CG_MTOW et CG_RESERVE.
    """
    MAX_TO, WEIGHT_RESERVE, CG_MAC_AT_MTOW, CG_MAC_AT_RESERVE = get_cg_parameters()

    W_fuel_total = MAX_TO - WEIGHT_RESERVE
    
    # Sécurité pour les bornes
    if current_weight >= MAX_TO:
        return CG_MAC_AT_MTOW
    
    if current_weight <= WEIGHT_RESERVE or W_fuel_total < 1e-6:
        return CG_MAC_AT_RESERVE
    
    # Facteur d'interpolation basé sur le carburant restant.
    # Quand W = MAX_TO, factor = 1.0
    # Quand W = WEIGHT_RESERVE, factor = 0.0
    factor = (current_weight - WEIGHT_RESERVE) / W_fuel_total
    
    # Interpolation linéaire:
    # cg_mac = CG_RESERVE + factor * (CG_MTOW - CG_RESERVE)
    cg_mac = CG_MAC_AT_RESERVE + factor * (CG_MAC_AT_MTOW - CG_MAC_AT_RESERVE)
    
    return cg_mac