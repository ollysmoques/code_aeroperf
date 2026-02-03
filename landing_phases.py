# -*- coding: utf-8 -*-
"""
Created on Thu Dec 4 2025

@author: up60044568
"""

import numpy as np
import math
from atmosphere import atmosphere, vitesses
from DRAG_TOTAL import drag_total
from Thrust_data import thrust_sw400pro_ft_lbf
from cg_shift import compute_cg_mac
from landing_run import v_touchdown, CL_MAX_LANDING # Nécessite l'import de CL_MAX_LANDING

# ==========================================================
# 1. CONSTANTES GLOBALES (À UNIFORMISER AVEC Aircraft_data)
# ==========================================================

# NOTE: Ces constantes doivent être importées de Aircraft_data ou ROC.py 
# pour l'uniformité. Assurez-vous que CL_MAX_LANDING est défini dans 
# landing_run.py ou dans Aircraft_data.py et importé ici.

# Utilisation des valeurs par défaut pour l'exemple :
R_FLARE_FT = 200.0   # Rayon de courbure estimé pour le flare [ft]

# ==========================================================
# 2. FONCTION PRINCIPALE : APPROCHE ET ARRONDIE
# ==========================================================

def approach_and_flare(hpi, hpf, h_flare, dT_isa, V_app_kts, weight_initial, sref, power_setting, gamma_deg):
    """
    Modélise l'approche stabilisée (glide) et l'arrondi (flare).

    hpi : Altitude de début d'approche [ft] (ex: 1000 ft)
    hpf : Altitude de fin d'approche (touchdown) [ft] (doit être 0 ft)
    h_flare : Altitude où l'arrondi commence [ft] (ex: 50 ft)
    V_app_kts : Vitesse d'approche stabilisée (V_ref) [kts]
    weight_initial : Poids au début de l'approche [lbf]
    gamma_deg : Pente d'approche stabilisée (négative) [deg] (ex: -3.0 deg)

    Returns:
        temps_total [min], distance_total [NM], Fuel_tot [lb], weight_final [lbf], V_td_kts [kts]
    """
    dh = 5.0          # Pas de pression [ft]
    hp1 = hpi         
    weight = weight_initial
    temps = 0.0       # [min]
    distance = 0.0    # [NM]
    Fuel_tot = 0.0    # [lb]
    
    gamma_rad = np.radians(gamma_deg)
    
    # 1. Calcul de V_TD pour la fin du segment (via Vs)
    Vs_ft_s, _ = v_touchdown(hpi, dT_isa, weight_initial, CL_MAX_LANDING, sref)
    V_td_kts = Vs_ft_s / 1.6878 * 1.15 
    

    # ==========================================================
    # PHASE 1 : GLIDE STABILISÉ (hpi -> h_flare)
    # L'avion maintient une pente et une vitesse constantes.
    # ==========================================================
    
    while hp1 > h_flare:
        # Détermination du prochain point
        hp2 = max(h_flare, hp1 - dh)
        
        hp_moy = 0.5 * (hp1 + hp2)
        atm = atmosphere(hp_moy, dT_isa, mode='delta_isa')
        vitesse = vitesses(atm, V_app_kts, 'calibree', weight, sref)
        TAS_kts = vitesse['vitesses avion kts']['vitesse vraie']
        
        T = atm['temperature_K']
        Tstd = T - atm['delta_ISA']

        # Temps nécessaire pour parcourir dh géométrique
        dhp = hp2 - hp1            # < 0
        dh_geo = dhp * T / Tstd    # < 0
        
        # Le taux de descente est imposé par la pente (ROD = V * sin(gamma))
        TAS_fps = TAS_kts * 1.6878
        ROD_fps = TAS_fps * np.sin(abs(gamma_rad))
        ROD_fpm = ROD_fps * 60.0
        
        # Le taux de descente doit être réalisable (Poussée - Traînée)
        # Mais pour une approche stabilisée, on suppose que la poussée est ajustée
        # pour maintenir V et gamma.
        
        delta_t = abs(dh_geo) / ROD_fpm   # [min]
        
        # Poussée & Carburant (pour la consommation)
        thrust, Fuel_consumption = thrust_sw400pro_ft_lbf(hp_moy, dT_isa, power_setting)
        
        temps += delta_t
        distance += TAS_kts * (delta_t / 60.0)
        delta_fuel = Fuel_consumption * delta_t
        Fuel_tot += delta_fuel
        
        # Mise à jour du poids pour le CG dynamique
        weight -= delta_fuel 
        hp1 = hp2
        
        if hp1 <= h_flare:
            break

    # ==========================================================
    # PHASE 2 : FLARE (Arrondi) (h_flare -> hpf=0 ft)
    # L'avion décélère verticalement et horizontalement.
    # ==========================================================
    h_start_flare = h_flare
    
    # Estimation de la distance de flare (Formule simplifiée Gudmundsson)
    # (Distance de vol stable pendant laquelle le pilote réduit le taux de chute)
    dist_flare_ft = R_FLARE_FT * np.sin(abs(gamma_rad)) + np.sqrt(R_FLARE_FT**2 * np.sin(abs(gamma_rad))**2 + 2 * R_FLARE_FT * h_start_flare)
    
    # Approximation du temps de flare : Assumer vitesse moyenne
    V_avg_flare_kts = (V_app_kts + V_td_kts) / 2
    time_flare_min = (dist_flare_ft / 6076.12) / V_avg_flare_kts * 60.0 
    
    # Mise à jour des totaux
    temps += time_flare_min
    distance += (dist_flare_ft / 6076.12) # Distance de flare en NM
    
    # Consommation de carburant durant le flare (approximation)
    hp_moy_flare = h_start_flare / 2
    thrust_flare, Fuel_consumption_flare = thrust_sw400pro_ft_lbf(hp_moy_flare, dT_isa, power_setting)
    delta_fuel_flare = Fuel_consumption_flare * time_flare_min
    
    Fuel_tot += delta_fuel_flare
    weight -= delta_fuel_flare
    
    # NOTE: Le poids 'weight' est maintenant le poids au toucher (W_td)

    return temps, distance, Fuel_tot, weight, V_td_kts

