# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 01:18:21 2025

@author: edoua
"""

import numpy as np
from atmosphere import atmosphere, vitesses
from total_drag import drag_total
from Thrust_data import thrust_sw400pro_ft_lbf
from Aircraft_data import get_default_inputs
from cg_shift import *
from Mission_parameters import MISSION_HEIGHT_FT


# ============================
# Fonction principale
# ============================

def compute_cruise_condition(CAS_kts, h_ft, dT_isa, weight_lbf, cg_mac_current):
    """
    Donne la condition de croisière pour une vitesse CAS imposée.
    Retourne : Drag, Thrust_needed, power_setting, fuel_flow associé.
    """

    # Charger géométrie et paramètres
    fc_default, geom, aero = get_default_inputs()
    Sref = geom.S_ref

    # --- Atmosphère ---
    atm = atmosphere(h_ft, dT_isa, mode="delta_isa")

    # --- Conversion CAS -> TAS + Mach ---
    speed = vitesses(atm, CAS_kts, "calibree", weight_lbf, Sref)
    TAS_kts = speed["vitesses avion kts"]["vitesse vraie"]
    q_psf   = speed["pression dynamique"]
    Mach    = speed["mach"]

    # --- Calcul du drag total ---
    CL, Drag_lbf, _ = drag_total(h_ft, dT_isa, TAS_kts, weight_lbf, cg_mac_current,'weight','clean')

    # ============================
    # Trouver Power Setting tel que Thrust = Drag
    # ============================

    # Fonction thrust = f(power_setting)
    def thrust_from_PS(PS):
        thrust, _ = thrust_sw400pro_ft_lbf(h_ft, dT_isa, PS)
        return thrust

    # On balaie le power setting de 0 → 1
    PS_vals = np.linspace(0, 1.0, 200)
    thrust_vals = np.array([thrust_from_PS(ps) for ps in PS_vals])

    # Trouver où thrust ≈ drag
    idx = np.argmin(np.abs(thrust_vals - Drag_lbf))
    PS_required = PS_vals[idx]
    thrust_required = thrust_vals[idx]

    # Fuel flow correspondant
    _, fuel_flow = thrust_sw400pro_ft_lbf(h_ft, dT_isa, PS_required)

    # ============================
    # Résultat
    # ============================

    result = {
        "CAS_kts": CAS_kts,
        "TAS_kts": TAS_kts,
        "Mach": Mach,
        "Altitude_ft": h_ft,
        "Delta_ISA": dT_isa,
        "CL": CL,
        "Drag_lbf": Drag_lbf,
        "Thrust_required": thrust_required,
        "Power_setting": PS_required,
        "Fuel_flow_lb_per_min": fuel_flow,
    }

    return result


def compute_cruise_range_time(Weight_initial, Weight_final, CAS_kts, h_ft, dT_isa):
    
    cg_mac_initial = compute_cg_mac(Weight_initial)
    cg_mac_final   = compute_cg_mac(Weight_final)
    
    
    cruise_initial = compute_cruise_condition(CAS_kts, h_ft, dT_isa, Weight_initial, cg_mac_initial)
    cruise_final   = compute_cruise_condition(CAS_kts, h_ft, dT_isa, Weight_final, cg_mac_final)
    
    wf_initial_lb_min = cruise_initial['Fuel_flow_lb_per_min']  # lb/min
    TAS_kts_initial   = cruise_initial['TAS_kts']               # NM/h
    
    wf_final_lb_min   = cruise_final['Fuel_flow_lb_per_min']
    TAS_kts_final     = cruise_final['TAS_kts']
    
    # Convertir en lb/h
    wdot_initial_lb_h = wf_initial_lb_min * 60.0
    wdot_final_lb_h   = wf_final_lb_min * 60.0
    
    SAR_initial = TAS_kts_initial / wdot_initial_lb_h   # NM/lb
    SAR_final   = TAS_kts_final   / wdot_final_lb_h     # NM/lb
    
    SAR_moyen = 0.5 * (SAR_initial + SAR_final)
    
    # Range en NM
    Range_NM = SAR_moyen * (Weight_initial - Weight_final)
    
    # Temps (min)
    TAS_kts_moyen = 0.5 * (TAS_kts_initial + TAS_kts_final)
    delta_t_min   = (Range_NM / TAS_kts_moyen) * 60.0
    
    return Range_NM, delta_t_min
    
    
def compute_fuel_burned_for_time(delta_t_min, Weight_initial, CAS_kts, h_ft, dT_isa):
    """
    Calcule le carburant consommé pour une durée de croisière donnée.
    Approche itérative (2 itérations) pour prendre en compte l'allègement.
    """
    
    # 1. Estimation Fuel Flow au poids initial
    cg_mac_initial = compute_cg_mac(Weight_initial)
    cruise_initial = compute_cruise_condition(CAS_kts, h_ft, dT_isa, Weight_initial, cg_mac_initial)
    ff_initial_lb_min = cruise_initial['Fuel_flow_lb_per_min']
    
    # 2. Premier guess du fuel brulé
    burned_guess_lb = ff_initial_lb_min * delta_t_min
    
    # 3. Itération pour affiner (milieu de segment)
    # On fait 2 itérations, ça suffit largement pour cette précision
    for _ in range(2):
        weight_mid = Weight_initial - 0.5 * burned_guess_lb
        cg_mac_mid = compute_cg_mac(weight_mid)
        
        cruise_mid = compute_cruise_condition(CAS_kts, h_ft, dT_isa, weight_mid, cg_mac_mid)
        ff_mid_lb_min = cruise_mid['Fuel_flow_lb_per_min']
        
        burned_guess_lb = ff_mid_lb_min * delta_t_min
        
    return burned_guess_lb, ff_mid_lb_min
    
    


# ============================
# Exemple d'utilisation
# ============================

if __name__ == "__main__":
    # Exemple : croisière
    CAS = 90       # [kts]
    h   = MISSION_HEIGHT_FT    # [ft]
    dT  = 0         # ISA
    W   = 378       # [lbf]

    out = compute_cruise_condition(CAS, h, dT, W)

    print("\n====== CRUISE CONDITION ======")
    print(f"CAS:               {out['CAS_kts']:.1f} kt")
    print(f"TAS:               {out['TAS_kts']:.1f} kt")
    print(f"Mach:              {out['Mach']:.3f}")
    print(f"Altitude:          {out['Altitude_ft']:.0f} ft")
    print(f"ΔISA:              {out['Delta_ISA']} °C")
    print("----------------------------------")
    print(f"CL:                {out['CL']:.4f}")
    print(f"Drag:              {out['Drag_lbf']:.2f} lbf")
    print(f"Thrust required:   {out['Thrust_required']:.2f} lbf")
    print(f"Power setting:     {100*out['Power_setting']:.1f} %")
    print(f"Fuel flow:         {out['Fuel_flow_lb_per_min']:.4f} lb/min")
    print("==================================\n")
    
    
    
    
Range, delta_t = compute_cruise_range_time(370, 350, 108, MISSION_HEIGHT_FT, 0)


print('Distance NM:',Range)
print('Temps de vol en croisière min:',delta_t)    
    