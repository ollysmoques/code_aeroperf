import numpy as np
import matplotlib.pyplot as plt
from atmosphere import atmosphere, vitesses
from total_drag import drag_total
from Thrust_data import thrust_sw400pro_ft_lbf
from Aircraft_data import get_default_inputs
from Cruise_Condition import compute_cruise_range_time, compute_fuel_burned_for_time, compute_cruise_condition
from ROC import montee, descente, ROC, acceleration , find_initial_weight_for_descent
from take_off_run import groundrun 
from cg_shift import *
from landing_run import *  
from landing_phases import * 
from Mission_parameters import MISSION_HEIGHT_FT


def FLIGHT_PHASES_TIME_IMPOSED(total_time_imposed_min, h_cruise, h_airport, dT_isa, VY, V_cruise_CAS, TO_weight_initial):
    
    # Récupération des paramètres
    fc_default, geom, aero = get_default_inputs()
    sref = geom.S_ref
    
    # Paramètres d'atterrissage / descente
    h_flare_start = 50.0        
    gamma_approach_deg = -3.0   
    IDLE_POWER_SETTING = 0.05 
    h_descent_end = 1000.0
    
    # -------------------------------------------------------------------
    # INITIALISATION
    # -------------------------------------------------------------------
    weight = TO_weight_initial
    weight_start = TO_weight_initial
    
    # Dictionnaires pour le résumé
    t_dict = {}
    d_dict = {}
    w_dict = {} # On stockera les poids ici
    
    w_dict["start"] = weight_start

    # ==========================
    # 1) PHASE DE DÉCOLLAGE (FIXE)
    # ==========================
    dist_TO_ft, t_TO_s, h_hist = groundrun(
        aero.v_trans_kts * 1.6878, weight, aero.alpha_trans_deg, aero.alpha_ini_deg, h_airport, dT_isa, 1.0, aero.cl_max_15
    )
    weight_decollage = h_hist['weight'][-1]
    weight = weight_decollage

    t_dict["takeoff"] = t_TO_s / 60.0
    d_dict["takeoff"] = dist_TO_ft / 6076.12
    w_dict["after_takeoff"] = weight_decollage

    # ==========================
    # 2) ACCÉLÉRATION (FIXE)
    # ==========================
    V_TAS_ft_s_TO = h_hist['v'][-1]
    V_TAS_kts_TO  = V_TAS_ft_s_TO / 1.6878
    atm_decollage   = atmosphere(h_airport + 5.0, dT_isa, 'delta_isa')
    speed_decollage = vitesses(atm_decollage, V_TAS_kts_TO, 'vraie', weight)
    CAS_kts_TO      = speed_decollage['vitesses avion kts']['vitesse calibree']
    CAS_climb_kts = VY

    dt_accel_min, dist_accel_NM, dFuel_accel = acceleration(
        CAS_kts_TO, CAS_climb_kts, h_airport + 5.0, dT_isa, weight, sref, 1.0
    )
    weight -= dFuel_accel
    weight_after_accel = weight

    t_dict["accel"] = dt_accel_min
    d_dict["accel"] = dist_accel_NM
    w_dict["after_accel"] = weight_after_accel

    # ==========================
    # 3) MONTÉE (FIXE)
    # ==========================
    h_start_climb = 5.0       
    h_end_climb   = h_cruise  

    t_climb_min, d_climb_NM, Fuel_tot_climb, weight_top_climb = montee(
        hpi=h_start_climb, hpf=h_end_climb, dT_isa=dT_isa, CAS_kts=CAS_climb_kts,
        weight_initial=weight, sref=sref, power_setting=0.75
    )
    weight = weight_top_climb
    
    t_dict["climb"] = t_climb_min
    d_dict["climb"] = d_climb_NM
    w_dict["top_climb"] = weight_top_climb

    # On a maintenant le poids de début de croisière
    Weight_start_cruise = weight_top_climb

    # ==========================
    # BOUCLE D'ITÉRATION POUR LA CROISIÈRE ET DESCENTE
    # ==========================
    # Le temps de descente dépend du poids au début de la descente (qui dépend de la conso croisière),
    # et le temps de croisière dépend du temps total restant (qui dépend du temps de descente).
    
    # Initialisation de la boucle
    # On fait une première estimation du temps de descente (ex: 15 min)
    t_descent_est = 15.0 
    t_approach_est = 5.0 # Souvent assez fixe mais dépend un peu du poids
    t_landing_est = 1.0
    
    error_t = 1000.0
    tol = 0.01 # minutes
    max_iter = 10
    iter_count = 0
    
    # Phases fixes déjà calculées
    t_fixed_pre_cruise = t_dict["takeoff"] + t_dict["accel"] + t_dict["climb"]
    
    # Variables de résultat de la boucle
    t_cruise = 0.0
    t_descent = 0.0
    t_approach = 0.0
    t_landing = 0.0
    
    d_cruise = 0.0
    d_descent = 0.0
    d_approach = 0.0
    d_landing = 0.0
    
    w_top_descent = 0.0
    w_final_1000ft = 0.0
    w_touchdown = 0.0
    w_final = 0.0
    
    history_roll = None

    while abs(error_t) > tol and iter_count < max_iter:
        iter_count += 1
        
        # 1. Calcul du temps dispo pour la croisière
        # On estime les temps post-croisière par les valeurs de l'itération précédente
        t_post_cruise_est = t_descent_est + t_approach_est + t_landing_est
        
        available_cruise_time = total_time_imposed_min - t_fixed_pre_cruise - t_post_cruise_est
        
        if available_cruise_time < 0:
            print(f"Warning: Le temps imposé ({total_time_imposed_min} min) est trop court pour effectuer les phases fixes.")
            available_cruise_time = 0.0 # On simule une croisière nulle
        
        t_cruise = available_cruise_time
        
        # 2. Calcul de la conso croisière pour ce temps
        if t_cruise > 0:
            fuel_cruise_lb, ff_cruise_avg = compute_fuel_burned_for_time(
                t_cruise, Weight_start_cruise, V_cruise_CAS, h_cruise, dT_isa
            )
            # Calcul distance croisière approx (Vitesse moyenne TAS * temps)
            # Pour plus de précision on pourrait le récupérer de compute_fuel_burned_for_time si on modifiait encore,
            # mais on peut le recalculer ici.
            # On utilise le code existant compute_cruise_condition pour avoir la TAS moyenne
            weight_mid = Weight_start_cruise - 0.5 * fuel_cruise_lb
            cg_mid = compute_cg_mac(weight_mid)
            res_mid = compute_cruise_condition(V_cruise_CAS, h_cruise, dT_isa, weight_mid, cg_mid)
            tas_mid_kts = res_mid['TAS_kts']
            d_cruise_nm = (tas_mid_kts * t_cruise) / 60.0
        else:
            fuel_cruise_lb = 0.0
            d_cruise_nm = 0.0
        
        w_top_descent = Weight_start_cruise - fuel_cruise_lb
        
        # 3. Calcul Descente (avec le nouveau poids)
        # VERIFICATION SECURITY : Si le poids est sous OEW+Payload (plus de fuel), on simule la descente "à vide" (OEW+Payload)
        # pour éviter les erreurs de physique (sqrt negatif etc), tout en comptant le déficit de fuel.
        
        limit_empty_weight = aero.OEW + aero.PAYLOAD
        
        if w_top_descent < limit_empty_weight:
            w_physics_descent = limit_empty_weight
        else:
            w_physics_descent = w_top_descent
            
        t_descent, d_descent, fuel_desc, w_final_1000ft_val = descente(
            hpi=h_cruise, hpf=h_descent_end, dT_isa=dT_isa, CAS_kts=VY,
            weight_initial=w_physics_descent, sref=sref, power_setting=IDLE_POWER_SETTING
        )
        
        # Ajustement du poids de sortie réel (avec déficit)
        if w_top_descent < limit_empty_weight:
             w_final_1000ft_val = w_top_descent - fuel_desc # On continue de creuser le déficit

        
        # 4. Calcul Approche
        if w_final_1000ft_val < limit_empty_weight:
            w_physics_approach = limit_empty_weight
        else:
            w_physics_approach = w_final_1000ft_val
            
        V_app_kts = VY 
        t_app, d_app, fuel_app, w_td_val, V_td_kts = approach_and_flare(
            hpi=h_descent_end, hpf=h_airport, h_flare=h_flare_start, dT_isa=dT_isa,
            V_app_kts=V_app_kts, weight_initial=w_physics_approach, sref=sref, 
            power_setting=IDLE_POWER_SETTING, gamma_deg=gamma_approach_deg
        )
        
        if w_final_1000ft_val < limit_empty_weight:
            w_td_val = w_final_1000ft_val - fuel_app

        # 5. Calcul Landing Roll
        if w_td_val < limit_empty_weight:
            w_physics_landing = limit_empty_weight
        else:
            w_physics_landing = w_td_val
        
        V_td_ft_s = V_td_kts * 1.6878
        dist_landing_roll_ft, t_land_s, hist_roll = landing_groundroll(
            weight_initial=w_physics_landing, h_ft=h_airport, dT_isa=dT_isa, V_td_ft_s=V_td_ft_s, 
            power_setting=IDLE_POWER_SETTING, 
            alpha_td_deg=4,
            mu_brake=0.4,
            mu_rolling=aero.mu_TO
        )
        t_land = t_land_s / 60.0
        d_land_nm = dist_landing_roll_ft / 6076.12
        
        # Le poids final returned par landing includes fuel burnt during roll
        fuel_roll = w_physics_landing - hist_roll['weight'][-1]
        
        w_final_val = w_td_val - fuel_roll
        
        # 6. Mise à jour des estimateurs pour la prochaine boucle
        # La boucle cherche à stabiliser t_descent + t_app + t_land
        # En fait, t_cruise dépend de ces temps.
        # Si t_post_cruise change, t_cruise change, donc w_top_descent change, donc t_post_cruise change.
        
        t_post_cruise_new = t_descent + t_app + t_land
        error_t = t_post_cruise_new - t_post_cruise_est
        
        # Mise à jour des estimations
        t_descent_est = t_descent
        t_approach_est = t_app
        t_landing_est = t_land
        
        # Sauvegarde des résultats courants
        d_cruise = d_cruise_nm
        d_descent = d_descent
        d_approach = d_app
        d_landing = d_land_nm
        
        w_final_1000ft = w_final_1000ft_val
        w_touchdown = w_td_val
        w_final = w_final_val
        history_roll = hist_roll

    # Fin de boucle
    
    # Remplissage des résultats finaux
    t_dict["cruise"] = t_cruise
    d_dict["cruise"] = d_cruise
    w_dict["top_descent"] = w_top_descent
    
    t_dict["descent"] = t_descent_est
    d_dict["descent"] = d_descent
    w_dict["final_1000ft"] = w_final_1000ft
    
    t_dict["approach"] = t_approach_est
    d_dict["approach"] = d_approach
    w_dict["touchdown"] = w_touchdown
    
    t_dict["groundroll"] = t_landing_est
    d_dict["groundroll"] = d_landing
    w_dict["final"] = w_final
    
    # Calcul du temps total réel simulé
    total_time_simulated = sum(t_dict.values())
    total_dist_simulated = sum(d_dict.values())

    print("\n========== FLIGHT PROFILE SUMMARY ==========")
    print(f"Total time         : {total_time_simulated:.2f} min ({total_time_simulated/60:.2f} h)")
    print(f"Total distance     : {total_dist_simulated:.2f} NM")
    print("--------------------------------------")
    print("WEIGHTS BY PHASE :")
    print(f"  Start mission        : {weight_start:.2f} lb")
    print(f"  After takeoff      : {weight_decollage:.2f} lb")
    print(f"  After acceleration   : {weight_after_accel:.2f} lb")
    print(f"  Top of climb     : {weight_top_climb:.2f} lb")
    print(f"  Start descent    : {w_top_descent:.2f} lb")
    print(f"  Final (1000 ft)      : {w_final_1000ft:.2f} lb")
    print(f"  Touchdown : {w_touchdown:.2f} lb")
    print(f"  Final Mission (Arrêt): {w_final:.2f} lb")
    print("--------------------------------------")
    print("TIME BY PHASE :")
    print(f"  Takeoff      : {t_dict['takeoff']:.2f} min")
    print(f"  Acceleration   : {t_dict['accel']:.2f} min")
    print(f"  Climb         : {t_dict['climb']:.2f} min")
    print(f"  Cruise      : {t_dict['cruise']:.2f} min")
    print(f"  Descent       : {t_dict['descent']:.2f} min")
    print(f"  Approach/Flare : {t_dict['approach']:.2f} min")
    print(f"  Ground roll  : {t_dict['groundroll']:.2f} min")
    
    results = {
        "weights_lb": w_dict,
        "times_min": t_dict,
        "ranges_NM": d_dict,
        "landing_history": history_roll,
        "total_time_simulated": total_time_simulated,
        "total_dist_simulated": total_dist_simulated
    }
    
    return results

# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    
    # Inputs utilisateur (Simulés ici)
    # Exemple: On impose 3 heures de vol (180 min)
    TIME_IMPOSED_MIN = 25 # <--- MODIFIER ICI
    
    # Paramètres de base (idem Flight_phases.py)
    fc_default, geom, aero = get_default_inputs()
    
    h_cruise = MISSION_HEIGHT_FT
    h_airport = 0
    V_cruise_CAS = 108
    VY = 75
    TO_weight = aero.MAX_TO # On part à pleine charge
    
    # Quantité de fuel "Max" disponible (Fuel Load + Reserve)
    # Note: Dans aero.MAX_TO, le fuel est inclus.
    # Fuel Total à bord initialement = aero.FUEL_LOAD + aero.RESERVE
    TOTAL_FUEL_AVAILABLE_LB = aero.FUEL_LOAD + aero.RESERVE
    OEW_PAYLOAD = aero.OEW + aero.PAYLOAD
    
    print(f"--- DÉTAILS MISSION ---")
    print(f"Temps de vol imposé : {TIME_IMPOSED_MIN} min")
    print(f"Poids au décollage  : {TO_weight} lb")
    print(f"Fuel Total à bord   : {TOTAL_FUEL_AVAILABLE_LB} lb")
    print(f"Zero Fuel Weight    : {OEW_PAYLOAD} lb")
    
    # Configurations à tester : (dT_isa, Label, Style)
    configs = [
        (0,   "ISA (Standard)", "k-",  2.0),
        (-15, "ISA - 15°C",    "C0--", 1.5),
        (15,  "ISA + 15°C",    "C3--", 1.5)
    ]
    
    plt.figure(figsize=(10, 6))

    for dT_Isa, label, style, width in configs:
        print(f"\nRunning simulation for {label}...")
        # Exécution de la simulation
        res = FLIGHT_PHASES_TIME_IMPOSED(TIME_IMPOSED_MIN, h_cruise, h_airport, dT_Isa, VY, V_cruise_CAS, TO_weight)
        
        t_dict = res["times_min"]
        
        # Construction des points temporels et d'altitude pour le plot
        # On suppose les altitudes suivantes pour les phases :
        # - Start: h_airport
        # - Takeoff: h_airport
        # - Accel: h_airport (approx, technically +50ft)
        # - Climb: h_cruise
        # - Cruise: h_cruise
        # - Descent: 1000 ft (h_descent_end)
        # - Approach: h_airport
        # - Groundroll: h_airport
        
        # Phases ordonnées
        phases = ["takeoff", "accel", "climb", "cruise", "descent", "approach", "groundroll"]
        
        # Altitudes de FIN de phase
        h_end_phases = {
            "takeoff": h_airport,
            "accel": h_airport, # Approx
            "climb": h_cruise,
            "cruise": h_cruise,
            "descent": 1000.0, # h_descent_end defined inside function
            "approach": h_airport,
            "groundroll": h_airport
        }
        
        times = [0.0]
        altitudes = [h_airport]
        
        current_time = 0.0
        
        for phase in phases:
            dt = t_dict[phase]
            h_target = h_end_phases[phase]
            
            # Ajout du point de fin de phase
            current_time += dt
            times.append(current_time)
            altitudes.append(h_target)

        # Plot de la courbe
        plt.plot(times, altitudes, style, linewidth=width, label=label)
        
        # Affichage résultats textuels pour cette config
        w_final = res["weights_lb"]["final"]
        fuel_remaining = w_final - OEW_PAYLOAD
        print(f"  -> Temps total: {res['total_time_simulated']:.2f} min")
        print(f"  -> Fuel Restant: {fuel_remaining:.2f} lb")

    # Mise en forme du graphique
    plt.title(f"Mission Profile Sensitivity to Temperature\n(Target Cruise Altitude: {h_cruise:.0f} ft)")
    plt.xlabel("Flight Time [min]")
    plt.ylabel("Altitude [ft]")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.ylim(bottom=0, top=h_cruise * 1.2) # Marge au dessus
    
    plt.tight_layout()
    plt.savefig("time_imposed_profile_sensitivity.png")
    print("\nGraph saved as 'time_imposed_profile_sensitivity.png'")
    plt.show()
