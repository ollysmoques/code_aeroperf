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
from Flight_phases import save_run_parameters
from config_loader import get as cfg
import os

# Output directory for saved figures and reports
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    t_climb_min, d_climb_NM, Fuel_tot_climb, weight_top_climb, history_climb = montee(
        hpi=h_start_climb, hpf=h_end_climb, dT_isa=dT_isa, CAS_kts=CAS_climb_kts,
        weight_initial=weight, sref=sref, power_setting=cfg("CLIMB_POWER_SETTING", 0.90)
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
            
            # Collect forces for cruise phase
            from total_drag import drag_total
            cl_cruise, d_total_cruise, q_cruise, alpha_cruise, breakdown_cruise = drag_total(
                h_cruise, dT_isa, tas_mid_kts, weight_mid, cg_mid, res_mid['Thrust_required'], 'cruise', return_breakdown=True
            )
            # Extract wing and empennage forces
            wing_lift_cruise = weight_mid / (q_cruise * sref)  # Total lift ≈ weight in cruise
            wing_drag_cruise = breakdown_cruise.get('induced_wing_from_equil', 0) if breakdown_cruise else 0
            empennage_lift_cruise = 0  # Not directly available from breakdown
            empennage_drag_cruise = breakdown_cruise.get('induced_tail', 0) if breakdown_cruise else 0
            empennage_lift_cruise = breakdown_cruise.get('induced_tail', 0) if breakdown_cruise else 0  # This is actually drag, lift not directly available
            empennage_drag_cruise = breakdown_cruise.get('induced_tail', 0) if breakdown_cruise else 0
        else:
            fuel_cruise_lb = 0.0
            d_cruise_nm = 0.0
            wing_lift_cruise = 0
            wing_drag_cruise = 0
            empennage_lift_cruise = 0
            empennage_drag_cruise = 0
        
        w_top_descent = Weight_start_cruise - fuel_cruise_lb
        
        # 3. Calcul Descente (avec le nouveau poids)
        # VERIFICATION SECURITY : Si le poids est sous OEW+Payload (plus de fuel), on simule la descente "à vide" (OEW+Payload)
        # pour éviter les erreurs de physique (sqrt negatif etc), tout en comptant le déficit de fuel.
        
        limit_empty_weight = aero.OEW + aero.PAYLOAD
        
        if w_top_descent < limit_empty_weight:
            w_physics_descent = limit_empty_weight
        else:
            w_physics_descent = w_top_descent
            
        t_descent, d_descent, fuel_desc, w_final_1000ft_val, history_descent = descente(
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
        t_app, d_app, fuel_app, w_td_val, V_td_kts, history_approach = approach_and_flare(
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
    
    # Collect forces for each phase (simplified - only cruise has detailed forces)
    # Initialize default forces
    wing_lift_cruise = 0
    wing_drag_cruise = 0
    empennage_lift_cruise = 0
    empennage_drag_cruise = 0
    
    forces_dict = {
        "takeoff": {"wing_lift": weight_decollage, "wing_drag": 0, "empennage_lift": 0, "empennage_drag": 0},
        "accel": {"wing_lift": weight_after_accel, "wing_drag": 0, "empennage_lift": 0, "empennage_drag": 0},
        "climb": {"wing_lift": weight_top_climb, "wing_drag": 0, "empennage_lift": 0, "empennage_drag": 0},
        "cruise": {"wing_lift": wing_lift_cruise, "wing_drag": wing_drag_cruise, "empennage_lift": empennage_lift_cruise, "empennage_drag": empennage_drag_cruise},
        "descent": {"wing_lift": w_top_descent, "wing_drag": 0, "empennage_lift": 0, "empennage_drag": 0},
        "approach": {"wing_lift": w_touchdown, "wing_drag": 0, "empennage_lift": 0, "empennage_drag": 0},
        "groundroll": {"wing_lift": w_final, "wing_drag": 0, "empennage_lift": 0, "empennage_drag": 0}
    }
    
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
        "forces_lbf": forces_dict,
        "landing_history": history_roll,
        "climb_history": history_climb if 'history_climb' in locals() else None,
        "descent_history": history_descent if 'history_descent' in locals() else None,
        "approach_history": history_approach if 'history_approach' in locals() else None,
        "total_time_simulated": total_time_simulated,
        "total_dist_simulated": total_dist_simulated,
        "speeds_kts": {
            "takeoff_cas": CAS_kts_TO,
            "cruise_cas": V_cruise_CAS,
            "approach_cas": V_app_kts,
            "touchdown_cas": V_td_kts
        }
    }
    
    return results


def plot_mission_fuel_analysis(res):
    w = res["weights_lb"]
    
    # Calcul du fuel consommé par phase (lb)
    fuel_phases = {
        "Takeoff":      w["start"] - w["after_takeoff"],
        "Acceleration": w["after_takeoff"] - w["after_accel"],
        "Climb":        w["after_accel"] - w["top_climb"],
        "Cruise":       w["top_climb"] - w["top_descent"],
        "Descent":      w["top_descent"] - w["final_1000ft"],
        "Approach":     w["final_1000ft"] - w["touchdown"],
        "Ground Roll":  w["touchdown"] - w["final"]
    }

    phases = list(fuel_phases.keys())
    values = list(fuel_phases.values())
    total_fuel = sum(values)

    # Graphique 1 : Histogramme de consommation
    plt.figure(figsize=(12, 6))
    bars = plt.bar(phases, values, color='royalblue', edgecolor='black', alpha=0.8)
    plt.ylabel('Fuel Burned [lb]', fontweight='bold')
    plt.title(f'Fuel Consumption per Flight Phase (Total: {total_fuel:.2f} lb)', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), 
                 f'{yval:.2f} lb', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fuel_burn_per_phase.png"))
    plt.close()

    # Graphique 2 : Répartition en pourcentage
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=phases, autopct='%1.1f%%', startangle=140, 
            colors=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'],
            explode=[0.05 if v == max(values) else 0 for v in values]) # Met en évidence la phase max
    plt.title('Fuel Burn Distribution', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "fuel_distribution_pie.png"))
    plt.close()

def plot_fuel_efficiency(res):
    """
    Plots Fuel Flow (lb/hr) and Specific Range (NM/lb) for each phase.
    """
    w = res["weights_lb"]
    t = res["times_min"]
    d = res["ranges_NM"]
    
    # Phases to analyze
    phases = ["takeoff", "accel", "climb", "cruise", "descent", "approach", "groundroll"]
    phase_labels = ["Takeoff", "Accel", "Climb", "Cruise", "Descent", "Approach", "Roll"]
    
    fuel_flow = []
    spec_range = []
    
    # Mapping weight keys to calculate delta fuel per phase
    # Phases: TO, Accel, Climb, Cruise, Descent, App, Roll
    # Weights keys: start, after_takeoff, after_accel, top_climb, top_descent, final_1000ft, touchdown, final
    
    w_keys_seq = ["start", "after_takeoff", "after_accel", "top_climb", "top_descent", "final_1000ft", "touchdown", "final"]
    
    for i, phase in enumerate(phases):
        # Fuel consummed
        w_start = w[w_keys_seq[i]]
        w_end   = w[w_keys_seq[i+1]]
        delta_fuel = w_start - w_end
        
        duration_min = t[phase]
        dist_nm = d[phase]
        
        # Avoid division by zero
        if duration_min > 0.01:
            ff = delta_fuel / (duration_min / 60.0) # lb/hr
        else:
            ff = 0.0
            
        if delta_fuel > 0.01:
            sr = dist_nm / delta_fuel # NM/lb
        else:
            sr = 0.0
            
        fuel_flow.append(ff)
        spec_range.append(sr)
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Fuel Flow
    ax1.bar(phase_labels, fuel_flow, color='orange', alpha=0.7, edgecolor='black')
    ax1.set_ylabel("Fuel Flow [lb/hr]", fontweight='bold')
    ax1.set_title("Average Fuel Flow per Phase", fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(fuel_flow):
        ax1.text(i, v + max(fuel_flow)*0.02, f"{v:.0f}", ha='center', va='bottom', fontsize=9)

    # 2. Specific Range
    ax2.bar(phase_labels, spec_range, color='green', alpha=0.7, edgecolor='black')
    ax2.set_ylabel("Specific Range [NM/lb]", fontweight='bold')
    ax2.set_title("Specific Range (Efficiency) per Phase", fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(spec_range):
        ax2.text(i, v + max(spec_range)*0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        
    plt.xlabel("Flight Phase")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fuel_efficiency_analysis.png"))
    plt.close()
    print("Graph 'fuel_efficiency_analysis.png' generated.")

def plot_mission_history(res):
    """
    Plots Weight history and CG shift history over time.
    """
    w = res["weights_lb"]
    t = res["times_min"]
    
    # Reconstruct time and weight history
    # Sequence matching FLIGHT_PHASES output structure
    # t=0 -> start
    # t=t_to -> after_takeoff
    # ...
    
    phases = ["takeoff", "accel", "climb", "cruise", "descent", "approach", "groundroll"]
    w_keys = ["start", "after_takeoff", "after_accel", "top_climb", "top_descent", "final_1000ft", "touchdown", "final"]
    
    times = [0.0]
    weights = [w["start"]]
    
    current_time = 0.0
    for i, phase in enumerate(phases):
        dt = t[phase]
        current_time += dt
        times.append(current_time)
        weights.append(w[w_keys[i+1]])
        
    # Calculate CG for each weight point
    cg_pos = [compute_cg_mac(weight) for weight in weights]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Weight History
    ax1.plot(times, weights, 'b-o', linewidth=2)
    ax1.set_ylabel("Aircraft Weight [lb]", fontweight='bold', color='blue')
    ax1.set_title("Mission Weight History", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Annotate start and end
    ax1.text(times[0], weights[0], f"{weights[0]:.0f}", ha='left', va='bottom', color='blue')
    ax1.text(times[-1], weights[-1], f"{weights[-1]:.0f}", ha='right', va='top', color='blue')

    # 2. CG History
    ax2.plot(times, cg_pos, 'r-o', linewidth=2)
    ax2.set_ylabel("CG Position [% MAC]", fontweight='bold', color='red')
    ax2.set_title("Center of Gravity Shift", fontweight='bold')
    ax2.set_xlabel("Time [min]", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Add Limits if available
    try:
        _, _, cg_fwd, cg_aft = get_cg_parameters()
        ax2.axhline(cg_fwd, color='k', linestyle='--', label='Fwd Limit')
        ax2.axhline(cg_aft, color='k', linestyle='--', label='Aft Limit')
        ax2.legend()
    except:
        pass

    for i, v in enumerate(cg_pos):
        # Annotate simplified phases on CG plot
        if i in [0, 3, 4, 7]: # Start, Top Climb, Top Descent, End
             ax2.text(times[i], v, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mission_weight_cg_history.png"))
    plt.close()
    print("Graph 'mission_weight_cg_history.png' generated.")


def plot_forces_analysis(res):
    """
    Plots forces on wing and empenage for each flight phase.
    Shows lift and drag forces separately for wing and empenage.
    """
    forces = res["forces_lbf"]
    
    phases = list(forces.keys())
    phase_labels = ["Takeoff", "Accel", "Climb", "Cruise", "Descent", "Approach", "Ground Roll"]
    
    wing_lift = [forces[phase]["wing_lift"] for phase in phases]
    wing_drag = [forces[phase]["wing_drag"] for phase in phases]
    empennage_lift = [forces[phase]["empennage_lift"] for phase in phases]
    empennage_drag = [forces[phase]["empennage_drag"] for phase in phases]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Wing Lift
    bars1 = ax1.bar(phase_labels, wing_lift, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.set_ylabel("Wing Lift [lbf]", fontweight='bold')
    ax1.set_title("Wing Lift by Flight Phase", fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars1, wing_lift):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(wing_lift)*0.01, 
                     f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Wing Drag
    bars2 = ax2.bar(phase_labels, wing_drag, color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.set_ylabel("Wing Drag [lbf]", fontweight='bold')
    ax2.set_title("Wing Drag by Flight Phase", fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars2, wing_drag):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(wing_drag + [1])*0.01, 
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Empennage Lift
    bars3 = ax3.bar(phase_labels, empennage_lift, color='lightgreen', alpha=0.8, edgecolor='black')
    ax3.set_ylabel("Empennage Lift [lbf]", fontweight='bold')
    ax3.set_title("Empennage Lift by Flight Phase", fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars3, empennage_lift):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(empennage_lift + [1])*0.01, 
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Empennage Drag
    bars4 = ax4.bar(phase_labels, empennage_drag, color='orange', alpha=0.8, edgecolor='black')
    ax4.set_ylabel("Empennage Drag [lbf]", fontweight='bold')
    ax4.set_title("Empennage Drag by Flight Phase", fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars4, empennage_drag):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(empennage_drag + [1])*0.01, 
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "forces_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Graph 'forces_analysis.png' generated.")


def plot_stall_warnings(res):
    """
    Plots stall margin vs altitude for climb, descent and approach.
    Highlights points where a stall warning was triggered.
    """
    histories = {
        'Climb': res.get('climb_history'),
        'Descent': res.get('descent_history'),
        'Approach': res.get('approach_history')
    }

    plt.figure(figsize=(8, 6))
    colors = {'Climb': 'blue', 'Descent': 'orange', 'Approach': 'green'}
    any_warning = False

    for phase, hist in histories.items():
        if not hist:
            continue

        alt = np.array(hist.get('altitude', []))
        margin = np.array(hist.get('stall_margin_pct', []))
        warn = np.array(hist.get('stall_warning', []), dtype=bool)

        if alt.size == 0:
            continue

        # Scatter all points for the phase
        plt.scatter(margin, alt, label=phase, color=colors.get(phase, 'gray'), alpha=0.6)

        # Overlay stall-warning points
        if warn.any():
            any_warning = True
            plt.scatter(margin[warn], alt[warn], facecolors='none', edgecolors='r',
                        s=80, linewidths=1.5, marker='X', label=f"{phase} - Stall Risk")

    # Visual threshold line (15% margin)
    plt.axvline(15.0, color='k', linestyle='--', linewidth=0.8, label='Warning threshold (15%)')
    plt.gca().invert_yaxis()
    plt.xlabel('Stall Margin [%]')
    plt.ylabel('Altitude [ft]')
    plt.title('Stall Margin by Phase — markers show stall-risk points')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stall_warnings_by_phase.png'))
    plt.close()

    if any_warning:
        print("[WARNING] One or more flight phases show stall-risk points. See 'stall_warnings_by_phase.png'.")
    else:
        print("No stall-risk points detected (stall margin > 15% everywhere). Figure saved as 'stall_warnings_by_phase.png'.")

# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    
    # Inputs utilisateur (Simulés ici)
    # Exemple: On impose 3 heures de vol (180 min)
    TIME_IMPOSED_MIN = cfg("TIME_IMPOSED_MIN", 25)
    
    # Paramètres de base (idem Flight_phases.py)
    fc_default, geom, aero = get_default_inputs()
    
    h_cruise = MISSION_HEIGHT_FT
    h_airport = cfg("h_airport", 0)
    V_cruise_CAS = cfg("V_cruise_CAS", 108)
    VY = cfg("VY", 81)
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

    res_std = None

    for dT_Isa, label, style, width in configs:
        print(f"\nRunning simulation for {label}...")
        # Exécution de la simulation
        res = FLIGHT_PHASES_TIME_IMPOSED(TIME_IMPOSED_MIN, h_cruise, h_airport, dT_Isa, VY, V_cruise_CAS, TO_weight)
        
        if dT_Isa == 0:
            res_std = res
            
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

    # Use Standard ISA results for detailed analysis if available, otherwise last run
    res_analysis = res_std if res_std else res
    
    plot_mission_fuel_analysis(res_analysis)
    plot_fuel_efficiency(res_analysis)
    plot_mission_history(res_analysis)
    plot_forces_analysis(res_analysis)
    plot_stall_warnings(res_analysis)
    print("Graphs 'fuel_burn_per_phase.png' and 'fuel_distribution_pie.png' generated.")

    # Mise en forme du graphique
    plt.title(f"Mission Profile Sensitivity to Temperature\n(Target Cruise Altitude: {h_cruise:.0f} ft)")
    plt.xlabel("Flight Time [min]")
    plt.ylabel("Altitude [ft]")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.ylim(bottom=0, top=h_cruise * 1.2) # Marge au dessus
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "time_imposed_profile_sensitivity.png"))
    print("\nGraph saved as 'time_imposed_profile_sensitivity.png'")
    plt.show()
    
    # SAUVEGARDE DES PARAMÈTRES
    if res_std:
        try:
             # Création du dictionnaire des inputs de mission
            mission_inputs = {
                "h_cruise_ft": h_cruise,
                "h_airport_ft": h_airport,
                "dT_isa_C": 0, # Pour le cas standard
                "VY_kts": VY,
                "V_cruise_CAS_kts": V_cruise_CAS,
                "V_r_kts": res_std["speeds_kts"]["rotation_cas"],
                "V_lof_kts": res_std["speeds_kts"]["liftoff_cas"],
                "V_stall_kts": res_std["speeds_kts"]["stall_cas"],
                "TO_weight_lb": TO_weight,
                "TIME_IMPOSED_MIN": TIME_IMPOSED_MIN,
                "Calculated_Range_NM": res_std["ranges_NM"]["cruise"], 
                "V_Takeoff_CAS_kts": res_std["speeds_kts"]["takeoff_cas"],
                "V_Cruise_CAS_kts": res_std["speeds_kts"]["cruise_cas"],
                "V_Approach_CAS_kts": res_std["speeds_kts"]["approach_cas"],
                "V_Touchdown_CAS_kts": res_std["speeds_kts"]["touchdown_cas"],
            }
        
            # Création du dictionnaire des objets avions
            aircraft_objects = {
                "fc": fc_default,
                "geom": geom,
                "aero": aero
            }
            
            output_path = os.path.join(OUTPUT_DIR, "simulation_parameters.txt")
            save_run_parameters(output_path, mission_inputs, aircraft_objects)
            
        except Exception as e:
            print(f"[WARNING] Could not save parameters report: {e}")
