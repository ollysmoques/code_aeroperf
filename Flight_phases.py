import numpy as np
import matplotlib.pyplot as plt
from atmosphere import atmosphere, vitesses
from total_drag import drag_total
from Thrust_data import thrust_sw400pro_ft_lbf
from Aircraft_data import get_default_inputs
from Cruise_Condition import compute_cruise_range_time
from ROC import montee, descente, ROC, acceleration , find_initial_weight_for_descent
from take_off_run import groundrun 
from cg_shift import *
from landing_run import *  
from landing_phases import * 
from Mission_parameters import MISSION_HEIGHT_FT
import pprint
from dataclasses import asdict

def save_run_parameters(filename, run_inputs, aircraft_data):
    """
    Saves the simulation parameters to a text file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("========================================================\n")
        f.write("       SIMULATION PARAMETERS & INPUT VARIABLES\n")
        f.write("========================================================\n\n")
        
        f.write("--- 1. MISSION INPUTS ---\n")
        for key, value in run_inputs.items():
            f.write(f"{key:<30}: {value}\n")
        f.write("\n")

        f.write("--- 2. AIRCRAFT DEFINITION (aero) ---\n")
        # aero object is a dataclass
        if hasattr(aircraft_data['aero'], '__dataclass_fields__'):
             pprint.pprint(asdict(aircraft_data['aero']), stream=f, indent=4)
        else:
             f.write(str(aircraft_data['aero']))
        f.write("\n\n")

        f.write("--- 3. GEOMETRY DEFINITION (geom) ---\n")
        if hasattr(aircraft_data['geom'], '__dataclass_fields__'):
             pprint.pprint(asdict(aircraft_data['geom']), stream=f, indent=4)
        else:
             f.write(str(aircraft_data['geom']))
        f.write("\n\n")
        
        f.write("--- 4. FLIGHT CONDITIONS (default) ---\n")
        if hasattr(aircraft_data['fc'], '__dataclass_fields__'):
             pprint.pprint(asdict(aircraft_data['fc']), stream=f, indent=4)
        else:
             f.write(str(aircraft_data['fc']))
        f.write("\n\n")

        f.write("========================================================\n")
        f.write("END OF REPORT\n")
    print(f"\n[INFO] Simulation parameters saved to: {filename}")


fc_default, geom, aero = get_default_inputs()



v_trans   = aero.v_trans_kts * 1.6878     
weight_0  = aero.MAX_TO
alpha_ini   = aero.alpha_ini_deg
alpha_trans = aero.alpha_trans_deg
alpha_rot   = aero.alpha_rot_deg
mu          = aero.mu_TO
sref        = geom.S_ref                
CL_max      = aero.CL_max_TO

VY = 75
h_cruise = MISSION_HEIGHT_FT
dT_Isa = 0
h_airport = 0
V_cruise_CAS = 108
Weight_Reserve = aero.OEW + aero.PAYLOAD + aero.RESERVE
TO_weight = weight_0



def FLIGHT_PHASES(h_cruise, h_airport, dT_isa, VY, V_cruise_CAS, TO_weight, Weight_Reserve):
    
    # Récupération des paramètres (requis pour les nouvelles phases)
    fc_default, geom, aero = get_default_inputs()
    sref = geom.S_ref
    
    # Paramètres d'atterrissage
    h_flare_start = 50.0        # Début de l'arrondi (screen height)
    gamma_approach_deg = -3.0   # Pente d'approche standard (Glide Slope)
    IDLE_POWER_SETTING = 0.05 # Assumé dans AeroParams
    
    # -------------------------------------------------------------------
    # INITIALISATION BILAN GLOBAL
    # -------------------------------------------------------------------
    temps_total_min    = 0.0   # [min]
    distance_total_NM  = 0.0   # [NM]
    weight             = TO_weight
    
    # Dictionnaires pour le résumé
    t_dict = {}
    d_dict = {}

    weight_start = TO_weight

    # ==========================
    # 1) PHASE DE DÉCOLLAGE
    # ==========================
    dist_TO_ft, t_TO_s, h_hist = groundrun(
        aero.v_trans_kts * 1.6878, weight, aero.alpha_trans_deg, aero.alpha_ini_deg, h_airport, dT_isa, 1.0, aero.CL_max_TO
    )

    weight_decollage = h_hist['weight'][-1]
    weight = weight_decollage

    dist_TO_NM = dist_TO_ft / 6076.12
    t_TO_min   = t_TO_s / 60.0

    temps_total_min   += t_TO_min
    distance_total_NM += dist_TO_NM
    
    t_dict["takeoff"] = t_TO_min
    d_dict["takeoff"] = dist_TO_NM


    # ==========================
    # 2) ACCÉLÉRATION JUSQU'À CAS DE MONTÉE
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

    temps_total_min   += dt_accel_min
    distance_total_NM += dist_accel_NM
    weight           -= dFuel_accel
    weight_after_accel = weight

    t_dict["accel"] = dt_accel_min
    d_dict["accel"] = dist_accel_NM

    # ==========================
    # 3) MONTÉE
    # ==========================
    h_start_climb = 5.0       # Début de montée après accélération
    h_end_climb   = h_cruise  

    t_climb_min, d_climb_NM, Fuel_tot_climb, weight_top_climb = montee(
        hpi=h_start_climb, hpf=h_end_climb, dT_isa=dT_isa, CAS_kts=CAS_climb_kts,
        weight_initial=weight, sref=sref, power_setting=0.75
    )

    temps_total_min   += t_climb_min
    distance_total_NM += d_climb_NM
    weight            = weight_top_climb

    t_dict["climb"] = t_climb_min
    d_dict["climb"] = d_climb_NM

    # ==========================
    # 4) CROISIÈRE
    # ==========================
    Weight_target_at_1000ft = Weight_Reserve # Poids au niveau de la réserve à 1000ft

    Weight_top_descent = find_initial_weight_for_descent(
        hpi=h_cruise, hpf=1000.0, dT_isa=dT_isa, CAS_kts=75.0,
        weight_final_target=Weight_target_at_1000ft, sref=sref,
        power_setting=IDLE_POWER_SETTING, tol=1e-3, max_iter=20
    )

    CAS_cruise_kts = V_cruise_CAS
    h_cruise_ft    = h_cruise

    Range_NM, delta_t_min = compute_cruise_range_time(
        Weight_initial=weight, Weight_final=Weight_top_descent,
        CAS_kts=CAS_cruise_kts, h_ft=h_cruise_ft, dT_isa=dT_isa
    )

    temps_total_min   += delta_t_min
    distance_total_NM += Range_NM
    weight            = Weight_top_descent

    t_dict["cruise"] = delta_t_min
    d_dict["cruise"] = Range_NM

    # ==========================
    # 5) DESCENTE (Jusqu'à 1000 ft)
    # ==========================
    h_descent_end = 1000.0
    
    t_descent_min, d_descent_NM, Fuel_desc, weight_final_descent = descente(
        hpi=h_cruise, hpf=h_descent_end, dT_isa=dT_isa, CAS_kts=VY,
        weight_initial=weight, sref=sref, power_setting=IDLE_POWER_SETTING
    )

    temps_total_min   += t_descent_min
    distance_total_NM += d_descent_NM
    weight            = weight_final_descent
    
    t_dict["descent"] = t_descent_min
    d_dict["descent"] = d_descent_NM
    
    weight_at_1000ft = weight # Point de transition
    
    # ==========================
    # NOUVELLE PHASE 6 : APPROCHE ET ARRONDIE (1000 ft -> 0 ft)
    # ==========================
    
    # Vitesse d'approche (Approximation Vapp ~ 1.3 * Vs)
    #V_app_kts = aero.V_ref_kts # Assumé que V_ref est défini/importé, sinon utiliser V_cruise_CAS * 1.05
    V_app_kts = VY # Réutilisation de VY si V_ref non défini

    t_approach_min, d_approach_nm, fuel_approach_lb, w_td, V_td_kts = approach_and_flare(
        hpi=h_descent_end, hpf=h_airport, h_flare=h_flare_start, dT_isa=dT_isa,
        V_app_kts=V_app_kts, weight_initial=weight, sref=sref, 
        power_setting=IDLE_POWER_SETTING, gamma_deg=gamma_approach_deg
    )
    
    temps_total_min += t_approach_min
    distance_total_NM += d_approach_nm
    weight = w_td # Poids au toucher des roues

    t_dict["approach"] = t_approach_min
    d_dict["approach"] = d_approach_nm
    
    # ==========================
    # NOUVELLE PHASE 7 : ROULEMENT AU SOL (Ground Roll)
    # ==========================
    
    V_td_ft_s = V_td_kts * 1.6878 # Vitesse de toucher en ft/s
    
    dist_landing_roll_ft, t_landing_s, history_roll = landing_groundroll(
        weight_initial=weight, h_ft=h_airport, dT_isa=dT_isa, V_td_ft_s=V_td_ft_s, 
        power_setting=IDLE_POWER_SETTING, 
        alpha_td_deg= 4, # Assumé dans AeroParams
        mu_brake = 0.4,      # Assumé dans AeroParams
        delay_braking_s=2.0,
        mu_rolling=aero.mu_TO 
    )
    
    # plot_landing_analysis(history_roll)  <-- Removed internal call

    t_landing_min = t_landing_s / 60.0
    dist_landing_roll_nm = dist_landing_roll_ft / 6076.12
    
    temps_total_min += t_landing_min
    distance_total_NM += dist_landing_roll_nm
    
    # Le poids final inclut la petite perte de fuel durant le roulage
    weight_final_mission = history_roll['weight'][-1]
    weight = weight_final_mission
    
    t_dict["groundroll"] = t_landing_min
    d_dict["groundroll"] = dist_landing_roll_nm

    # ==========================
    # 8) RÉSULTATS FINAUX
    # ==========================
    print("\n========== BILAN DE MISSION COMPLÈTE ==========")
    print(f"Temps total        : {temps_total_min:.2f} min ({temps_total_min/60:.2f} h)")
    print(f"Distance totale    : {distance_total_NM:.2f} NM")
    print("--------------------------------------")
    print("POIDS PAR PHASE :")
    print(f"  Début mission        : {weight_start:.2f} lb")
    print(f"  Après décollage      : {weight_decollage:.2f} lb")
    print(f"  Après accélération   : {weight_after_accel:.2f} lb")
    print(f"  Sommet de montée     : {weight_top_climb:.2f} lb")
    print(f"  Début descente       : {Weight_top_descent:.2f} lb")
    print(f"  Final (1000 ft)      : {weight_at_1000ft:.2f} lb")
    print(f"  Au Toucher des Roues : {w_td:.2f} lb")
    print(f"  Final Mission (Arrêt): {weight_final_mission:.2f} lb")
    print("--------------------------------------")
    print("TEMPS PAR PHASE :")
    print(f"  Décollage      : {t_dict['takeoff']:.2f} min")
    print(f"  Accélération   : {t_dict['accel']:.2f} min")
    print(f"  Montée         : {t_dict['climb']:.2f} min")
    print(f"  Croisière      : {t_dict['cruise']:.2f} min")
    print(f"  Descente       : {t_dict['descent']:.2f} min")
    print(f"  Approche/Flare : {t_dict['approach']:.2f} min")
    print(f"  Roulement Sol  : {t_dict['groundroll']:.2f} min")
    

    # Retour structuré des poids et temps par phase (avec ajout des nouvelles phases)
    results = {
        "weights_lb": {
            "start": weight_start,
            "after_takeoff": weight_decollage,
            "after_accel": weight_after_accel,
            "top_climb": weight_top_climb,
            "top_descent": Weight_top_descent,
            "final_1000ft": weight_at_1000ft, # Nouveau point
            "touchdown": w_td, # Nouveau point
            "final": weight_final_mission,
        },
        "times_min": t_dict,
        "ranges_NM": d_dict,
        "landing_history": history_roll, # Ajout de l'historique pour l'affichage externe
    }

    return results

# appel
res = FLIGHT_PHASES(h_cruise, h_airport, dT_Isa, VY, V_cruise_CAS, TO_weight, Weight_Reserve)

# =========================================================
# SAUVEGARDE DES PARAMÈTRES DANS UN FICHIER TEXTE
# =========================================================
try:
    # Création du dictionnaire des inputs de mission (ceux passés à FLIGHT_PHASES)
    mission_inputs = {
        "h_cruise_ft": h_cruise,
        "h_airport_ft": h_airport,
        "dT_isa_C": dT_Isa,
        "VY_kts": VY,
        "V_cruise_CAS_kts": V_cruise_CAS,
        "TO_weight_lb": TO_weight,
        "Weight_Reserve_lb": Weight_Reserve,
        "Calculated_Range_NM": res["ranges_NM"]["cruise"], # Ajout d'une sortie calculée
    }

    # Création du dictionnaire des objets avions (déjà définis globalement)
    aircraft_objects = {
        "fc": fc_default,
        "geom": geom,
        "aero": aero
    }
    
    save_run_parameters("simulation_parameters.txt", mission_inputs, aircraft_objects)

except Exception as e:
    print(f"[WARNING] Could not save parameters report: {e}")

# Appel affichage atterrissage (une seule fois ici)
if "landing_history" in res:
   plot_landing_analysis(res["landing_history"])



# ---------------------------------------------------------
# PRÉPARATION DES DONNÉES POUR LE GRAPHIQUE
# ---------------------------------------------------------

# Extraction pour simplifier la lecture
w_dict = res["weights_lb"]
t_dict = res["times_min"]

# Construction des listes X (Temps cumulé) et Y (Poids)
# On part de t=0
times = [0]
weights = [w_dict["start"]]

# 1. Après Décollage
times.append(times[-1] + t_dict["takeoff"])
weights.append(w_dict["after_takeoff"])

# 2. Après Accélération
times.append(times[-1] + t_dict["accel"])
weights.append(w_dict["after_accel"])

# 3. Sommet de montée (Top of Climb)
times.append(times[-1] + t_dict["climb"])
weights.append(w_dict["top_climb"])

# 4. Début descente (Top of Descent) - Fin de croisière
times.append(times[-1] + t_dict["cruise"])
weights.append(w_dict["top_descent"])

# 5. Fin de mission (Final) - Après descente
times.append(times[-1] + t_dict["descent"])
weights.append(w_dict["final_1000ft"])

# 6. Approche
times.append(times[-1] + t_dict["approach"])
weights.append(w_dict["touchdown"])

# 7. Ground Roll
times.append(times[-1] + t_dict["groundroll"])
weights.append(w_dict["final"])

# ---------------------------------------------------------
# TRACÉ DU GRAPHIQUE (Style Matplotlib)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Tracer la ligne et les points
plt.plot(times, weights, color='#0B5A81', marker='o', linewidth=2, markersize=8, label='Fuel Burn Profile')

# Ajouter les annotations (les valeurs de poids au-dessus des points)
for x, y in zip(times, weights):
    # Le "+ 2" sert à décaler le texte un peu au-dessus du point pour ne pas le cacher
    plt.text(x, y + 1.5, f"{y:.0f}", ha='center', va='bottom', fontsize=10, color='#333333')

# Mise en forme semblable à ton image
plt.title("Estimated weight for the aircraft during each flight phase at 10 000 ft.", fontsize=14)
plt.xlabel("Flight time [min]", fontsize=12)
plt.ylabel("Weight [lbs]", fontsize=12)

# Ajuster les limites Y pour avoir de l'espace au-dessus et en-dessous
plt.ylim(min(weights) - 10, max(weights) + 10)

# Grille légère en arrière-plan
plt.grid(True, linestyle='-', alpha=0.6)

# Afficher
plt.tight_layout()
plt.show()

# =================================================================
# 1. PRÉPARATION DES DONNÉES CUMULÉES (POIDS ET TEMPS)
# =================================================================
# On récupère les dictionnaires de résultats
w_dict = res["weights_lb"]
t_dict = res["times_min"]

# --- Construction de l'Axe X : Temps cumulé [min] ---
times_cumulative = [0]
times_cumulative.append(times_cumulative[-1] + t_dict["takeoff"]) # T1: Fin TO
times_cumulative.append(times_cumulative[-1] + t_dict["accel"])   # T2: Fin Accel
times_cumulative.append(times_cumulative[-1] + t_dict["climb"])   # T3: Top of Climb
times_cumulative.append(times_cumulative[-1] + t_dict["cruise"])  # T4: Top of Descent
times_cumulative.append(times_cumulative[-1] + t_dict["descent"]) # T5: Final (1000ft)

# --- Construction de l'Axe Y1 : Poids [lbs] ---
weights_list = [
    w_dict["start"],         # T0
    w_dict["after_takeoff"], # T1
    w_dict["after_accel"],   # T2
    w_dict["top_climb"],     # T3
    w_dict["top_descent"],   # T4
    w_dict["final"]          # T5
]

# --- Construction de l'Axe Y2 : Altitude [ft] ---
# Basé sur la logique de ton code principal :
alt_ground = h_airport
alt_screen = h_airport + 5.0  # Fin du groundrun + écran
alt_crz    = h_cruise         # Altitude de croisière définie
alt_end    = 1000.0           # Ta descente s'arrête à 1000ft (hpf)

altitudes_list = [
    alt_ground,  # T0: Départ
    alt_screen,  # T1: Fin décollage
    alt_screen,  # T2: Fin accélération (palier)
    alt_crz,     # T3: Sommet de montée
    alt_crz,     # T4: Fin de croisière
    alt_end      # T5: Point final
]

# =================================================================
# PRÉPARATION DES DONNÉES POUR LE PROFIL DE MISSION
# =================================================================

# 1. Récupération des durées
t_dict = res["times_min"]

# Calcul du temps cumulé pour l'axe X [min]
# On part de t=0 et on ajoute la durée de chaque phase
times_cumulative = [0]
times_cumulative.append(times_cumulative[-1] + t_dict["takeoff"]) # Fin décollage
times_cumulative.append(times_cumulative[-1] + t_dict["accel"])   # Fin accélération
times_cumulative.append(times_cumulative[-1] + t_dict["climb"])   # Sommet montée (Top of Climb)
times_cumulative.append(times_cumulative[-1] + t_dict["cruise"])  # Début descente (Top of Descent)
times_cumulative.append(times_cumulative[-1] + t_dict["descent"]) # Point final
times_cumulative.append(times_cumulative[-1] + t_dict["approach"])
times_cumulative.append(times_cumulative[-1] + t_dict["groundroll"])

# 2. Définition des Altitudes aux points correspondants [ft]
# Basé sur la logique et les variables de ton code principal
alt_ground = h_airport         # t=0
alt_screen = h_airport + 5.0   # Fin du groundrun/accélération (5ft dans ton code)
alt_crz    = h_cruise          # Altitude de croisière
alt_final  = 1000.0            # Altitude finale hardcodée dans ta phase descente

altitudes_list = [
    alt_ground,  # t=0
    alt_screen,  # Fin décollage
    alt_screen,  # Fin accélération (palier)
    alt_crz,     # Sommet montée
    alt_crz,     # Fin croisière
    alt_final,   # Point final
    alt_ground,  # Fin approche
    alt_ground   # Fin ground roll
]

# =================================================================
# TRACÉ DU GRAPHIQUE "FLIGHT PROFILE"
# =================================================================
plt.figure(figsize=(10, 6))

# Choix d'une couleur (rouge brique pour l'altitude, c'est classique)
color_alt = '#D9534F'

# Tracer la ligne principale
plt.plot(times_cumulative, altitudes_list, color=color_alt, linewidth=3, marker='o', markersize=6, label='Altitude trajectory')

# Remplir la zone sous la courbe pour donner du corps au profil
plt.fill_between(times_cumulative, altitudes_list, 0, color=color_alt, alpha=0.2)

# --- Ajout d'annotations textuelles sur le graphique pour identifier les phases ---
# Calcul des points médians (en temps) pour placer le texte
mid_climb_t = (times_cumulative[2] + times_cumulative[3]) / 2
mid_cruise_t = (times_cumulative[3] + times_cumulative[4]) / 2
mid_descent_t = (times_cumulative[4] + times_cumulative[5]) / 2

# Placement des textes
# Montée (placé à mi-hauteur)
plt.text(mid_climb_t, h_cruise/2, 'Climb',
         ha='center', va='center', color=color_alt, fontweight='bold', fontsize=11, backgroundcolor='white')

# Croisière (placé un peu au-dessus de l'altitude de croisière)
plt.text(mid_cruise_t, h_cruise + 800, f'Cruise\n({h_cruise:.0f} ft)',
         ha='center', va='bottom', color=color_alt, fontweight='bold', fontsize=11)

# Descente (placé à mi-hauteur)
plt.text(mid_descent_t, h_cruise/2, 'Descent',
         ha='center', va='center', color=color_alt, fontweight='bold', fontsize=11, backgroundcolor='white')

# --- Mise en forme globale ---
plt.title("Flight Mission Profile (Altitude vs. Time)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Flight Time [min]", fontsize=12)
plt.ylabel("Altitude [ft]", fontsize=12, color=color_alt)

# Définir les limites de l'axe Y pour avoir de la marge au-dessus de la croisière
plt.ylim(0, h_cruise * 1.25)
# S'assurer que l'axe X commence à 0
plt.xlim(left=0)

# Grille en arrière-plan
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()




# =================================================================
# COMPARAISON ISA / ISA-15 / ISA+15
# =================================================================

# 1. Définition des scénarios (Delta ISA et couleurs associées)
scenarios = [
    {"dt": 0,   "label": "ISA (Standard)", "color": "black",   "style": "-"},
    {"dt": -15, "label": "ISA - 15°C",     "color": "#1f77b4", "style": "--"}, # Bleu
    {"dt": 15,  "label": "ISA + 15°C",     "color": "#d62728", "style": "--"}  # Rouge
]

plt.figure(figsize=(12, 7))

# 2. Boucle de calcul et de tracé
for scen in scenarios:
    dt = scen["dt"]
    
    # Appel de TA fonction avec le dT_isa spécifique
    # On réutilise les autres variables globales définies plus haut dans ton code
    res_sim = FLIGHT_PHASES(h_cruise, h_airport, dt, VY, V_cruise_CAS, TO_weight, Weight_Reserve)
    
    # --- Reconstruction de l'axe Temps (X) ---
    t_dict_scen = res_sim["times_min"]
    times_scen = [0]
    times_scen.append(times_scen[-1] + t_dict_scen["takeoff"])
    times_scen.append(times_scen[-1] + t_dict_scen["accel"])
    times_scen.append(times_scen[-1] + t_dict_scen["climb"])
    times_scen.append(times_scen[-1] + t_dict_scen["cruise"])
    times_scen.append(times_scen[-1] + t_dict_scen["descent"])
    times_scen.append(times_scen[-1] + t_dict_scen["approach"])
    times_scen.append(times_scen[-1] + t_dict_scen["groundroll"])
    
    # --- Reconstruction de l'axe Altitude (Y) ---
    # On utilise les mêmes hauteurs que dans ton code
    alt_ground = h_airport
    alt_screen = h_airport + 5.0
    alt_crz    = h_cruise
    alt_final  = 1000.0
    
    alts = [
        alt_ground,  # t=0
        alt_screen,  # Fin décollage
        alt_screen,  # Fin accélération
        alt_crz,     # Sommet montée
        alt_crz,     # Fin croisière
        alt_final,   # Point final descente (1000ft)
        alt_ground,  # Fin approche (0ft)
        alt_ground   # Fin groundroll (0ft)
    ]
    
    # --- Tracé de la courbe ---
    plt.plot(times_scen, alts, 
             label=scen["label"], 
             color=scen["color"], 
             linestyle=scen["style"], 
             linewidth=2.5 if dt == 0 else 2) # Ligne ISA un peu plus épaisse

# =================================================================
# MISE EN FORME DU GRAPHIQUE
# =================================================================
plt.title(f"Mission Profile Sensitivity to Temperature\n(Target Cruise Altitude: {h_cruise} ft)", fontsize=14, fontweight='bold')
plt.xlabel("Flight Time [min]", fontsize=12)
plt.ylabel("Altitude [ft]", fontsize=12)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11, loc='best')

# S'assurer que ça commence à 0
plt.xlim(left=0)
plt.ylim(0, h_cruise * 1.2)

plt.tight_layout()
plt.show()


# =================================================================
# 3. TRACÉ DU DÉPLACEMENT DU CG (CG SHIFT)
# =================================================================

# Récupération des données nécessaires (temps cumulé et poids)
# On utilise les listes 'times' et 'weights' déjà créées pour le graphique de poids.

# Calcul des positions CG (en fraction de MAC)
cg_mac_positions = [compute_cg_mac(w) for w in weights]

# --- Tracé du graphique CG vs Temps ---
plt.figure(figsize=(10, 6))

# Tracé principal
plt.plot(
    times, cg_mac_positions,
    color='#6B48FF',
    marker='o',
    linewidth=2.5,
    markersize=7,
    label='Position du CG (en % MAC)'
)

# Ajout des annotations de phase (comme dans le graphique de poids)
phase_labels = [
    "Start",
    "Après TO",
    "Après Accel",
    "Sommet Montée (TOC)",
    "Début Descente (TOD)",
    "Final (1000 ft)",
    "Touchdown",
    "Stop"
]

for i in range(len(times)):
    # Annotation du point (valeur CG)
    plt.text(
        times[i], cg_mac_positions[i] + 0.005,
        f"{cg_mac_positions[i]:.3f}",
        ha='center',
        va='bottom',
        fontsize=9,
        color='#333333'
    )
    # Annotation de la phase (pour référence)
    plt.text(
        times[i], cg_mac_positions[i] - 0.005,
        phase_labels[i],
        ha='center',
        va='top',
        fontsize=8,
        color='gray'
    )

# --- Mise en forme globale ---
plt.title(
    "Progression du Centre de Gravité (CG) par Phase de Vol",
    fontsize=14,
    fontweight='bold',
    pad=20
)
plt.xlabel("Temps de Vol [min]", fontsize=12)
plt.ylabel("Position du CG (Fraction de MAC)", fontsize=12)

# Récupération des valeurs extrêmes pour l'axe Y
min_cg = min(cg_mac_positions)
max_cg = max(cg_mac_positions)

# Affichage des bornes théoriques du CG (selon cg_shift.py)
_, _, CG_MAC_AT_MTOW, CG_MAC_AT_RESERVE = get_cg_parameters()

plt.axhline(
    CG_MAC_AT_MTOW,
    color='blue',
    linestyle=':',
    alpha=0.6,
    label=f'CG Max Avt. ({CG_MAC_AT_MTOW:.3f})'
)
plt.axhline(
    CG_MAC_AT_RESERVE,
    color='red',
    linestyle=':',
    alpha=0.6,
    label=f'CG Max Arr. ({CG_MAC_AT_RESERVE:.3f})'
)

# Ajuster les limites Y
plt.ylim(min_cg * 0.95, max_cg * 1.1)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# =================================================================
# 4. TRACÉ CG vs POIDS (pour vérifier l'interpolation linéaire)
# =================================================================

plt.figure(figsize=(10, 6))
plt.plot(
    weights, cg_mac_positions,
    color='#2CA02C',
    marker='o',
    linewidth=2.5,
    markersize=7,
    label='CG vs Poids'
)

# Annotations des phases (W vs CG)
for i in range(len(times)):
    plt.text(
        weights[i], cg_mac_positions[i],
        phase_labels[i],
        ha='left',
        va='bottom',
        fontsize=9,
        color='#333333'
    )

plt.axhline(CG_MAC_AT_MTOW, color='blue', linestyle=':', alpha=0.6)
plt.axhline(CG_MAC_AT_RESERVE, color='red', linestyle=':', alpha=0.6)

plt.title(
    "Position du Centre de Gravité en fonction du Poids",
    fontsize=14,
    fontweight='bold',
    pad=20
)
plt.xlabel("Poids [lbs]", fontsize=12)
plt.ylabel("Position du CG (Fraction de MAC)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()