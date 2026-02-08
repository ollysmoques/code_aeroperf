# -*- coding: utf-8 -*-

"""
Created on Thu Dec 4 2025

@author: up60044568
"""

import numpy as np
import matplotlib.pyplot as plt
from atmosphere import atmosphere, vitesses # Importe les fonctions atmosphère et vitesses
from total_drag import drag_total           # Importe la fonction de traînée totale
from Thrust_data import thrust_sw400pro_ft_lbf # Importe la fonction de poussée
from Aircraft_data import get_default_inputs, AeroParams # Importe les données avion
from cg_shift import compute_cg_mac         # CG dynamique

# ============================================================
# 1. PARAMÈTRES D'ATTERRISSAGE (doivent être cohérents avec AeroParams)
# ============================================================

fc_default, geom, aero = get_default_inputs()
sref = geom.S_ref
g0 = 32.174  # Gravité [ft/s²]

# Paramètres spécifiques à l'atterrissage (à valider dans Aircraft_data)
MU_BRAKE = 0.40      # Coefficient de friction au freinage (0.3 à 0.5 typique)
CL_MAX_LANDING = aero.CL_max_TO # CL_max pour la configuration atterrissage (volets max)
IDLE_POWER_SETTING = 0.05 # Réglage de puissance au ralenti (idle)

# Angles de toucher (assumés ici)
ALPHA_TOUCHDOWN = 4.0 # Angle d'attaque au toucher [deg]
ALPHA_STATIC_DEG = 7.0 # NOUVEAU: AoA lorsque la queue est au sol (3-points) [deg]
V_TAIL_DOWN_FACTOR = 1.05 # NOUVEAU: Vitesse de transition V_TD / Vs (Vs * 1.05)

# ============================================================
# 2. FONCTIONS DE FORCE ET DE VITESSE
# ============================================================

def v_touchdown(h_ft, dT_isa, weight, CL_max, sref):
    """
    Calcule la vitesse de décrochage Vs et la vitesse de toucher V_TD.
    V_TD est typiquement 1.15 * Vs.
    """
    atm = atmosphere(h=h_ft, variable=dT_isa, mode='delta_isa')
    rho = atm['densite'] 
    
    # Vs [ft/s]
    Vs_ft_s = np.sqrt(2 * weight / (rho * sref * CL_max))
    v_td_ft_s = 1.15 * Vs_ft_s
    
    return Vs_ft_s, v_td_ft_s

def sumforce_landing(
    mu_brake, 
    h_ft, 
    dT_isa, 
    V_ft_s, 
    weight, 
    alpha_deg, 
    thrust_lbf, 
    cg_mac_current
):
    """
    Calcule l'accélération sur le roulement au sol à l'atterrissage.
    L'accélération 'a' est négative (décélération).
    """
    alpha_rad = np.radians(alpha_deg)
    
    # Vitesse en kts pour la fonction drag_total
    V_kts = V_ft_s / 1.6878

    # 1. Calcul de la traînée (configuration 'TO' est utilisé ici pour config flaps, à changer si vous ajoutez 'Landing')
    # NOTE: Dans un vrai modèle, vous définiriez une config 'LANDING' dans DRAG_TOTAL.py
    cltot, drag, qpsf = drag_total(h_ft, dT_isa, V_kts, weight, cg_mac_current, 'alpha', 'TO', alpha_deg) 
    
    # 2. Portance
    lift = cltot * qpsf * sref
    
    # 3. Force Normale (poids sur les roues)
    # Norme = Poids - Portance - Composante verticale de la Poussée
    norme = weight - lift - np.sin(alpha_rad) * thrust_lbf
    norme = max(norme, 0) # La force normale ne peut pas être négative

    # 4. Forces de frottement et freinage
    friction_brake = mu_brake * norme
    
    # 5. Forces motrices (ralenti ou inversion)
    thrust_x = thrust_lbf * np.cos(alpha_rad)
    
    # 6. Force Résultante NÉGATIVE (Décélération)
    Fres = thrust_x - drag - friction_brake
    
    # 7. Accélération
    mslugs = weight / g0
    a = Fres / mslugs  # 'a' devrait être négatif (décélération)
    
    return a, lift, drag, friction_brake, thrust_x, norme, cltot, qpsf

# ============================================================
# 3. FONCTION PRINCIPALE : ROULEMENT AU SOL
# ============================================================

def landing_groundroll(
    weight_initial, 
    h_ft, 
    dT_isa, 
    V_td_ft_s, 
    power_setting, 
    alpha_td_deg,
    mu_brake,
    delay_braking_s=2.0,   # NOUVEAU: Délai avant freinage [s]
    mu_rolling=0.04        # NOUVEAU: Friction en roulement libre (0.03-0.05 typique)
):
    """
    Simule la phase de roulement au sol d'un taildragger depuis le toucher (V_TD) jusqu'à l'arrêt (V=0).
    
    La phase est divisée en deux segments pour modéliser la décélération :
    1. Phase 1 (Assiette haute): L'avion roule sur les roues principales (portance élevée, freinage limité).
    2. Phase 2 (Queue au sol): La queue est posée, l'AoA est réduit (portance minimale, freinage maximal).

    Args:
        weight_initial (float): Poids de l'avion au toucher [lbf].
        h_ft (float): Altitude de l'aéroport [ft].
        dT_isa (float): Delta ISA [°C].
        V_td_ft_s (float): Vitesse de toucher [ft/s].
        power_setting (float): Réglage de puissance (idle ou reverse) [0-1].
        alpha_td_deg (float): Angle d'attaque au toucher (Phase 1) [deg].
        mu_brake (float): Coefficient de friction au freinage.
        delay_braking_s (float): Temps avant application des freins [s].
        mu_rolling (float): Coefficient de friction sans freins.

    Returns:
        float: distance_totale [ft], t_total [s], history [dict]
    """
    
    # Assurer que les constantes sont accessibles (définies globalement ou passées)
    # Dans ce contexte de module, on les utilise telles quelles.
    
    # 1. Initialisation
    history = {
        'v': [], 'dist': [], 'a': [], 
        'lift': [], 'drag': [], 'friction': [], 
        'thrust': [], 'norme': [], 'cl': [], 'weight': []
    }

    v = V_td_ft_s
    distance = 0.0
    t = 0.0
    dt = 0.005 # Pas de temps
    
    # Initialisation des variables pour éviter UnboundLocalError si les boucles ne s'exécutent pas
    a = 0.0
    lift = 0.0
    drag = 0.0
    fric = 0.0
    thrust = 0.0
    norm = 0.0
    cl = 0.0
    
    poussee, fuel_consumption_per_min = thrust_sw400pro_ft_lbf(h_ft, dT_isa, power_setting)
    weight = weight_initial
    
    # Calcul de la vitesse de transition (où la queue tombe)
    Vs_ft_s, _ = v_touchdown(h_ft, dT_isa, weight_initial, CL_MAX_LANDING, sref)
    V_tail_down = Vs_ft_s * V_TAIL_DOWN_FACTOR # Ex: 1.05 * Vs

    
    def record_step(v, dist, a, lift, drag, fric, thrust, norm, cl, weight):
        history['v'].append(v)
        history['dist'].append(dist)
        history['a'].append(a)
        history['lift'].append(lift)
        history['drag'].append(drag)
        history['friction'].append(fric)
        history['thrust'].append(thrust)
        history['norme'].append(norm)
        history['cl'].append(cl)
        history['weight'].append(weight)

    # =======================================================
    # PHASE 1: Roulement à haute vitesse (AoA = ALPHA_TOUCHDOWN)
    # L'avion est principalement supporté par l'ascenseur (Lift).
    # Le freinage est moins efficace car la force normale (Norme) est faible.
    # =======================================================
    alpha_current = alpha_td_deg
    
    while v > V_tail_down:
        
        cg_mac_current = compute_cg_mac(weight)
        
        # Gestion du délai de freinage
        if t < delay_braking_s:
            mu_current = mu_rolling
        else:
            mu_current = mu_brake

        a, lift, drag, fric, thrust, norm, cl, q = sumforce_landing(
            mu_current, h_ft, dT_isa, v, weight, alpha_current, poussee, cg_mac_current
        )

        # Vérification: L'accélération doit être négative pour décélérer
        if a >= -1e-6:
             print(f"⚠ Arrêt de la simulation (Phase 1) : a >= 0 à {v/1.6878:.1f} kts.")
             break

        record_step(v, distance, a, lift, drag, fric, thrust, norm, cl, weight)

        # Intégration temporelle
        t += dt
        v_2 = v + dt * a 
        distance += (v + v_2) / 2 * dt 
        v = v_2
        
        # Consommation de carburant
        weight -= fuel_consumption_per_min / 60.0 * dt

    # =======================================================
    # PHASE 2: Queue au sol (AoA = ALPHA_STATIC_DEG)
    # L'AoA est réduit, la Portance chute, la Force Normale et le Freinage augmentent.
    # Cette phase assure l'arrêt complet.
    # =======================================================
    while v > 0.1:
        
        cg_mac_current = compute_cg_mac(weight)
        alpha_current = ALPHA_STATIC_DEG # Réduction de l'AoA pour le 3-points
        
        # Gestion du délai de freinage
        if t < delay_braking_s:
            mu_current = mu_rolling
        else:
            mu_current = mu_brake

        a, lift, drag, fric, thrust, norm, cl, q = sumforce_landing(
            mu_current, h_ft, dT_isa, v, weight, alpha_current, poussee, cg_mac_current
        )

        if a >= -1e-6:
             print(f"⚠ Arrêt de la simulation (Phase 2) : a >= 0 à {v/1.6878:.1f} kts.")
             break

        record_step(v, distance, a, lift, drag, fric, thrust, norm, cl, weight)

        # Intégration temporelle
        t += dt
        v_2 = v + dt * a
        distance += (v + v_2) / 2 * dt 
        v = v_2
        
        # Consommation de carburant
        weight -= fuel_consumption_per_min / 60.0 * dt
        
    # Final step at V=0
    record_step(0.0, distance, 0.0, lift, drag, fric, thrust, norm, cl, weight)

    return distance, t, history


def plot_landing_analysis(h):
    """
    Génère les graphiques d'analyse pour le roulement à l'atterrissage.
    h: dictionnaire 'history' retourné par landing_groundroll containing lists of v, dist, etc.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    v_kts = np.array(h['v']) / 1.6878
    dist_ft = np.array(h['dist'])

    # Graphique 1 : Forces
    axs[0].plot(dist_ft, h['thrust'], label='Thrust (Idle)', color='green', linewidth=2)
    axs[0].plot(dist_ft, np.array(h['drag']), label='Drag (Aero)', color='red')
    axs[0].plot(dist_ft, np.array(h['friction']), label='Braking Friction (μ*Norm)', color='orange', linestyle='--')
    axs[0].plot(dist_ft, np.array(h['drag']) + np.array(h['friction']) - np.array(h['thrust']), label='Net Deceleration Force', color='black')
    axs[0].set_ylabel('Forces [lbf]')
    axs[0].set_title('Landing Roll Forces Analysis')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)

    # Graphique 2 : Accélération / Vitesse
    axs[1].plot(dist_ft, np.array(h['a']), color='purple', linewidth=2, label='Acceleration')
    ax2 = axs[1].twinx()
    ax2.plot(dist_ft, v_kts, color='blue', linestyle=':', label='Groundspeed [kts]')
    axs[1].set_ylabel('Acceleration [ft/s²]', color='purple')
    ax2.set_ylabel('Groundspeed [kts]', color='blue')
    axs[1].grid(True, alpha=0.3)
    
    # Graphique 3 : Portance vs Poids
    axs[2].plot(dist_ft, h['lift'], label='Lift', color='teal')
    axs[2].plot(dist_ft, h['norme'], label='Weight on wheels (Norm)', color='grey', linestyle='--')
    axs[2].plot(dist_ft, h['weight'], label='Aircraft Weight', color='black', linestyle=':')
    axs[2].set_xlabel('Distance [ft]')
    axs[2].set_ylabel('Vertical forces [lbf]')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================
# 4. EXEMPLE D'UTILISATION (pour tester)
# ============================================================

if __name__ == "__main__":
    
    # Poids final après la descente (exemple)
    W_landing_lb = 330.0 
    
    # Calcule la V_TD à partir de ce poids
    Vs, V_td = v_touchdown(h_ft=0, dT_isa=0, weight=W_landing_lb, CL_max=CL_MAX_LANDING, sref=sref)
    
    dist_totale, t_total, h = landing_groundroll(
        weight_initial=W_landing_lb,
        h_ft=0,
        dT_isa=0,
        V_td_ft_s=V_td,
        power_setting=IDLE_POWER_SETTING,
        alpha_td_deg=ALPHA_TOUCHDOWN,
        mu_brake=MU_BRAKE
    )

    print("\n====== LANDING GROUND ROLL RESULTS ======")
    print(f"Vitesse de décrochage (Vs): {Vs / 1.6878:.1f} kts")
    print(f"Vitesse de toucher (V_TD): {V_td / 1.6878:.1f} kts")
    print(f"Distance de roulage: {dist_totale:.2f} ft")
    print(f"Temps de roulage: {t_total:.2f} s")
    print("=========================================\n")
    
    # Appel de la nouvelle fonction de plotting
    plot_landing_analysis(h)
