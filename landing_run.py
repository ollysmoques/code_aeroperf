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
CL_MAX_LANDING = aero.cl_max_30 # CL_max pour la configuration atterrissage (volets max)
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

    # 1. Calcul de la traînée (configuration 'landing' pour flaps à 30°)
    cltot, drag, qpsf, _ = drag_total(h_ft, dT_isa, V_kts, weight, cg_mac_current, weight, 'landing', alpha_deg) 
    
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
        'thrust': [], 'norme': [], 'cl': [], 'weight': [],
        'stall_margin_pct': [], 'stall_warning': []  # NEW: Stall warning tracking
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

    
    def record_step(v, dist, a, lift, drag, fric, thrust, norm, cl, weight, stall_margin_pct=0.0, is_stall_warning=False):
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
        history['stall_margin_pct'].append(stall_margin_pct)
        history['stall_warning'].append(is_stall_warning)

    # =======================================================
    # PHASE 1: Roulement à haute vitesse (AoA = ALPHA_TOUCHDOWN)
    # L'avion est principalement supporté par l'ascenseur (Lift).
    # Le freinage est moins efficace car la force normale (Norme) est faible.
    # =======================================================
    alpha_current = alpha_td_deg
    stall_warning_printed_phase1 = False  # Flag to avoid repeated warnings
    
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

        # NEW: Calculate current stall speed and margin
        Vs_current, _ = v_touchdown(h_ft, dT_isa, weight, CL_MAX_LANDING, sref)
        stall_margin = ((v - Vs_current) / Vs_current) * 100.0  # Margin in percentage above Vs
        is_stall_warning = stall_margin < 15.0  # Warning if within 15% of Vs
        
        # Print warning if approaching stall
        if is_stall_warning and not stall_warning_printed_phase1:
            print(f"⚠️  STALL WARNING (Phase 1): Velocity {v/1.6878:.1f} kts approaching Vs {Vs_current/1.6878:.1f} kts (Margin: {stall_margin:.1f}%)")
            stall_warning_printed_phase1 = True
        
        record_step(v, distance, a, lift, drag, fric, thrust, norm, cl, weight, stall_margin, is_stall_warning)

        # Intégration temporelle
        t += dt
        v_2 = v + dt * a 
        distance += (v + v_2) / 2 * dt 
        v = v_2
        
        # Consommation de carburant
        weight -= fuel_consumption_per_min / 60.0 * dt
        
        # Reset warning flag if we gain margin
        if stall_margin > 20.0:
            stall_warning_printed_phase1 = False

    # =======================================================
    # PHASE 2: Queue au sol (AoA = ALPHA_STATIC_DEG)
    # L'AoA est réduit, la Portance chute, la Force Normale et le Freinage augmentent.
    # Cette phase assure l'arrêt complet.
    # =======================================================
    stall_warning_printed_phase2 = False  # Flag to avoid repeated warnings
    
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

        # NEW: Calculate current stall speed and margin
        Vs_current, _ = v_touchdown(h_ft, dT_isa, weight, CL_MAX_LANDING, sref)
        stall_margin = ((v - Vs_current) / Vs_current) * 100.0  # Margin in percentage above Vs
        is_stall_warning = stall_margin < 15.0  # Warning if within 15% of Vs
        
        # Print warning if approaching stall
        if is_stall_warning and not stall_warning_printed_phase2:
            print(f"⚠️  STALL WARNING (Phase 2): Velocity {v/1.6878:.1f} kts approaching Vs {Vs_current/1.6878:.1f} kts (Margin: {stall_margin:.1f}%)")
            stall_warning_printed_phase2 = True
        
        record_step(v, distance, a, lift, drag, fric, thrust, norm, cl, weight, stall_margin, is_stall_warning)

        # Intégration temporelle
        t += dt
        v_2 = v + dt * a
        distance += (v + v_2) / 2 * dt 
        v = v_2
        
        # Consommation de carburant
        weight -= fuel_consumption_per_min / 60.0 * dt
        
        # Reset warning flag if we gain margin
        if stall_margin > 20.0:
            stall_warning_printed_phase2 = False
        
    # Final step at V=0
    # Calculate final stall margin for last point
    Vs_final, _ = v_touchdown(h_ft, dT_isa, weight, CL_MAX_LANDING, sref)
    stall_margin_final = ((0.0 - Vs_final) / Vs_final) * 100.0
    record_step(0.0, distance, 0.0, lift, drag, fric, thrust, norm, cl, weight, stall_margin_final, False)

    return distance, t, history


def plot_landing_analysis(h):
    """
    Génère les graphiques d'analyse pour le roulement à l'atterrissage.
    h: dictionnaire 'history' retourné par landing_groundroll containing lists of v, dist, etc.
    Ajoute un 4ème subplot pour l'énergie cinétique (KE) et un 5ème pour le stall margin.
    """
    fig, axs = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

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
    axs[2].set_ylabel('Vertical forces [lbf]')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)

    # Graphique 4 : Énergie cinétique (KE)
    # KE = 0.5 * m * v^2 (avec m en slugs, v en ft/s) -> unité: ft*lbf
    weights = np.array(h['weight'])
    mass_slugs = weights / g0
    v_ft_s = np.array(h['v'])
    KE_ft_lbf = 0.5 * mass_slugs * v_ft_s**2

    axs[3].plot(dist_ft, KE_ft_lbf, color='brown', linewidth=2, label='Kinetic Energy')
    axs[3].set_ylabel('KE [ft·lbf]')
    axs[3].legend(loc='upper right')
    axs[3].grid(True, alpha=0.3)

    # Graphique 5 : Stall Margin (NEW)
    stall_margin = np.array(h['stall_margin_pct'])
    stall_warnings = np.array(h['stall_warning'])
    
    axs[4].plot(dist_ft, stall_margin, color='darkred', linewidth=2.5, label='Stall Margin')
    
    # Mark warning zones in red
    warning_zones = np.where(stall_warnings)[0]
    if len(warning_zones) > 0:
        for idx in warning_zones:
            axs[4].plot(dist_ft[idx], stall_margin[idx], 'r.', markersize=8)  # Red dots for warnings
    
    # Add threshold line and warning zone
    axs[4].axhline(15.0, color='orange', linestyle='--', linewidth=2, label='Stall Warning Threshold (15%)')
    axs[4].axhline(0.0, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Stall Speed (0% margin)')
    axs[4].fill_between(dist_ft, 0, 15, color='red', alpha=0.1, label='Warning Zone')
    
    axs[4].set_xlabel('Distance [ft]')
    axs[4].set_ylabel('Stall Margin [%]')
    axs[4].set_title('Stall Warning Monitor')
    axs[4].legend(loc='upper right')
    axs[4].grid(True, alpha=0.3)
    
    # Add summary statistics
    min_margin = np.min(stall_margin)
    min_margin_idx = np.argmin(stall_margin)
    stall_margin_dist = dist_ft[min_margin_idx]
    
    textstr = f'Min Margin: {min_margin:.1f}% @ {stall_margin_dist:.0f} ft'
    axs[4].text(0.98, 0.95, textstr, transform=axs[4].transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

# ============================================================
# 4. EXEMPLE D'UTILISATION (pour tester)
# ============================================================

if __name__ == "__main__":
    
    # Poids final après la descente (exemple) - utiliser le poids de réserve
    fc_default, geom, aero = get_default_inputs()
    W_landing_lb = aero.OEW + aero.PAYLOAD  # Poids sans carburant de réserve
    
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
