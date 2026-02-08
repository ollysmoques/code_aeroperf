# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:56:41 2025

@author: edoua
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from atmosphere import *
from total_drag import * 
from Aircraft_data import get_default_inputs
from Thrust_data import thrust_sw400pro_ft_lbf
from cg_shift import *
from Mission_parameters import MISSION_HEIGHT_FT

fc_default, geom, aero = get_default_inputs()


v_trans = aero.v_trans_kts * 1.6878     
weight  = aero.MAX_TO
alpha_ini   = aero.alpha_ini_deg
alpha_trans = aero.alpha_trans_deg
alpha_rot   = aero.alpha_rot_deg

h_ft        = aero.h_TO_ft
dT_isa      = aero.dT_isa_TO
mu          = aero.mu_TO
sref        = geom.S_ref                
CL_max      = aero.CL_max_TO
cg_mac_initial = compute_cg_mac(weight)



def ROC(CAS_kts,h_ft,dT_isa,weight,sref,power_setting, cg_mac_current):
    
    
    atm = atmosphere(h_ft,dT_isa,mode='delta_isa')
    vitesse = vitesses(atm,CAS_kts,'calibree',weight,sref)
    
    Mach = vitesse['mach']
    TAS_kts = vitesse['vitesses avion kts']['vitesse vraie']
    
    T = atm['temperature_K']
    dT_isa = atm['delta_ISA']
    Tstd = T- dT_isa
    
    phi = 1/(0.7*Mach**2) * ((1+0.2*Mach**2)**3.5 -1)/((1+0.2*Mach**2)**2.5)
    
    if dT_isa == 0:
        AF = 0.7*Mach**2 * (phi - 0.190263)
        
    elif dT_isa != 0:
      
        
        AF = 0.7*Mach**2 * (phi - 0.190263*(Tstd/T))
        
    TAS_fps = TAS_kts*1.6878
    cltot, drag, qpsf = drag_total(h_ft, dT_isa, TAS_kts, weight,cg_mac_current, 'weight','clean')
    
    Thrust,Fuel_consumption = thrust_sw400pro_ft_lbf(h_ft,dT_isa,power_setting)
    
    ROC_fps = (TAS_fps * (Thrust - drag)/weight )/(1+AF)
    ROC_fpm = 60*ROC_fps
    
    ROC_fpm_p = ROC_fpm * (Tstd/T)
    # print('Drag : ',drag)
    # print('Thrust:', Thrust)
    # print('CL_tot:',cltot)
    # print('ROC_fpm_p:',ROC_fpm_p)
    
    return ROC_fpm_p


ROC_fpm_p = ROC(90,0,0,weight,sref,0.7, cg_mac_initial)
print(ROC_fpm_p)



# Plage de vitesses CAS [kts]
CAS_min = 60
CAS_max = 120
n_CAS   = 40
CAS_vals = np.linspace(CAS_min, CAS_max, n_CAS)

# Plage d'altitudes [ft]
h_min = 0

# changer la plage ici

h_max = MISSION_HEIGHT_FT
n_h   = 30
h_vals = np.linspace(h_min, h_max, n_h)

# Grille 2D (CAS, h)
CAS_grid, h_grid = np.meshgrid(CAS_vals, h_vals)

# Grille pour stocker le ROC correspondant
ROC_grid = np.zeros_like(CAS_grid)

# Remplir la grille : ROC(CAS, h)
for i in range(n_h):
    for j in range(n_CAS):
        CAS_ij = CAS_grid[i, j]
        h_ij   = h_grid[i, j]
        ROC_grid[i, j] = ROC(CAS_ij, h_ij, dT_isa, weight, sref,0.75, cg_mac_initial)

# --- Figure 3D ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    CAS_grid, h_grid, ROC_grid,
    cmap='viridis', edgecolor='none', alpha=0.9
)

ax.set_xlabel('CAS [kts]')
ax.set_ylabel('Altitude [ft]')
ax.set_zlabel('ROC corrigé [ft/min]')
ax.set_title('Rate of Climb vs CAS and altitude')

fig.colorbar(surf, shrink=0.6, aspect=12, label='ROC [ft/min]')
plt.tight_layout()
plt.show()





# ============================
#       PLOT 2D ROC vs CAS
# ============================

CAS_list = np.linspace(60, 120, 50)   # [kts]

ROC_0ft     = []
ROC_10000ft = []

for CAS in CAS_list:
    ROC_0ft.append(ROC(CAS, 0, dT_isa, weight, sref, 0.75, cg_mac_initial))
    ROC_10000ft.append(ROC(CAS, MISSION_HEIGHT_FT, dT_isa, weight, sref, 0.75, cg_mac_initial))

ROC_0ft     = np.array(ROC_0ft)
ROC_10000ft = np.array(ROC_10000ft)

# ---- Vitesse donnant le meilleur ROC (Vy) ----
idx_0ft     = np.argmax(ROC_0ft)
idx_10000ft = np.argmax(ROC_10000ft)

Vy_0ft      = CAS_list[idx_0ft]
Vy_10000ft  = CAS_list[idx_10000ft]

ROCmax_0ft     = ROC_0ft[idx_0ft]
ROCmax_10000ft = ROC_10000ft[idx_10000ft]

print(f"Vy à 0 ft     ≈ {Vy_0ft:.1f} kts, ROC_max ≈ {ROCmax_0ft:.0f} ft/min")
print(f"Vy à {MISSION_HEIGHT_FT:.0f} ft ≈ {Vy_10000ft:.1f} kts, ROC_max ≈ {ROCmax_10000ft:.0f} ft/min")

# --- Trace 2D ---
plt.figure(figsize=(8,5))
plt.plot(CAS_list, ROC_0ft, label="Altitude = 0 ft", linewidth=2)
plt.plot(CAS_list, ROC_10000ft, label=f"Altitude = {MISSION_HEIGHT_FT:.0f} ft", linewidth=2)

# Marqueurs pour Vy
plt.scatter(Vy_0ft, ROCmax_0ft, color='blue', marker='o')
plt.scatter(Vy_10000ft, ROCmax_10000ft, color='orange', marker='o')

plt.text(Vy_0ft, ROCmax_0ft,  f" Vy0 ≈ {Vy_0ft:.0f} kt",  fontsize=8, ha='left', va='bottom')
plt.text(Vy_10000ft, ROCmax_10000ft, f" Vy10k ≈ {Vy_10000ft:.0f} kt", fontsize=8, ha='left', va='bottom')

plt.axhline(0, color='black', linewidth=0.8)

plt.xlabel("CAS [kts]")
plt.ylabel("ROC  [ft/min]")
plt.title(f"Rate of Climb vs CAS (0 ft et {MISSION_HEIGHT_FT:.0f} ft)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()





def montee(hpi,hpf,dT_isa,CAS_kts,weight_initial,sref,power_setting):
    dh = 5
    hp1 = hpi
    weight = weight_initial
    temps = 0
    distance = 0
    Fuel_tot = 0
    while hp1 < hpf:
        if hp1 + dh > hpf:
            hp2 = hpf
            
        else :
            hp2 = hp1 + dh
        
        
        hp_moy = (hp1+hp2)/2
        atm = atmosphere(hp_moy,dT_isa,mode='delta_isa')
        vitesse = vitesses(atm,CAS_kts,'calibree',weight,sref)
        TAS_kts = vitesse['vitesses avion kts']['vitesse vraie']
        
        T = atm['temperature_K']
        dT_isa = atm['delta_ISA']
        Tstd = T- dT_isa
        
        cg_mac_current = compute_cg_mac(weight)
        
        dhp = 5
        dh_geo = dhp*T/Tstd
        
        ROC_geo_fpm = ROC(CAS_kts,hp_moy,dT_isa,weight,sref,power_setting, cg_mac_current)
        
        if ROC_geo_fpm < 300:
            print(f" ROC trop faible ({ROC_geo_fpm:.0f} ft/min) à {hp_moy:.0f} ft — montée impossible.")
            break
            
        thrust, Fuel_consumption = thrust_sw400pro_ft_lbf(hp_moy, dT_isa, power_setting)
        delta_t = (dh_geo/ROC_geo_fpm)
        temps += delta_t
        delta_distance = TAS_kts*delta_t/60
        distance += delta_distance
        delta_fuel = Fuel_consumption*delta_t
        weight -= delta_fuel
        Fuel_tot += delta_fuel
        hp1 = hp2
        
        if hp1 >= hpf:
            break
        
    return temps,distance,Fuel_tot,weight



temps,distance,Fuel_tot,weight = montee(50,MISSION_HEIGHT_FT,0,73,375,sref,0.75)

print('=========Valeurs montée===========')
print('Temps:',temps)
print('Distance:', distance)
print('Carburant_Consommé:',Fuel_tot)
print('Poids final:',weight)



def descente(hpi, hpf, dT_isa, CAS_kts, weight_initial, sref, power_setting):
    dh = 5.0          # pas de pression [ft]
    hp1 = hpi         # altitude de départ (doit être > hpf)
    weight = weight_initial
    temps = 0.0       # [min]
    distance = 0.0    # [NM]
    Fuel_tot = 0.0    # [lb]

    while hp1 > hpf:
        # Choix du point suivant
        if hp1 - dh < hpf:
            hp2 = hpf
        else:
            hp2 = hp1 - dh

        # Altitude moyenne du segment
        hp_moy = 0.5 * (hp1 + hp2)
        
        

        # Atmosphère & vitesses
        atm = atmosphere(hp_moy, dT_isa, mode='delta_isa')
        vitesse = vitesses(atm, CAS_kts, 'calibree', weight, sref)
        TAS_kts = vitesse['vitesses avion kts']['vitesse vraie']

        T = atm['temperature_K']
        dT_isa_loc = atm['delta_ISA']
        Tstd = T - dT_isa_loc

        # Conversion dh pression -> dh géométrique (peut être négatif en descente)
        dhp = hp2 - hp1            # < 0
        dh_geo = dhp * T / Tstd    # < 0 aussi (normal en descente)

        cg_mac_current = compute_cg_mac(weight)
        
        # Rate of climb "géométrique" (sera négatif si Thrust < Drag)
        ROC_geo_fpm = ROC(CAS_kts, hp_moy, dT_isa_loc, weight, sref, power_setting, cg_mac_current)

        # Sécurité : éviter division par ~0 ou ROC dans le mauvais sens
        if ROC_geo_fpm >= -1e-3:
            print(f"⚠ ROC trop faible ou positif ({ROC_geo_fpm:.1f} ft/min) à {hp_moy:.0f} ft — descente interrompue.")
            break

        # Temps de traversée du segment (ft / (ft/min) = min)
        delta_t = dh_geo / ROC_geo_fpm   # [min] (négatif / négatif = positif)
        temps += delta_t

        # Distance parcourue (TAS en kts, delta_t en min → NM)
        delta_distance = TAS_kts * (delta_t / 60.0)
        distance += delta_distance

        # Poussée & carburant
        thrust, Fuel_consumption = thrust_sw400pro_ft_lbf(hp_moy, dT_isa_loc, power_setting)
        delta_fuel = Fuel_consumption * delta_t   # lb/min * min = lb
        Fuel_tot += delta_fuel
        weight -= delta_fuel

        # Mise à jour altitude pour le prochain pas
        hp1 = hp2

    return temps, distance, Fuel_tot, weight
    
    

# temps, distance, Fuel_tot, weight_final = descente(
#     hpi=10000,
#     hpf=1000,
#     dT_isa=0,
#     CAS_kts=70,
#     weight_initial=340,
#     sref=sref,
#     power_setting=0.07
# )

# print('=========Valeurs Descente===========')
# print(f'Temps           : {temps:.2f} min')
# print(f'Distance        : {distance:.2f} NM')
# print(f'Carburant brûlé : {Fuel_tot:.2f} lb')
# print(f'Poids final     : {weight_final:.2f} lb')



def find_initial_weight_for_descent(hpi, hpf, dT_isa, CAS_kts,
                                    weight_final_target, sref,
                                    power_setting,
                                    tol=1e-3, max_iter=20):
    """
    Trouve le poids initial de la descente pour obtenir
    un poids final donné, en utilisant la fonction descente().

    Paramètres
    ----------
    hpi : float
        Altitude initiale de la descente [ft].
    hpf : float
        Altitude finale désirée [ft].
    dT_isa : float
        Delta ISA [°C].
    CAS_kts : float
        Vitesse calibrée en kts.
    weight_final_target : float
        Poids désiré à la fin de la descente [lb].
    sref : float
        Surface de référence [ft^2].
    power_setting : float
        Réglage de puissance (0–1).
    tol : float
        Tolérance sur la différence de poids final [lb].
    max_iter : int
        Nombre max d'itérations.

    Retour
    ------
    weight_initial_est : float
        Poids initial estimé [lb].
    Fuel_tot : float
        Carburant brûlé sur la descente [lb] avec ce poids initial.
    weight_final_obtenu : float
        Poids final obtenu avec cette solution [lb].
    """
    # 1) Premier guess : poids final + petite marge
    weight_initial_guess = weight_final_target + 5.0

    for it in range(max_iter):
        # Appel à ta fonction descente avec ce poids initial
        temps, distance, Fuel_tot, weight_final = descente(
            hpi, hpf, dT_isa, CAS_kts, weight_initial_guess, sref, power_setting
        )

        # Erreur par rapport au poids final désiré
        error = weight_final - weight_final_target
        print(f"[Iter {it}] W_init = {weight_initial_guess:.3f} lb, "
              f"W_final = {weight_final:.3f} lb, "
              f"Erreur = {error:.4f} lb")

        if abs(error) < tol:
            # On a convergé
            return weight_initial_guess

        # Mise à jour du guess :
        # on veut W_init ≈ W_target + Fuel_tot(W_init)
        # donc on utilise le Fuel_tot actuel pour améliorer le guess.
        weight_initial_guess = weight_final_target + Fuel_tot

    # Si on arrive ici, pas de convergence stricte, on retourne la dernière valeur
    print("⚠ Attention : convergence non atteinte dans le nombre d'itérations max.")
    return weight_initial_guess


def acceleration(CAS_kts_initial, CAS_kts_final,
                 h_ft, dT_isa_input,
                 weight_lbf, sref_ft2,
                 power_setting):

    # Vitesse moyenne en CAS
    CAS_kts_avg = 0.5 * (CAS_kts_initial + CAS_kts_final)

    # Atmosphère
    atm = atmosphere(h_ft, dT_isa_input, mode='delta_isa')

    # Vitesses
    vit_ini  = vitesses(atm, CAS_kts_initial, 'calibree', weight_lbf, sref_ft2)
    vit_fin  = vitesses(atm, CAS_kts_final,   'calibree', weight_lbf, sref_ft2)
    vit_avg  = vitesses(atm, CAS_kts_avg,     'calibree', weight_lbf, sref_ft2)

    TAS_kts_initiale = vit_ini['vitesses avion kts']['vitesse vraie']
    TAS_kts_finale   = vit_fin['vitesses avion kts']['vitesse vraie']
    TAS_kts_avg      = vit_avg['vitesses avion kts']['vitesse vraie']


    cg_mac_current = compute_cg_mac(weight_lbf) # On utilise le poids sur le segment

    # Delta ISA réel renvoyé par atmosphere
    delta_ISA = atm['delta_ISA']

    # Traînée et poussée (Thrust, drag en lbf, Fuel_consumption en fuel/h)
    cltot, drag_lbf, qpsf = drag_total(h_ft, delta_ISA, TAS_kts_avg, weight_lbf,cg_mac_current, 'weight','TO')
    thrust_lbf, fuel_flow_per_min = thrust_sw400pro_ft_lbf(h_ft, delta_ISA, power_setting)

    # Accélération en ft/s^2
    g0 = 32.174
    a_ft_s2 = (thrust_lbf - drag_lbf) / weight_lbf * g0

    # Temps d'accélération
    dV_ft_s = (TAS_kts_finale - TAS_kts_initiale) * 1.6878  # kts -> ft/s
    dt_s    = dV_ft_s / a_ft_s2                             # secondes
    dt_min  = dt_s / 60.0                                   # minutes
    dt_hr   = dt_s / 3600.0                                 # heures

    # Distance (TAS en kts = NM/h)
    d_dist_NM = TAS_kts_avg * dt_hr

    # Fuel (si fuel_flow_per_hour est en fuel/h)
    d_Fuel = fuel_flow_per_min * dt_min

    return dt_min, d_dist_NM, d_Fuel


# dt_min, d_dist, d_Fuel = acceleration(65,90,50,0,375,36.8,1)

# print(dt_min)
# print(d_dist)
# print(d_Fuel)
    
    
    
    
    
    
    
    


    
        
    
    
