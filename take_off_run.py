import numpy as np
import matplotlib.pyplot as plt
from atmosphere import *
from total_drag import * 
from Aircraft_data import get_default_inputs
from Thrust_data import *
from cg_shift import *

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
CL_max      = aero.cl_max_15

def v_lof_v_r(h_ft,dT_isa,weight,CL_max,sref):
    
    atm = atmosphere(h=h_ft, variable=dT_isa, mode='delta_isa')
    rho = atm['densite'] 
    
    Vs_ft_s = np.sqrt(2*weight/(rho*sref*CL_max))
    
    v_r = 1.25*Vs_ft_s
    v_lof = 1.35*Vs_ft_s
    
    return v_r, v_lof
    
    
    
    
def sumforce(mu, h_ft, dT_isa, V, weight, alpha, thrust_lbf, cg_mac_current):
    flag = False
    alpha_rad = np.radians(alpha)
    
    # On NE recalcul plus la poussée ici !
    poussee = thrust_lbf

    # On récupère cltot ici
    cltot, drag, qpsf = drag_total(h_ft, dT_isa,V/1.6878,weight,cg_mac_current,thrust_lbf,'take_off',alpha)
    
    lift = cltot * qpsf * sref
    norme = weight - lift - np.sin(alpha_rad) * poussee
    
    if norme < 0:
        norme = 0
        flag = True
        
    friction = mu * norme
    mslugs = weight / 32.174  # ou 32.124 selon ta convention
    
    thrust_x = poussee * np.cos(alpha_rad)
    Fres = thrust_x - drag - friction
    a = Fres / mslugs
    
    return a, flag, lift, drag, friction, thrust_x, norme, cltot, qpsf

def groundrun(v_trans, weight_initial, alpha_trans, alpha_ini, h_ft, dT_isa, power_setting, CL_max):
    
    history = {
        'v': [], 'dist': [], 'a': [], 
        'lift': [], 'drag': [], 'friction': [], 
        'thrust': [], 'norme': [], 'cl': [] , 'weight':[]
    }

    v = 1
    distance = 0
    t = 0
    dt = 0.005 
    
    # On calcule la poussée UNE FOIS pour ce power setting
    poussee, fuel_consumption = thrust_sw400pro_ft_lbf(h_ft, dT_isa, power_setting)
   
    
    v_r, v_lof = v_lof_v_r(h_ft, dT_isa, weight_initial, CL_max, sref)
    
    def record_step(v, dist, a, lift, drag, fric, thrust, norm, cl, sref ,weight):
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
        
    weight = weight_initial

    # --- PHASE 1 ---
    while v < v_trans:
        
        cg_mac_current = compute_cg_mac(weight)
        
        a, flag, lift, drag, fric, thrust, norm, cl, q = sumforce(
            mu, h_ft, dT_isa, v, weight, alpha_ini, poussee, cg_mac_current
        )
        record_step(v, distance, a, lift, drag, fric, thrust, norm, cl, sref, weight)
        t += dt
        v_2 = dt*a + v
        distance += ((v_2 + v)/2)*dt
        v = v_2
        weight -= fuel_consumption/60 * dt
        if flag: break
        
    # --- PHASE 2 ---
    while v < v_r:
        
        cg_mac_current = compute_cg_mac(weight)
        
        a, flag, lift, drag, fric, thrust, norm, cl, q = sumforce(
            mu, h_ft, dT_isa, v, weight, alpha_trans, poussee, cg_mac_current
        )
        record_step(v, distance, a, lift, drag, fric, thrust, norm, cl, sref,weight)
        t += dt
        v_2 = dt*a + v
        distance += ((v_2 + v)/2)*dt
        v = v_2
        weight -= fuel_consumption/60 * dt
        if flag: break

    # --- PHASE 3 ---
    while v < v_lof:
        
        cg_mac_current = compute_cg_mac(weight)
        
        a, flag, lift, drag, fric, thrust, norm, cl, q = sumforce(
            mu, h_ft, dT_isa, v, weight, alpha_rot, poussee, cg_mac_current
        )
        record_step(v, distance, a, lift, drag, fric, thrust, norm, cl, sref,weight)
        t += dt
        v_2 = dt*a + v
        distance += ((v_2 + v)/2)*dt
        v = v_2
        weight -= fuel_consumption/60 * dt
        if flag: break
    
    C_L = weight/(q*sref)
    print('--------- CL FINAL ---------- = ', C_L)
    print('---------- Weight Final---------=', weight)
    return distance, t, history

# --- Exécution ---
dist_totale, t_total, h = groundrun(v_trans, weight, alpha_trans, alpha_ini, h_ft, dT_isa, 1 ,CL_max)

print(f"Distance totale: {dist_totale:.2f} ft")
print(f"Temps total: {t_total:.2f} s")

# --- PLOTTING (Mise à jour pour 4 graphiques) ---
# On passe à 4 lignes (nrows=4) et on augmente la hauteur (figsize)
fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

v_kts = np.array(h['v']) / 1.6878

# Graphique 1 : Bilan des Forces
axs[0].plot(v_kts, h['thrust'], label='Thrust', color='green', linewidth=2)
axs[0].plot(v_kts, h['drag'], label='Drag', color='red')
axs[0].plot(v_kts, h['friction'], label='Friction', color='orange', linestyle='--')
axs[0].set_ylabel('Forces [lbf]')
axs[0].set_title('Forces analysis')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.3)


# Graphique 2 : Accélération
#axs[1].plot(v_kts, h['a'], color='purple', linewidth=2)
#axs[1].set_ylabel('Acceleration [ft/s²]')
#axs[1].grid(True, alpha=0.3)


# Graphique 4 : Portance vs Poids
axs[1].plot(v_kts, h['lift'], label='Lift', color='blue')
axs[1].plot(v_kts, h['weight'], label='Weight', color='black', linestyle='--')
axs[1].plot(v_kts, h['norme'], label='Weight on wheels', color='grey', linestyle=':')
axs[1].set_xlabel('Speed [kts]')
axs[1].set_ylabel('Vertical forces [lbf]')
axs[1].legend(loc='upper left')
axs[1].grid(True, alpha=0.3)

# Ajout des lignes verticales sur TOUS les graphiques
#for ax in axs:
   #ax.axvline(x=v_trans, color='k', linestyle='--', alpha=0.5)
    #ax.axvline(x=v_r, color='k', linestyle='--', alpha=0.5)

# On met les textes seulement sur le graphe du haut pour ne pas surcharger
#axs[0].text(v_trans, axs[0].get_ylim()[0], ' Trans', rotation=90, verticalalignment='bottom')
#axs[0].text(v_r, axs[0].get_ylim()[0], ' Rot', rotation=90, verticalalignment='bottom')

plt.tight_layout()
plt.savefig('analyse_decollage.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# ZONE DE PLOTTING 3D : ALTITUDE vs ISA vs DISTANCE
# =============================================================================

def generate_3d_takeoff_plot():
    # --- IMPORTS ---
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    
    print("\n--- Génération du Graphique 3D : Altitude vs ISA vs Distance ---")

    # 1. Définition du Design Space (Grille de calcul)
    # Altitude de 0 à 10 000 ft
    alt_range = np.linspace(0, 10000, 10)
    # Delta ISA de -20 à +35 °C
    isa_range = np.linspace(-20, 35, 10)
    
    ALT_GRID, ISA_GRID = np.meshgrid(alt_range, isa_range)
    DIST_GRID = np.zeros_like(ALT_GRID)
    
    # 2. Boucle de Simulation
    rows, cols = ALT_GRID.shape
    for i in range(rows):
        for j in range(cols):
            h_val = ALT_GRID[i, j]
            dt_val = ISA_GRID[i, j]
            
            # Appel de la simulation 'groundrun'
            # On garde les paramètres avion fixes (Poids max, CL_max, etc.)
            # On ne fait varier que l'environnement (h_ft et dT_isa)
            dist, _, _ = groundrun(
                v_trans,        
                weight,       # Poids fixe (MAX_TO)
                alpha_trans, 
                alpha_ini, 
                h_val,        # Variable : Altitude
                dt_val,       # Variable : Delta ISA
                1.0,          # Pleine puissance
                CL_max        
            )
            
            DIST_GRID[i, j] = dist

    # 3. Création du Plot 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracé de la surface
    surf = ax.plot_surface(
        ALT_GRID, 
        ISA_GRID, 
        DIST_GRID, 
        cmap='viridis',
        edgecolor='k',        
        linewidth=0.2,
        alpha=0.9
    )
    
    # Mise en forme des axes
    ax.set_title(f"Takeoff Ground Run vs Altitude & $\\Delta$ISA\n(Weight={weight:.0f} lb)", fontsize=14)
    ax.set_xlabel('Airport Altitude [ft]', fontsize=11, labelpad=10)
    ax.set_ylabel('Delta ISA [°C]', fontsize=11, labelpad=10)
    ax.set_zlabel('Ground Run Distance [ft]', fontsize=11, labelpad=10)
    
    # Barre de couleur
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Distance [ft]')
    
    # Angle de vue (Elevation, Azimut)
    ax.view_init(elev=30, azim=-135)
    
    plt.tight_layout()
    plt.savefig('takeoff_dist_vs_alt_isa.png', dpi=300)
    plt.show()
    print("Graphique 3D généré et sauvegardé sous 'takeoff_dist_vs_alt_isa.png'.")

if __name__ == "__main__":
    generate_3d_takeoff_plot()