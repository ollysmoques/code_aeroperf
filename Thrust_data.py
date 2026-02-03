# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 23:06:47 2025

@author: edoua
"""

import numpy as np
import matplotlib.pyplot as plt
from atmosphere import *


fc_default, geom, aero = get_default_inputs()
from Mission_parameters import MISSION_HEIGHT_FT


def thrust_sw400pro_ft_lbf(h_ft, dT_isa, power_setting):
    atm = atmosphere(h_ft,dT_isa,mode='delta_isa')

    T_sl = geom.Motor.Max_thrust_sl
    Fuel_flow = geom.Motor.Fuelflow_power_setting
    
    
    sigma = atm['rapport_densité']
    Thrust_total = T_sl * (sigma)
    
    Thrust_output = power_setting * Thrust_total
    
    Fuel_consumption = Fuel_flow*power_setting
    return Thrust_output,Fuel_consumption




thrust, Fuel_consumption = thrust_sw400pro_ft_lbf(0,0,1)



# ======================
#   2D Plot Thrust vs Altitude
# ======================
altitudes = np.linspace(0, MISSION_HEIGHT_FT, 100)

# Temp variations to compare
dT_list = [-20, 0, +20]   # °C
power_setting = 0.7       # max power thrust


plt.figure(figsize=(10,6))

for dT in dT_list:
    thrust_vals = []
    for h in altitudes:
        thrust, fc = thrust_sw400pro_ft_lbf(h, dT, power_setting)
        thrust_vals.append(thrust)

    plt.plot(altitudes, thrust_vals, label=f"ΔISA = {dT:+}°C", linewidth=2)

plt.xlabel("Altitude [ft]")
plt.ylabel("Thrust [lbf]")
plt.title("SW400 Pro Thrust vs Altitude for Different ΔISA")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()