import numpy as np 
from induced_equilibrium import *


def atmosphere(h, variable, mode):

    '''
    fonction qui retourne les propriétés de l'atmosphère 
    @h: float altitude basée sur la pression
    @variable: float valeur de température ou de delta_isa
    @mode: string choix delta_isa ou temperature

    return
    @atmosphere: dictionnaire de toutes les propriétés de l'atmosphère
    '''

    assert mode == 'temperature' or mode == 'delta_isa'

    temperature_zero = 15.0                                 #C
    temperature_zero_kelvin = temperature_zero + 273.15     #kelvin 
    pression_zero = 2116.22                                 #lb/ft^2
    rho_zero = 0.002377                                     #slug/ft^3
    lambdaa = 0.0019812                                     #C/ft


    if h <= 36089:
        
        theta_std = 1 - 6.87535e-6*h
        temperature_std = temperature_zero - lambdaa*h

        if mode == 'temperature':

            delta_isa = variable - temperature_std
            temperature = variable
            temperature_kelvin = temperature + 273.15
            theta = temperature_kelvin/temperature_zero_kelvin 
            delta = theta_std**5.2559
            sigma = delta/theta
        
        elif mode == 'delta_isa':

            temperature = variable + temperature_std
            delta_isa = variable
            temperature_kelvin = temperature + 273.15
            theta = temperature_kelvin/temperature_zero_kelvin
            delta = theta_std**5.2559
            sigma = delta/theta

    elif 36089 < h < 65617: 
        
        temperature_std = -56.5 

        if mode == 'temperature':

            delta_isa = variable - temperature_std
            temperature = variable
            temperature_kelvin = temperature + 273.15
            theta = temperature_kelvin/temperature_zero_kelvin
            delta = 0.22336*np.exp(-(h - 36089)/(20806))
            sigma = delta/theta
        
        elif mode == 'delta_isa':

            temperature = variable + temperature_std
            delta_isa = variable
            temperature_kelvin = temperature + 273.15
            theta = temperature_kelvin/temperature_zero_kelvin
            delta = 0.22336*np.exp(-(h - 36089)/(20806))
            sigma = delta/theta
            
        
    pression = delta * pression_zero
    rho = sigma * rho_zero

    atmosphere = {'altitude pression': h,
    'rapport_temperature': round(theta,4),
    'rapport_pression':round(delta, 4), 
    'rapport_densité':round(sigma,4),
    'temperature_K': round(temperature_kelvin,4),
    'temperature_C': round(temperature,4), 
    'delta_ISA': round(delta_isa, 4),
    'pression': round(pression, 4),
    'densite': round(rho,8)}

    return atmosphere

def vitesses(atmosphere, vitesse, type, masse, Sref = 520, MAC = 8.286):
    
    '''
    fonction qui retourne les propriétés de vitesse de l'avion

    entrées
    @atmosphere: dictionnaire, des propriétés de l'atmosphere
    @vitesse: float, valeur de vitesse selon le type énoncé
    @type: string, type de donnée de vitesse
    @masse: float, masse de l'avion
    @Sref: float, surface de référence de l'avion 
    @MAC: float, longueur de référence de l'avion, corde moyenne

    return
    @vitesses: dictionnaire de vitesse de l'avion 
    '''

    theta = atmosphere['rapport_temperature']
    delta = atmosphere['rapport_pression']
    sigma = atmosphere['rapport_densité']
    T_k = atmosphere['temperature_K']
    pression = atmosphere['pression']
    densite = atmosphere['densite']

    gamma = 1.4 #-
    a_zero = 661.48 #knots
    p_zero = 2116.22 #lb/ft^2
    a = a_zero * np.sqrt(theta)

    assert type == 'mach' or type == 'vraie' or type == 'equivalente' or type == 'calibree'

    if type == 'mach':
        
        mach = vitesse
        v_vrai = vitesse * a
        v_eq = v_vrai * np.sqrt(sigma)
        pression_impact = pression * ((1 + mach**2/5)**(1/0.2857)-1 )
        v_c = a_zero * np.sqrt(5 * ((pression_impact/p_zero +1)**0.2857 -1))

    if type == 'vraie':

        v_vrai = vitesse
        mach = v_vrai/ a
        v_eq = v_vrai * np.sqrt(sigma)
        pression_impact = pression * ((1 + mach**2/5)**(1/0.2857)-1 )
        v_c = a_zero * np.sqrt(5 * ((pression_impact/p_zero +1)**0.2857 -1))
    
    if type == 'equivalente':

        v_eq = vitesse
        v_vrai = v_eq/np.sqrt(sigma)
        mach = v_vrai/a 
        pression_impact = pression * ((1 + mach**2/5)**(1/0.2857)-1 )
        v_c = a_zero * np.sqrt(5 * ((pression_impact/p_zero +1)**0.2857 -1))
    
    if type == 'calibree':

        v_c = vitesse
        pression_impact = p_zero * ((1 + (gamma -1)/2*(v_c/a_zero)**2)**(gamma/(gamma-1))-1)
        mach = (5*((pression_impact/pression+1)**0.2857-1))**0.5
        v_vrai = mach * a
        v_eq = v_vrai * np.sqrt(sigma)
    
    pression_tot = pression + pression_impact
    pression_dynamique = 1481.3 * delta * mach**2
    T_t_k = T_k * (1 + 0.2 * mach**2)
    T_t_c = T_t_k -273.15
    mu = (0.3125e-7 * T_k**1.5)/(T_k + 120)
    CL = masse/(pression_dynamique * Sref)

    v_vrai_ft = v_vrai * 1.6878
    v_c_ft = v_c * 1.6878
    v_eq_ft = v_eq * 1.6878
    a_ft = a * 1.6878
    reynolds = densite*MAC*v_vrai_ft/ mu 
    
    vitesses = {'vitesse du son kts': a, 
    'vitesse du son ft': a_ft, 
    'mach': mach,
    'vitesses avion kts': {'mach': mach, 
                        'vitesse vraie': v_vrai,
                        'vitesse equivalente': v_eq,
                        'vitesse calibree': v_c},
    'vitesses avion ft': {'mach': mach, 
                        'vitesse vraie': v_vrai_ft,
                        'vitesse equivalente': v_eq_ft,
                        'vitesse calibree': v_c_ft},
    'pression totale': pression_tot,
    'pression dynamique': pression_dynamique,
    'pression impact': pression_impact, 
    'temperature totale celsius': T_t_c,
    'temperature totale kelvin': T_t_k,
    'viscosité dynamique': mu,
    'reynolds': reynolds,
    'coefficient de portance': CL
    }

    return vitesses 

def get_cl_value_helper(file, mode, speed_dict, alpha = 4):

    assert mode in ['weight', 'alpha']
    
    if mode == 'weight':
        cl_value = speed_dict["coefficient de portance"]
    if mode == 'alpha':
        cl_value =  get_cl_from_polar(file,alpha)
        
    return cl_value

