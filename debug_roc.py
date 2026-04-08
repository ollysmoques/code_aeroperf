"""Debug script to check ROC values at different temperatures"""
import sys
# Suppress matplotlib plots from imported modules
import matplotlib
matplotlib.use('Agg')

from atmosphere import atmosphere, vitesses
from Aircraft_data import get_default_inputs
from cg_shift import compute_cg_mac
from Thrust_data import thrust_sw400pro_ft_lbf
from total_drag import drag_total

fc, geom, aero = get_default_inputs()
sref = geom.S_ref
weight = aero.MAX_TO
cg = compute_cg_mac(weight)

print(f"Weight (MTOW): {weight:.1f} lb")
print(f"Sref: {sref:.1f} ft2")
print(f"CAS_climb: 81 kts, Power: 0.75")
print(f"Max thrust SL: {geom.Motor.Max_thrust_sl:.1f} lbf")
print()

# ============================================
# 1) Detailed ROC breakdown at h=5 ft
# ============================================
print("=" * 70)
print("DETAILED ROC BREAKDOWN AT h=5 ft (start of climb)")
print("=" * 70)

for dT in [-15, 0, 15]:
    h = 5.0
    atm = atmosphere(h, dT, 'delta_isa')
    vit = vitesses(atm, 81, 'calibree', weight, sref)
    TAS_kts = vit['vitesses avion kts']['vitesse vraie']
    TAS_fps = TAS_kts * 1.6878
    Mach = vit['mach']
    
    sigma = atm['rapport_densité']
    T = atm['temperature_K']
    dT_loc = atm['delta_ISA']
    Tstd = T - dT_loc
    
    # Thrust
    thrust, fuel_flow = thrust_sw400pro_ft_lbf(h, dT, 0.75)
    
    # Drag
    cltot, drag, qpsf, _ = drag_total(h, dT, TAS_kts, weight, cg, weight, 'cruise')
    
    # Excess thrust
    excess = thrust - drag
    
    # Simple ROC (before AF correction)
    roc_simple = TAS_fps * excess / weight * 60  # ft/min
    
    # AF correction
    phi = 1/(0.7*Mach**2) * ((1+0.2*Mach**2)**3.5 - 1)/((1+0.2*Mach**2)**2.5)
    if dT == 0:
        AF = 0.7*Mach**2 * (phi - 0.190263)
    else:
        AF = 0.7*Mach**2 * (phi - 0.190263*(Tstd/T))
    
    roc_corrected = roc_simple / (1 + AF)
    roc_final = roc_corrected * (Tstd/T)
    
    print(f"\n--- dT_ISA = {dT:+d} C ---")
    print(f"  sigma      = {sigma:.4f}")
    print(f"  T          = {T:.2f} K, Tstd = {Tstd:.2f} K, Tstd/T = {Tstd/T:.4f}")
    print(f"  TAS        = {TAS_kts:.2f} kts ({TAS_fps:.2f} ft/s)")
    print(f"  Mach       = {Mach:.4f}")
    print(f"  Thrust     = {thrust:.2f} lbf")
    print(f"  CL_tot     = {cltot:.4f}")
    print(f"  Drag       = {drag:.2f} lbf")
    print(f"  Excess T-D = {excess:.2f} lbf")
    print(f"  AF         = {AF:.4f}")
    print(f"  ROC simple = {roc_simple:.1f} ft/min")
    print(f"  ROC w/ AF  = {roc_corrected:.1f} ft/min")
    print(f"  ROC final  = {roc_final:.1f} ft/min (with Tstd/T)")

# ============================================
# 2) Climb simulation
# ============================================
print("\n" + "=" * 70)
print("CLIMB SIMULATION (5 ft -> 1000 ft)")
print("=" * 70)

from ROC import montee

for dT in [-15, 0, 15]:
    t, d, fuel, w_final, hist = montee(
        hpi=5.0, hpf=1000.0, dT_isa=dT, CAS_kts=81,
        weight_initial=weight, sref=sref, power_setting=0.75
    )
    print(f"dT_ISA={dT:+3d}C : time={t:.3f} min, dist={d:.3f} NM, fuel={fuel:.3f} lb, w_final={w_final:.1f} lb")
