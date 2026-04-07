"""Quick takeoff diagnostic — traces forces at key speeds."""
import numpy as np
from atmosphere import atmosphere
from Aircraft_data import get_default_inputs
from helpers import get_wing_cl_cd_from_aoa

fc, geom, aero = get_default_inputs()

W      = aero.MAX_TO
rho_sl = atmosphere(0, 0, 'delta_isa')['densite']
S      = geom.S_ref
CL_max = aero.cl_max_15
T      = geom.Motor.Max_thrust_sl  # sea-level static thrust
alpha  = aero.alpha_rot_deg
mu     = aero.mu_TO

print("="*60)
print("TAKEOFF DIAGNOSTIC")
print("="*60)
print(f"Weight (MAX_TO)   = {W:.1f} lb")
print(f"S_ref             = {S:.1f} ft²")
print(f"CL_max (f15)      = {CL_max:.4f}")
print(f"Thrust (SL)       = {T:.1f} lbf")
print(f"alpha_rot          = {alpha:.1f}°")
print(f"mu                = {mu}")
print(f"rho (SL)          = {rho_sl:.6f} slug/ft³")

Vs  = np.sqrt(2*W / (rho_sl * S * CL_max))
Vr  = 1.1 * Vs
Vlof = 1.2 * Vs

print(f"\nVs   = {Vs:.2f} ft/s  ({Vs/1.6878:.1f} kts)")
print(f"V_R  = {Vr:.2f} ft/s  ({Vr/1.6878:.1f} kts)")
print(f"V_LOF= {Vlof:.2f} ft/s ({Vlof/1.6878:.1f} kts)")

# CL and CD from polar at rotation angle
CL_at_rot, CD_at_rot = get_wing_cl_cd_from_aoa(alpha, 15)
print(f"\nWing polar at {alpha}° (flaps 15):")
print(f"  CL = {CL_at_rot:.4f}")
print(f"  CD = {CD_at_rot:.5f}")

print("\n" + "-"*60)
print(f"{'V (ft/s)':>10} {'V (kts)':>8} {'q (psf)':>8} {'Lift':>8} {'T*sinA':>8} "
      f"{'Total_up':>8} {'Weight':>8} {'Norme':>8} {'Liftoff?':>10}")
print("-"*60)

for V in np.arange(60, 130, 2):
    q = 0.5 * rho_sl * V**2
    L = CL_at_rot * q * S
    Tsin = T * np.sin(np.radians(alpha))
    total_up = L + Tsin
    norme = W - total_up
    if norme < 0:
        norme = 0
    liftoff = "YES" if total_up >= W else "no"
    print(f"{V:10.1f} {V/1.6878:8.1f} {q:8.2f} {L:8.1f} {Tsin:8.1f} "
          f"{total_up:8.1f} {W:8.1f} {norme:8.1f} {liftoff:>10}")
    if total_up >= W:
        print(f"\n>>> Liftoff at {V:.1f} ft/s ({V/1.6878:.1f} kts) — "
              f"that's {V/Vs:.2f} × Vs")
        break
else:
    print("\n>>> Aircraft NEVER lifts off at alpha_rot with this config!")

# What alpha would be needed at V_LOF?
print("\n" + "="*60)
print("WHAT ALPHA IS NEEDED AT V_LOF?")
q_lof = 0.5 * rho_sl * Vlof**2
CL_needed = (W - T*np.sin(np.radians(alpha))) / (q_lof * S)
print(f"  q at V_LOF = {q_lof:.2f} psf")
print(f"  CL needed  = {CL_needed:.4f}")

# Find the alpha from polar
alphas_test = np.arange(0, 23, 0.5)
for a in alphas_test:
    cl_test, _ = get_wing_cl_cd_from_aoa(a, 15)
    if cl_test >= CL_needed:
        print(f"  → alpha needed ≈ {a:.1f}° (CL={cl_test:.4f})")
        print(f"  → Your alpha_rot is {alpha}° — "
              f"{'OK' if alpha >= a else 'TOO LOW, increase alpha_rot!'}")
        break
