import numpy as np
import matplotlib.pyplot as plt
from take_off_run import groundrun
from Aircraft_data import get_default_inputs

def generate_takeoff_carpet_plot():
    # 1. Setup Inputs
    fc_default, geom, aero = get_default_inputs()
    
    # Parameters constant for this analysis (MTOW condition)
    weight      = aero.MAX_TO
    v_trans_fts = aero.v_trans_kts * 1.6878
    alpha_trans = aero.alpha_trans_deg
    alpha_ini   = aero.alpha_ini_deg
    slope       = 0.0 # Standard runway
    CL_max      = aero.CL_max_TO
    
    # 2. Define Ranges
    altitudes_ft = np.linspace(0, 8000, 9) # 0 to 8000 ft in 1000ft steps
    isa_deltas   = [0, 15, 30]             # ISA, ISA+15, ISA+30
    
    colors = {0: 'black', 15: '#1f77b4', 30: '#d62728'}
    labels = {0: 'ISA', 15: 'ISA + 15°C', 30: 'ISA + 30°C'}
    
    results = {dt: [] for dt in isa_deltas}
    
    print("Calculating Takeoff Performance...")
    
    # 3. Calculate Loop
    for dt in isa_deltas:
        print(f"  > Processing {labels[dt]}...")
        for h in altitudes_ft:
            dist_ft, _, _ = groundrun(
                v_trans_fts, weight, alpha_trans, alpha_ini, h, dt, slope, CL_max
            )
            results[dt].append(dist_ft)

    # 4. Plotting
    plt.figure(figsize=(10, 7))
    
    for dt in isa_deltas:
        plt.plot(altitudes_ft, results[dt], 
                 label=labels[dt], 
                 color=colors[dt], 
                 linewidth=2, 
                 marker='o', 
                 markersize=5)
        
        # Annotate end points
        plt.text(altitudes_ft[-1], results[dt][-1], 
                 f" {results[dt][-1]:.0f} ft", 
                 color=colors[dt], 
                 va='center', 
                 fontweight='bold')

    plt.title(f"Takeoff Performance Carpet Plot\n(MTOW = {weight:.0f} lb)", fontsize=14, fontweight='bold')
    plt.xlabel("Airport Altitude [ft]", fontsize=12)
    plt.ylabel("Takeoff Ground Roll [ft]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Add minor grid
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.4)
    
    plt.tight_layout()
    
    filename = "takeoff_carpet_plot.png"
    plt.savefig(filename, dpi=300)
    print(f"\nPlot generated: {filename} (Displaying now...)")
    plt.show()

if __name__ == "__main__":
    generate_takeoff_carpet_plot()
