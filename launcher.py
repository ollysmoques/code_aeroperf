# -*- coding: utf-8 -*-
"""
AeroPerf Launcher — Interface de sélection.
Le script choisi s'exécute dans la console de l'IDE après fermeture du menu.
"""

import tkinter as tk
from tkinter import ttk
import runpy
import os
import sys

# ============================================================
# CONFIGURATION DES MODULES
# ============================================================

MODULES = [
    {
        "name": "🛫 Mission complète (Range libre)",
        "file": "Flight_phases.py",
        "desc": (
            "Simule le profil de mission complet : décollage, accélération, "
            "montée, croisière (range libre), descente, approche et atterrissage. "
            "Produit les graphiques de poids, altitude, CG et thrust %."
        ),
        "detail": (
            "FLIGHT_PHASES.PY — Simulation de mission à range libre\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Ce script simule un vol complet de l'avion en 7 phases :\n\n"
            "  1. Décollage (groundrun) — Roulement au sol jusqu'au décollage.\n"
            "     Utilise la poussée max (100%) et CL_max avec flaps.\n\n"
            "  2. Accélération — Transition de la vitesse de décollage vers\n"
            "     la vitesse de montée (VY = best ROC speed).\n\n"
            "  3. Montée — De 5 ft à l'altitude de croisière (MISSION_HEIGHT_FT).\n"
            "     Power setting configurable (CLIMB_POWER_SETTING).\n\n"
            "  4. Croisière — Vol en palier jusqu'à épuisement du carburant\n"
            "     (moins la réserve). Le range est calculé librement.\n\n"
            "  5. Descente — De l'altitude de croisière à 1000 ft (idle power).\n\n"
            "  6. Approche & Arrondi — De 1000 ft au sol avec glide slope -3°.\n\n"
            "  7. Roulement à l'atterrissage — Freinage jusqu'à l'arrêt.\n\n"
            "Graphiques produits :\n"
            "  • Poids vs Temps (fuel burn profile)\n"
            "  • Altitude vs Temps (flight profile)\n"
            "  • Sensibilité ISA (-15°C, 0°C, +15°C)\n"
            "  • Position du CG vs Temps\n"
            "  • CG vs Poids\n"
            "  • Thrust % vs Phase de vol\n"
            "  • Stall warnings par phase\n\n"
            "Paramètres clés :\n"
            "  • h_cruise : défini dans Mission_parameters.py\n"
            "  • VY, V_cruise_CAS : vitesses en kts\n"
            "  • Données avion : Aircraft_data.py"
        ),
        "category": "Mission",
    },
    {
        "name": "⏱️ Mission à temps imposé",
        "file": "Flight_phases_imposed_time.py",
        "desc": (
            "Simule le profil de mission avec un temps de vol imposé. "
            "La durée de croisière est ajustée pour respecter le temps total. "
            "Produit les graphiques de mission + analyse sensibilité ISA."
        ),
        "detail": (
            "FLIGHT_PHASES_IMPOSED_TIME.PY — Mission à durée imposée\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Variante de Flight_phases.py où le temps total de vol est fixé\n"
            "(variable TIME_IMPOSED_MIN dans le __main__).\n\n"
            "La durée de croisière est calculée par itération :\n"
            "  t_cruise = t_total - t_takeoff - t_accel - t_climb\n"
            "             - t_descent - t_approach - t_landing\n\n"
            "Comme t_descent dépend du poids (qui dépend de t_cruise),\n"
            "une boucle d'itération converge vers la solution.\n\n"
            "Graphiques produits :\n"
            "  • Profil altitude vs temps (sensibilité ISA)\n"
            "  • Consommation de fuel par phase (histogramme)\n"
            "  • Distribution du fuel (camembert)\n"
            "  • Fuel flow et specific range par phase\n"
            "  • Historique poids et CG\n"
            "  • Forces aile/empennage par phase\n"
            "  • Stall warnings\n\n"
            "Paramètre clé à modifier :\n"
            "  • TIME_IMPOSED_MIN (ligne ~667) : durée totale en minutes"
        ),
        "category": "Mission",
    },
    {
        "name": "📈 Analyse ROC (Rate of Climb)",
        "file": "ROC.py",
        "desc": (
            "Calcule le taux de montée (ROC) vs CAS et altitude. "
            "Produit une surface 3D, courbes 2D et identifie Vy."
        ),
        "detail": (
            "ROC.PY — Rate of Climb Analysis\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Calcule le taux de montée corrigé (ROC) selon la formule :\n"
            "  ROC = TAS × (Thrust − Drag) / Weight / (1 + AF)\n"
            "  ROC_corrigé = ROC × (T_std / T)\n\n"
            "Où AF est le facteur d'accélération qui tient compte\n"
            "de l'énergie cinétique absorbée en montée à CAS constante.\n\n"
            "Fonctions principales :\n"
            "  • ROC() — Calcul ponctuel du ROC\n"
            "  • montee() — Simulation de montée pas à pas (dh=5 ft)\n"
            "  • descente() — Simulation de descente pas à pas\n"
            "  • acceleration() — Phase d'accélération en palier\n"
            "  • find_initial_weight_for_descent() — Itération inverse\n\n"
            "Graphiques produits :\n"
            "  • Surface 3D : ROC vs CAS vs Altitude\n"
            "  • Courbe 2D : ROC vs CAS à 0 ft et altitude mission\n"
            "  • Identification de Vy (best rate of climb speed)\n\n"
            "Seuil de sécurité : ROC < 300 ft/min → montée interrompue"
        ),
        "category": "Performance",
    },
    {
        "name": "✈️ Analyse de traînée",
        "file": "drag_analysis.py",
        "desc": (
            "Décomposition de la traînée pour 3 conditions (décollage, croisière, "
            "atterrissage). Diagrammes camembert par composante."
        ),
        "detail": (
            "DRAG_ANALYSIS.PY — Décomposition de la traînée\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Effectue un calcul d'équilibre (alpha_eq) pour 3 conditions :\n"
            "  • Décollage : h=0 ft, V=51.4 kts, flaps 15°\n"
            "  • Croisière : h=10000 ft, V=108 kts, clean\n"
            "  • Atterrissage : h=0 ft, V=75 kts, flaps 30°\n\n"
            "Décomposition par composante :\n"
            "  ─ Traînée parasite (friction + forme + interférence) :\n"
            "      Aile, Empennage H, Empennage V, Fuselage,\n"
            "      Nacelle, Pylône, Train principal, Train arrière\n"
            "  ─ Traînée induite :\n"
            "      Aile (polaire), Empennage (trim)\n\n"
            "Chaque condition produit :\n"
            "  • Résumé console (alpha_eq, D_total, CD_total)\n"
            "  • Diagramme camembert de répartition\n\n"
            "Dépendances : Cdmin.py, induced_equilibrium.py, helpers.py"
        ),
        "category": "Aérodynamique",
    },
    {
        "name": "🔍 Diagnostic décollage",
        "file": "diag_takeoff.py",
        "desc": (
            "Vérifie si l'avion peut décoller. Calcule Vs, Vr, Vlof et "
            "trace les forces en fonction de la vitesse."
        ),
        "detail": (
            "DIAG_TAKEOFF.PY — Diagnostic de décollage\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Vérifie la faisabilité du décollage avec la configuration\n"
            "actuelle de l'avion (poids, CL_max, poussée, alpha_rot).\n\n"
            "Calculs effectués :\n"
            "  • Vs  = vitesse de décrochage (stall speed)\n"
            "  • Vr  = 1.1 × Vs (vitesse de rotation)\n"
            "  • Vlof = 1.2 × Vs (vitesse de décollage)\n\n"
            "Tableau de forces vs vitesse :\n"
            "  Pour chaque vitesse de 60 à 130 ft/s, affiche :\n"
            "  Portance, T×sin(α), Force totale vers le haut, Poids,\n"
            "  et indique si le décollage est possible (YES/no).\n\n"
            "Diagnostic final :\n"
            "  • À quelle vitesse l'avion décolle-t-il ?\n"
            "  • Quel alpha serait nécessaire à Vlof ?\n"
            "  • L'alpha_rot actuel est-il suffisant ?"
        ),
        "category": "Décollage",
    },
    {
        "name": "📊 Carpet plot décollage",
        "file": "takeoff_carpet_plot.py",
        "desc": (
            "Abaque de distance de décollage vs altitude aéroport "
            "et température (ISA, ISA+15, ISA+30)."
        ),
        "detail": (
            "TAKEOFF_CARPET_PLOT.PY — Abaque de décollage\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Génère un carpet plot montrant la distance de roulement\n"
            "au décollage en fonction de :\n\n"
            "  • Altitude de l'aéroport : 0 à 8000 ft (9 points)\n"
            "  • Température : ISA, ISA+15°C, ISA+30°C\n\n"
            "Conditions fixes :\n"
            "  • Poids : MTOW (poids max au décollage)\n"
            "  • Pente de piste : 0° (standard)\n"
            "  • CL_max : configuration décollage avec flaps\n\n"
            "Utilité :\n"
            "  Permet d'évaluer rapidement si l'avion peut opérer\n"
            "  depuis un aéroport donné (altitude + température).\n"
            "  Plus l'altitude et la température sont élevées,\n"
            "  plus la distance de décollage augmente.\n\n"
            "Sortie : Graphique PNG (takeoff_carpet_plot.png)"
        ),
        "category": "Décollage",
    },
    {
        "name": "🔧 Poussée moteur (Thrust data)",
        "file": "Thrust_data.py",
        "desc": (
            "Poussée du SW400 Pro vs altitude pour différentes "
            "conditions de température (ΔISA = -20, 0, +20°C)."
        ),
        "detail": (
            "THRUST_DATA.PY — Modèle de poussée moteur\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Modélise la poussée du moteur SW400 Pro :\n"
            "  Thrust = T_sl × σ × power_setting\n\n"
            "Où :\n"
            "  • T_sl = poussée max au niveau de la mer (89 lbf)\n"
            "  • σ = rapport de densité (sigma = ρ / ρ₀)\n"
            "  • power_setting = réglage de puissance (0 à 1)\n\n"
            "Consommation de carburant :\n"
            "  Fuel_flow = base_flow × power_setting\n"
            "  (base_flow = 2.3 lb/min à 100%)\n\n"
            "Graphique produit :\n"
            "  Poussée vs Altitude pour ΔISA = -20, 0, +20°C\n"
            "  Montre la dégradation de poussée avec l'altitude\n"
            "  et la sensibilité à la température.\n\n"
            "Ce modèle est utilisé par toutes les phases de vol\n"
            "(ROC, croisière, descente, etc.)."
        ),
        "category": "Propulsion",
    },
]

# ============================================================
# STYLE
# ============================================================
BG_MAIN       = "#1e1e2e"
BG_CARD       = "#2a2a3e"
BG_CARD_HOVER = "#363650"
BG_INFO       = "#181825"
FG_TITLE      = "#cdd6f4"
FG_DESC       = "#a6adc8"
FG_CAT        = "#89b4fa"
ACCENT        = "#f5c2e7"
GREEN_RUN     = "#a6e3a1"
BORDER        = "#45475a"
INFO_BLUE     = "#74c7ec"


def show_info_popup(root, mod):
    """Ouvre une fenêtre popup avec les détails du module."""
    popup = tk.Toplevel(root)
    popup.title(f"ℹ  {mod['name']}")
    popup.geometry("550x500")
    popup.configure(bg=BG_INFO)
    popup.resizable(True, True)

    # Centrer par rapport à la fenêtre parente
    popup.transient(root)
    popup.grab_set()

    # Header
    tk.Label(
        popup, text=mod["name"],
        font=("Segoe UI", 14, "bold"), fg=ACCENT, bg=BG_INFO,
        anchor="w", padx=16,
    ).pack(fill="x", pady=(12, 4))

    tk.Label(
        popup, text=f"📄 {mod['file']}  —  [{mod['category']}]",
        font=("Consolas", 10), fg=FG_CAT, bg=BG_INFO,
        anchor="w", padx=16,
    ).pack(fill="x")

    # Séparateur
    tk.Frame(popup, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)

    # Contenu scrollable
    text_frame = tk.Frame(popup, bg=BG_INFO)
    text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

    text = tk.Text(
        text_frame, wrap="word",
        font=("Consolas", 10), bg=BG_INFO, fg=FG_TITLE,
        relief="flat", highlightthickness=0,
        padx=8, pady=8,
    )
    scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)

    text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    text.insert("1.0", mod.get("detail", "Pas de détails disponibles."))
    text.config(state="disabled")

    # Bouton fermer
    tk.Button(
        popup, text="Fermer", font=("Segoe UI", 10, "bold"),
        fg=BG_MAIN, bg=FG_DESC, relief="flat",
        padx=20, pady=4, cursor="hand2",
        command=popup.destroy,
    ).pack(pady=(0, 12))


def show_launcher():
    """
    Affiche le menu Tkinter et retourne le nom du fichier sélectionné.
    Retourne None si la fenêtre est fermée sans sélection.
    """
    selected = {"file": None}

    root = tk.Tk()
    root.title("AeroPerf Launcher — PI IV")
    root.geometry("780x720")
    root.configure(bg=BG_MAIN)
    root.minsize(650, 500)

    def on_select(filename):
        selected["file"] = filename
        root.destroy()

    # ---- HEADER ----
    header = tk.Frame(root, bg=BG_MAIN, pady=14)
    header.pack(fill="x")

    tk.Label(
        header, text="✈  AeroPerf Launcher",
        font=("Segoe UI", 22, "bold"), fg=ACCENT, bg=BG_MAIN,
    ).pack(side="left", padx=20)

    tk.Label(
        header, text="Sélectionnez un module à exécuter",
        font=("Segoe UI", 11), fg=FG_DESC, bg=BG_MAIN,
    ).pack(side="left", padx=10)

    # ---- SCROLLABLE ----
    container = tk.Frame(root, bg=BG_MAIN)
    container.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    canvas = tk.Canvas(container, bg=BG_MAIN, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=BG_MAIN)

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    canvas.bind_all("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    # ---- CARTES ----
    for mod in MODULES:
        card = tk.Frame(inner, bg=BG_CARD, padx=16, pady=12,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", padx=6, pady=4)

        # Hover
        def make_hover(c):
            def enter(e):
                c.config(bg=BG_CARD_HOVER)
                for w in c.winfo_children():
                    try: w.config(bg=BG_CARD_HOVER)
                    except tk.TclError: pass
            def leave(e):
                c.config(bg=BG_CARD)
                for w in c.winfo_children():
                    try: w.config(bg=BG_CARD)
                    except tk.TclError: pass
            return enter, leave

        ent, lv = make_hover(card)
        card.bind("<Enter>", ent)
        card.bind("<Leave>", lv)

        top = tk.Frame(card, bg=BG_CARD)
        top.pack(fill="x")

        tk.Label(top, text=f"[{mod['category']}]",
                 font=("Segoe UI", 9), fg=FG_CAT, bg=BG_CARD).pack(side="left")
        tk.Label(top, text=f"  {mod['name']}",
                 font=("Segoe UI", 12, "bold"), fg=FG_TITLE, bg=BG_CARD).pack(side="left")

        # ---- BOUTON ▶ LANCER ----
        tk.Button(
            top, text="▶ Lancer", font=("Segoe UI", 10, "bold"),
            fg=BG_MAIN, bg=GREEN_RUN, relief="flat",
            padx=14, pady=3, cursor="hand2",
            command=lambda f=mod["file"]: on_select(f),
        ).pack(side="right")

        # ---- BOUTON ℹ INFO ----
        tk.Button(
            top, text=" ℹ ", font=("Segoe UI", 10, "bold"),
            fg=BG_MAIN, bg=INFO_BLUE, relief="flat",
            padx=6, pady=3, cursor="hand2",
            command=lambda m=mod: show_info_popup(root, m),
        ).pack(side="right", padx=(0, 6))

        tk.Label(card, text=mod["desc"],
                 font=("Segoe UI", 9), fg=FG_DESC, bg=BG_CARD,
                 justify="left", anchor="w", wraplength=650).pack(fill="x", pady=(6, 0))

        tk.Label(card, text=f"📄 {mod['file']}",
                 font=("Consolas", 9), fg="#585b70", bg=BG_CARD,
                 anchor="w").pack(fill="x", pady=(3, 0))

    # Attendre la sélection ou la fermeture
    root.mainloop()

    return selected["file"]


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    while True:
        # 1. Afficher le menu et récupérer le choix
        chosen_file = show_launcher()

        if chosen_file is None:
            print("Aucun module sélectionné. Fin.")
            break

        # 2. Exécuter le script choisi dans le processus courant
        filepath = os.path.join(script_dir, chosen_file)
        print(f"\n{'='*60}")
        print(f"▶  Exécution de : {chosen_file}")
        print(f"{'='*60}\n")

        try:
            runpy.run_path(filepath, run_name="__main__")
        except Exception as e:
            print(f"\n❌ Erreur durant l'exécution : {e}")

        print(f"\n{'='*60}")
        print(f"✅ {chosen_file} terminé — réouverture du launcher...")
        print(f"{'='*60}\n")
