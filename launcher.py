# -*- coding: utf-8 -*-
"""
AeroPerf Launcher — Interface de sélection avec paramètres éditables.
Le script choisi s'exécute dans la console de l'IDE après fermeture du menu.
"""

import tkinter as tk
from tkinter import ttk
import runpy
import os
import sys
import json

# ============================================================
# CONFIGURATION DES MODULES
# ============================================================

MODULES = [
    {
        "name": "🛫 Mission complète (Range libre)",
        "file": "Flight_phases.py",
        "desc": "Mission complète : décollage → croisière (range libre) → atterrissage. Graphs de poids, altitude, CG, thrust %.",
        "detail": (
            "FLIGHT_PHASES.PY — Simulation de mission à range libre\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "7 phases : décollage, accélération, montée, croisière,\n"
            "descente, approche & arrondi, roulement atterrissage.\n\n"
            "Le range de croisière est libre (vole jusqu'à la réserve).\n\n"
            "Graphiques : poids, altitude, sensibilité ISA, CG, thrust %."
        ),
        "category": "Mission",
    },
    {
        "name": "⏱️ Mission à temps imposé",
        "file": "Flight_phases_imposed_time.py",
        "desc": "Mission avec temps de vol imposé. La croisière s'ajuste au temps total.",
        "detail": (
            "FLIGHT_PHASES_IMPOSED_TIME.PY — Mission à durée imposée\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "t_cruise = t_total - t_fixe. Boucle itérative pour converger.\n\n"
            "Paramètre clé : TIME_IMPOSED_MIN (modifiable ci-dessous).\n"
            "Graphiques : profil, fuel par phase, efficacité, CG, stall."
        ),
        "category": "Mission",
    },
    {
        "name": "📈 Analyse ROC",
        "file": "ROC.py",
        "desc": "Rate of Climb vs CAS et altitude. Surface 3D, courbes 2D, Vy.",
        "detail": (
            "ROC.PY — Rate of Climb Analysis\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "ROC = TAS × (T−D) / W / (1+AF)\n"
            "Fonctions : montee(), descente(), acceleration()\n"
            "Seuil sécurité : ROC < 300 ft/min → montée interrompue."
        ),
        "category": "Performance",
    },
    {
        "name": "✈️ Analyse de traînée",
        "file": "drag_analysis.py",
        "desc": "Décomposition traînée (décollage, croisière, atterrissage). Camemberts.",
        "detail": (
            "DRAG_ANALYSIS.PY — Décomposition de la traînée\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "3 conditions : TO (flaps 15°), Cruise (clean), Landing (flaps 30°).\n"
            "Parasite + induite par composante. Camemberts."
        ),
        "category": "Aérodynamique",
    },
    {
        "name": "🔍 Diagnostic décollage",
        "file": "diag_takeoff.py",
        "desc": "Vérifie la faisabilité du décollage. Calcule Vs, Vr, Vlof.",
        "detail": (
            "DIAG_TAKEOFF.PY — Diagnostic de décollage\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Calcule Vs, Vr (1.1×Vs), Vlof (1.2×Vs).\n"
            "Tableau forces vs vitesse. Indique si liftoff est possible."
        ),
        "category": "Décollage",
    },
    {
        "name": "🛞 Simulation roulement décollage",
        "file": "take_off_run.py",
        "desc": "Simulation pas-à-pas du roulement au sol : forces, portance, vitesse de liftoff.",
        "detail": (
            "TAKE_OFF_RUN.PY — Ground Roll Simulation\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Simulation dynamique du roulement au décollage en 3 phases :\n"
            "  1. Roulage initial (α initial) jusqu'à V_transition\n"
            "  2. Transition (α transition) jusqu'à V_rotation\n"
            "  3. Rotation (α rotation) jusqu'au liftoff (L ≥ W)\n\n"
            "Graphiques :\n"
            "  • Bilan des forces (Thrust, Drag, Friction) vs distance\n"
            "  • Portance vs Poids vs distance\n\n"
            "Sorties console : distance totale, temps, vitesse de liftoff."
        ),
        "category": "Décollage",
    },
    {
        "name": "📊 Carpet plot décollage",
        "file": "takeoff_carpet_plot.py",
        "desc": "Abaque distance de décollage vs altitude et température.",
        "detail": (
            "TAKEOFF_CARPET_PLOT.PY — Abaque de décollage\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Distance de roulement vs altitude (0-8000ft)\n"
            "et température (ISA, ISA+15, ISA+30). MTOW fixe."
        ),
        "category": "Décollage",
    },
    {
        "name": "🔧 Poussée moteur",
        "file": "Thrust_data.py",
        "desc": "SW400 Pro : poussée vs altitude pour ΔISA = -20, 0, +20°C.",
        "detail": (
            "THRUST_DATA.PY — Modèle de poussée\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Thrust = T_sl × σ × power_setting\n"
            "T_sl = 89 lbf, Fuel_flow = 2.3 lb/min @ 100%."
        ),
        "category": "Propulsion",
    },
]

# ============================================================
# PARAMÈTRES ÉDITABLES
# ============================================================
# Chaque paramètre : (clé JSON, label affiché, valeur par défaut, unité)

PARAM_GROUPS = [
    {
        "group": "Masses",
        "params": [
            ("OEW",       "OEW (Operating Empty Weight)", 218,   "lb"),
            ("FUEL_LOAD", "Fuel Load",                    75.0,  "lb"),
            ("RESERVE",   "Fuel Reserve",                 7.0,   "lb"),
            ("PAYLOAD",   "Payload",                      170,   "lb"),
        ],
    },
    {
        "group": "Mission",
        "params": [
            ("MISSION_HEIGHT_FT", "Altitude de croisière",   2000,  "ft"),
            ("h_airport",         "Altitude de l'aéroport",  0,     "ft"),
            ("dT_Isa",            "Delta ISA",               0,     "°C"),
            ("TIME_IMPOSED_MIN",  "Temps imposé (si applicable)", 25, "min"),
        ],
    },
    {
        "group": "Vitesses",
        "params": [
            ("VY",           "VY (best ROC speed)",    81,  "kts CAS"),
            ("V_cruise_CAS", "Vitesse de croisière",   108, "kts CAS"),
        ],
    },
    {
        "group": "Puissance",
        "params": [
            ("CLIMB_POWER_SETTING", "Power setting montée", 0.90, "0-1"),
        ],
    },
]

# ============================================================
# STYLE
# ============================================================
BG_MAIN       = "#1e1e2e"
BG_CARD       = "#2a2a3e"
BG_CARD_HOVER = "#363650"
BG_INFO       = "#181825"
BG_PARAM      = "#1a1a2e"
BG_ENTRY      = "#313244"
FG_TITLE      = "#cdd6f4"
FG_DESC       = "#a6adc8"
FG_CAT        = "#89b4fa"
FG_GROUP      = "#fab387"
ACCENT        = "#f5c2e7"
GREEN_RUN     = "#a6e3a1"
BORDER        = "#45475a"
INFO_BLUE     = "#74c7ec"
YELLOW        = "#f9e2af"

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_launcher_config.json")


def _write_config(param_entries):
    """Écrit les valeurs actuelles des champs dans le fichier JSON."""
    config = {}
    for key, entry in param_entries.items():
        val_str = entry.get().strip()
        try:
            # Essayer int d'abord, puis float
            if "." in val_str:
                config[key] = float(val_str)
            else:
                config[key] = int(val_str)
        except ValueError:
            config[key] = val_str
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _delete_config():
    """Supprime le fichier de config (retour aux défauts)."""
    if os.path.exists(_CONFIG_PATH):
        os.remove(_CONFIG_PATH)


def show_info_popup(root, mod):
    """Ouvre une fenêtre popup avec les détails du module."""
    popup = tk.Toplevel(root)
    popup.title(f"ℹ  {mod['name']}")
    popup.geometry("550x400")
    popup.configure(bg=BG_INFO)
    popup.transient(root)
    popup.grab_set()

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

    tk.Frame(popup, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)

    text_frame = tk.Frame(popup, bg=BG_INFO)
    text_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

    text = tk.Text(
        text_frame, wrap="word",
        font=("Consolas", 10), bg=BG_INFO, fg=FG_TITLE,
        relief="flat", highlightthickness=0, padx=8, pady=8,
    )
    scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    text.insert("1.0", mod.get("detail", "Pas de détails disponibles."))
    text.config(state="disabled")

    tk.Button(
        popup, text="Fermer", font=("Segoe UI", 10, "bold"),
        fg=BG_MAIN, bg=FG_DESC, relief="flat",
        padx=20, pady=4, cursor="hand2", command=popup.destroy,
    ).pack(pady=(0, 12))


def show_launcher():
    """
    Affiche le menu Tkinter avec panneau de paramètres.
    Retourne le nom du fichier sélectionné, ou None.
    """
    selected = {"file": None}
    param_entries = {}  # key -> tk.Entry

    root = tk.Tk()
    root.title("AeroPerf Launcher — PI IV")
    root.geometry("900x780")
    root.configure(bg=BG_MAIN)
    root.minsize(750, 600)

    def on_select(filename):
        _write_config(param_entries)
        selected["file"] = filename
        root.destroy()

    # ================================================================
    # HEADER
    # ================================================================
    header = tk.Frame(root, bg=BG_MAIN, pady=10)
    header.pack(fill="x")

    tk.Label(
        header, text="✈  AeroPerf Launcher",
        font=("Segoe UI", 20, "bold"), fg=ACCENT, bg=BG_MAIN,
    ).pack(side="left", padx=20)

    tk.Label(
        header, text="Paramètres + Modules",
        font=("Segoe UI", 11), fg=FG_DESC, bg=BG_MAIN,
    ).pack(side="left", padx=10)

    # ================================================================
    # MAIN PANED (paramètres à gauche, modules à droite)
    # ================================================================
    main_pane = tk.PanedWindow(
        root, orient="horizontal", bg=BG_MAIN,
        sashwidth=4, sashrelief="flat",
    )
    main_pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ================================================================
    # LEFT: PANNEAU PARAMÈTRES
    # ================================================================
    left_frame = tk.Frame(main_pane, bg=BG_PARAM, width=320)
    main_pane.add(left_frame, minsize=280)

    # Header paramètres
    param_header = tk.Frame(left_frame, bg=BG_PARAM)
    param_header.pack(fill="x", padx=10, pady=(10, 4))

    tk.Label(
        param_header, text="⚙  Paramètres de simulation",
        font=("Segoe UI", 12, "bold"), fg=YELLOW, bg=BG_PARAM,
    ).pack(side="left")

    # Bouton reset
    tk.Button(
        param_header, text="↺ Reset",
        font=("Segoe UI", 9), fg=BG_MAIN, bg=FG_DESC,
        relief="flat", padx=8, cursor="hand2",
        command=lambda: _reset_params(param_entries),
    ).pack(side="right")

    tk.Frame(left_frame, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)

    # Scrollable param area
    param_canvas = tk.Canvas(left_frame, bg=BG_PARAM, highlightthickness=0)
    param_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=param_canvas.yview)
    param_inner = tk.Frame(param_canvas, bg=BG_PARAM)

    param_inner.bind("<Configure>",
                     lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all")))
    param_canvas.create_window((0, 0), window=param_inner, anchor="nw")
    param_canvas.configure(yscrollcommand=param_scroll.set)
    param_canvas.pack(side="left", fill="both", expand=True)
    param_scroll.pack(side="right", fill="y")

    # Charger les valeurs existantes du config (si fichier existe)
    existing_cfg = {}
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
        except Exception:
            pass

    # Créer les champs par groupe
    for group in PARAM_GROUPS:
        # Titre du groupe
        tk.Label(
            param_inner, text=group["group"],
            font=("Segoe UI", 10, "bold"), fg=FG_GROUP, bg=BG_PARAM,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        for key, label, default, unit in group["params"]:
            row = tk.Frame(param_inner, bg=BG_PARAM)
            row.pack(fill="x", padx=12, pady=2)

            tk.Label(
                row, text=label,
                font=("Segoe UI", 9), fg=FG_DESC, bg=BG_PARAM,
                anchor="w",
            ).pack(fill="x")

            entry_row = tk.Frame(row, bg=BG_PARAM)
            entry_row.pack(fill="x")

            entry = tk.Entry(
                entry_row,
                font=("Consolas", 11), bg=BG_ENTRY, fg=FG_TITLE,
                insertbackground=FG_TITLE, relief="flat",
                highlightbackground=BORDER, highlightthickness=1,
            )
            entry.pack(side="left", fill="x", expand=True, ipady=3)

            # Pré-remplir avec la valeur existante ou le défaut
            current_val = existing_cfg.get(key, default)
            entry.insert(0, str(current_val))

            tk.Label(
                entry_row, text=f" {unit}",
                font=("Segoe UI", 9), fg="#585b70", bg=BG_PARAM,
            ).pack(side="left")

            param_entries[key] = entry

    # ================================================================
    # RIGHT: MODULES
    # ================================================================
    right_frame = tk.Frame(main_pane, bg=BG_MAIN)
    main_pane.add(right_frame, minsize=400)

    tk.Label(
        right_frame, text="📂  Modules disponibles",
        font=("Segoe UI", 12, "bold"), fg=FG_CAT, bg=BG_MAIN,
        anchor="w",
    ).pack(fill="x", padx=10, pady=(10, 4))

    tk.Frame(right_frame, bg=BORDER, height=1).pack(fill="x", padx=10, pady=4)

    # Scrollable modules
    mod_canvas = tk.Canvas(right_frame, bg=BG_MAIN, highlightthickness=0)
    mod_scroll = ttk.Scrollbar(right_frame, orient="vertical", command=mod_canvas.yview)
    mod_inner = tk.Frame(mod_canvas, bg=BG_MAIN)

    mod_inner.bind("<Configure>",
                   lambda e: mod_canvas.configure(scrollregion=mod_canvas.bbox("all")))
    mod_canvas.create_window((0, 0), window=mod_inner, anchor="nw")
    mod_canvas.configure(yscrollcommand=mod_scroll.set)
    mod_canvas.pack(side="left", fill="both", expand=True)
    mod_scroll.pack(side="right", fill="y")

    # Mousewheel sur les deux panneaux
    def _on_mousewheel(event):
        # Déterminer quel canvas scroller
        widget = event.widget
        # Remonter l'arbre pour trouver le canvas parent
        w = widget
        while w:
            if w == mod_canvas or w == mod_inner:
                mod_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return
            if w == param_canvas or w == param_inner:
                param_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return
            w = w.master
        mod_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    root.bind_all("<MouseWheel>", _on_mousewheel)

    # Cartes modules
    for mod in MODULES:
        card = tk.Frame(mod_inner, bg=BG_CARD, padx=12, pady=10,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", padx=6, pady=3)

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
                 font=("Segoe UI", 8), fg=FG_CAT, bg=BG_CARD).pack(side="left")
        tk.Label(top, text=f" {mod['name']}",
                 font=("Segoe UI", 11, "bold"), fg=FG_TITLE, bg=BG_CARD).pack(side="left")

        tk.Button(
            top, text="▶ Lancer", font=("Segoe UI", 9, "bold"),
            fg=BG_MAIN, bg=GREEN_RUN, relief="flat",
            padx=10, pady=2, cursor="hand2",
            command=lambda f=mod["file"]: on_select(f),
        ).pack(side="right")

        tk.Button(
            top, text=" ℹ ", font=("Segoe UI", 9, "bold"),
            fg=BG_MAIN, bg=INFO_BLUE, relief="flat",
            padx=4, pady=2, cursor="hand2",
            command=lambda m=mod: show_info_popup(root, m),
        ).pack(side="right", padx=(0, 4))

        tk.Label(card, text=mod["desc"],
                 font=("Segoe UI", 8), fg=FG_DESC, bg=BG_CARD,
                 justify="left", anchor="w", wraplength=450).pack(fill="x", pady=(4, 0))

    root.mainloop()
    return selected["file"]


def _reset_params(param_entries):
    """Remet tous les champs à leur valeur par défaut."""
    defaults = {}
    for group in PARAM_GROUPS:
        for key, label, default, unit in group["params"]:
            defaults[key] = default

    for key, entry in param_entries.items():
        entry.delete(0, tk.END)
        entry.insert(0, str(defaults.get(key, "")))


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    while True:
        chosen_file = show_launcher()

        if chosen_file is None:
            _delete_config()
            print("Aucun module sélectionné. Fin.")
            break

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

    # Nettoyage à la sortie
    _delete_config()
