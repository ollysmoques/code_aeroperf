# -*- coding: utf-8 -*-
"""
Chargeur de configuration du launcher.
Si _launcher_config.json existe, ses valeurs surchargent les défauts.
Sinon, retourne les valeurs par défaut du code.
"""
import os
import json

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_launcher_config.json")

def load_config():
    """Charge le fichier de config du launcher s'il existe. Retourne un dict (vide si pas de fichier)."""
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get(key, default):
    """Retourne la valeur du config si elle existe, sinon le défaut."""
    cfg = load_config()
    return cfg.get(key, default)
