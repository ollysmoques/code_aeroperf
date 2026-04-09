# -*- coding: utf-8 -*-
"""
Centralized Mission Parameters
"""
from config_loader import get as cfg

# Mission Height in feet
MISSION_HEIGHT_FT = cfg("MISSION_HEIGHT_FT", 2000)
