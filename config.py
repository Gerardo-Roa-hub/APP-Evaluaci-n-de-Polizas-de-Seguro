# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:26:08 2025

@author: Dell
"""

import os

# Configuración de la aplicación
class Config:
    APP_NAME = "Proyecto App Streamlit"
    DEBUG = os.getenv("DEBUG", False)
    SECRET_KEY = os.getenv("SECRET_KEY", "tu_clave_secreta")