# -*- coding: utf-8 -*-
"""
@author: pmartinez
"""

import os

def sistemaCarpetas():
    
    ruta=os.getcwd().replace("\\", "/")
    
    if not os.path.exists(ruta+"/IN/"):
        os.mkdir(ruta+"/IN/")
        print("CREADA CARPETA IN")
        
    if not os.path.exists(ruta+"/OUT/"):
        os.mkdir(ruta+"/OUT/")
        print("CREADA CARPETA OUT")
        
    if not os.path.exists(ruta+"/OUT/PLOTS/"):
        os.mkdir(ruta+"/OUT/PLOTS/")
        print("CREADA CARPETA OUT/PLOTS/")
        
    if not os.path.exists(ruta+"/OUT/MODELS/"):
        os.mkdir(ruta+"/OUT/MODELS/")
        print("CREADA CARPETA OUT/MODELS/")

    if not os.path.exists(ruta+"/PROCESS/"):
        os.mkdir(ruta+"/PROCESS/")
        print("CREADA CARPETA PROCESS")
    
