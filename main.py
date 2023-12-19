# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:32:13 2023

@author: pablo
"""
import Reloxo
import Github_ListarDescargar
import LecturaEDA_PCA_TUNNING
import StreamRiver

import os
import pandas as pd

github_url = 'https://github.com/ari-dasci/OD-TINA'
github_token = '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'

#%%

if __name__ == '__main__':
    
    print("INICIO LISTADO ARCHIVOS... (0 s)")
    tempo=Reloxo.ElapsedTimer()
    files_in_repo = Github_ListarDescargar.listar_files(github_url, token=github_token)
    print("FIN LISTADO ARCHIVOS... (%s)" % tempo.elapsed_time())
    
    #leer_files(files_in_repo)
    
    files_in_repo=[i for i in files_in_repo if i.endswith(".parquet")]
    #a veces hay problemas y hay que reiniciar el kernel, guardar el listado
    pd.DataFrame(files_in_repo).to_csv("ListaArchivos.txt", index=False, header=False)
    
#%% DESCARGAR    
    if 'files_in_repo' not in locals():
        files_in_repo=pd.read_csv("ListaArchivos.txt", header= None)
        files_in_repo=files_in_repo[0].tolist()
        
    files_in_local=os.listdir()    
    files_in_local=[i.split(".")[0] for i in files_in_local if i.endswith(".csv")]
    
    print("INICIO DESCARGA ARCHIVOS...")
    #revisar límite archivos
    #
    #
    for file in files_in_repo:
        #Checkear si ya está descargado. Por si hay algún problema en la descarga
        if file.split("/")[-2].split("=")[-1] not in files_in_local:
            Github_ListarDescargar.descargar_file(github_url, filepath=file, token=github_token)
    
    print("FIN DESCARGA ARCHIVOS... ")
#%% EDA con archivos locales offline

    print("INICIO AGREGACIÓN ARCHIVOS...")
    files_in_local=os.listdir()    
    files_in_local=[i for i in files_in_local if i.endswith(".csv")]
    if "datos_agregados.csv" in files_in_local:
        files_in_local.remove("datos_agregados.csv")
    
    LecturaEDA_PCA_TUNNING.agregarDatosTandas(files_in_local, chunk_size=300)
    print("FIN AGREGACIÓN ARCHIVOS...")
    
    #EDA
    dataset=pd.read_csv('datos_agregados.csv')
    #datosCuantit, datosCualit=LecturaEDA_PCA_TUNNING.EDA(dataset)
    #print(datosCuantit)
    #print(datosCualit)
    
    #PCA
    LecturaEDA_PCA_TUNNING.pca2D(dataset,5)
    #LDA
    LecturaEDA_PCA_TUNNING.lda2D(dataset,5)
    

#%% Tuning de hiperparámetros
    



#%% Flujo en Stream


    # descarga síncrona con la lectura usando threads? emplear directamente el archivo agregado?
    #seleccion="individual"
    seleccion="agrupado"
    flujo=StreamRiver.modoTrabajo(seleccion)
    objetivo="alarms"
    modeloRL=StreamRiver.modeloLogistico(flujo, objetivo)
    




