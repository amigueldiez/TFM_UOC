# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:32:13 2023

@author: pablo
"""
import setupInicial
import reloxo
import githubInteraccion
import transformarEDA
import pcaLDA
import afinadoHiperparametros
import streamRiver

import os
import pandas as pd

github_url = 'https://github.com/ari-dasci/OD-TINA'
github_token = ''

#%%

if __name__ == '__main__':
    
    setupInicial.sistemaCarpetas()
    
    print("INICIO LISTADO ARCHIVOS... (0 s)")
    tempo=reloxo.ElapsedTimer()
    files_in_repo = githubInteraccion.listar_files(github_url, token=github_token)
    print("FIN LISTADO ARCHIVOS... (%s)" % tempo.elapsed_time())
    
    #leer_files(files_in_repo)
    
    files_in_repo=[i for i in files_in_repo if i.endswith(".parquet")]
    #a veces hay problemas y hay que reiniciar el kernel, guardar el listado
    pd.DataFrame(files_in_repo).to_csv("PROCESS/ListaArchivos.txt", index=False, header=False)
    
#%% DESCARGAR    
    if 'files_in_repo' not in locals() or None:
        files_in_repo=pd.read_csv("PROCESS/ListaArchivos.txt", header= None)
        files_in_repo=files_in_repo[0].tolist()
        
    files_in_local=os.listdir("IN/")    
    files_in_local=[i.split(".")[0] for i in files_in_local if i.endswith(".csv")]
    
    print("INICIO DESCARGA ARCHIVOS...")
    #revisar límite archivos
    #
    #
    for file in files_in_repo[0:70]:
        #Checkear si ya está descargado. Por si hay algún problema en la descarga
        if file.split("/")[-2].split("=")[-1] not in files_in_local:
            githubInteraccion.descargar_file(github_url, filepath=file, token=github_token)
        else:
            print("ARCHIVO (%s) YA DESCARGADO." % file.split("/")[-2].split("=")[-1])
    
    print("FIN DESCARGA ARCHIVOS... ")
#%% Agregar

    print("INICIO AGREGACIÓN ARCHIVOS...")
    files_in_local=os.listdir("IN/")    
    files_in_local=[i for i in files_in_local if i.endswith(".csv")]
    if "datos_agregados.csv" in files_in_local:
        files_in_local.remove("datos_agregados.csv")
    
    transformarEDA.agregarDatosTandas(files_in_local, chunk_size=300)
    print("FIN AGREGACIÓN ARCHIVOS...")

#%% EDA con archivos locales offline
    #EDA
    tempo.reset()
    print("EDA - (%s) " %(tempo.current_time()))
    dataset=pd.read_csv('PROCESS/datos_agregados.csv')
    datosCuantit, datosCualit=transformarEDA.EDA(dataset)
    #print(datosCuantit)
    #print(datosCualit)
    #numero niveles variables cuantitativas
    transformarEDA.analisisCategorico(datosCualit)
    #cruce variables cualitativas
    transformarEDA.cruceCategorico(dataset)

#%% Reducción dimensionalidad
    print("PCA - (%s) " %(tempo.current_time()))
    #PCA
    pca_dataset=pcaLDA.pca2D(dataset,5)
    
    print("LDA- (%s) " %(tempo.current_time()))
    #LDA
    lda_dataset=pcaLDA.lda2D(dataset,5)
    
    print("FIN EDA, PCA y LDA - (%s) " %(tempo.elapsed_time()))
    
#%% Tuning de hiperparámetros
    
    tempo.reset()
    #objetivos=["m_id","alarms","m_subid"]
    objetivos=["m_id"]
    
    for objetivo in objetivos:

        #regresión logística
        htRL=afinadoHiperparametros.afinadoRL(dataset, objetivo)
    
        #regresión NB
        htNB=afinadoHiperparametros.afinadoNB(dataset, objetivo)
    
        #regresión DT
        htDT=afinadoHiperparametros.afinadoDT(dataset, objetivo)
    
    print("FIN AFINADO - (%s) " %(tempo.elapsed_time()))
    
    #recuperables usando joblib.load()


#%% Flujo en Stream

    # descarga síncrona con la lectura usando threads? emplear directamente el archivo agregado?
    #seleccion="individual"
    seleccion="agrupado"

    flujo, w_size=streamRiver.modoTrabajo(seleccion)
    #objetivos=["m_id","alarms","m_subid"]
    objetivos=["m_id"]
    for objetivo in objetivos:
        modeloRL, matrizConfusionRL=streamRiver.modeloLogistico(flujo, objetivo)
        modeloNB, matrizConfusionNB=streamRiver.modeloProbabilistico(flujo, objetivo)
        modeloDT, matrizConfusionDT=streamRiver.modeloArbol(flujo, objetivo)
    #si detectaamos uno, no hacemos nada, si detectamos varios seguidos
    #definimos ventana. umbral de ventana
    #anomalía de las entradas, anomalía de las salidas




