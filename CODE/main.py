# -*- coding: utf-8 -*-
"""
@author: pablo
"""

"""
A veces hay alguna imcompatibilidad entre versiones, si pasa se recomienda:
pip install --upgrade numpy pandas seaborn matplotlib --user
Posteriormente, pasó lo mismo con KSWIN y scipy
pip install --upgrade scipy
"""

import setupInicial
import githubInteraccion
import transformarEDA
import pcaLDA
import afinadoHiperparametros
import streamPropio


import os
import pandas as pd

github_url = 'https://github.com/ari-dasci/OD-TINA'
github_token = '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'

#%%

if __name__ == '__main__':
    
    setupInicial.sistemaCarpetas()
    
    print("INICIO LISTADO ARCHIVOS...")
    files_in_repo = githubInteraccion.listar_files(github_url, token=github_token)
    print("FIN LISTADO ARCHIVOS")
    
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
    for file in files_in_repo:
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
    print("EDA - Análisis exploratorio")
    dataset=pd.read_csv('PROCESS/datos_agregados.csv')
    datosCuantit, datosCualit=transformarEDA.EDA(dataset)
    #print(datosCuantit)
    #print(datosCualit)
    #numero niveles variables cuantitativas
    transformarEDA.analisisCategorico(datosCualit)
    #cruce variables cualitativas
    transformarEDA.cruceCategorico(dataset)

#%% Reducción dimensionalidad
    print("PCA")
    #PCA
    pca_dataset=pcaLDA.pca2D(dataset,40)
    
    print("LDA")
    #LDA
    lda_dataset=pcaLDA.lda2D(dataset,5)
    
    print("FIN EDA, PCA y LDA")
    
#%% Tuning de hiperparámetros
    
    dataset=pd.read_csv('PROCESS/datos_agregados.csv')

    objetivos=["m_id","alarms","m_subid"]
    algoritmos=["RL","NB","DT"]
    
    modelosAH={}
    for objetivo in objetivos:
        for algoritmo in algoritmos:
            modelosAH[algoritmo+"_"+objetivo+"_"]=afinadoHiperparametros.afinado(dataset, objetivo, algoritmo)
    
    print("FIN AFINADO")
    
    #modelos recuperables usando joblib.load()


#%% Flujo en Stream

    #archivos a emplear
    
    #seleccionDatos="individual"
    seleccionDatos="agrupado"
    
    #configuracion streaming
    _fin_train =0.5 # apartir del 70 % deja de entrenar, empieza "test"
    _it_drift=[0.25,0.8]
    #_it_drift=0.7 # a partir del 80% arranca drift artificiañ
    _dur_drift=0.1
    #Ìdur drift se puede dar en porcentual
    _n_var_driftAL=100
    _noise_driftAL=0.9
    _n_var_driftCR=100
    _algoritmoAnomalia="HST" #["HST","LOF","OCSVM", None]
    _umbral_deteccion=0.75
    _algoritmo_drift="ADWIN" #[None,"ADWIN","KSWIN","PH"]
    #_wsize=3000
    _consigna_drift="nuevo" #["reentrenar","nuevo", None]
    
    #algoritmos=["RL","NB","DT"]
    #algoritmos=["NB","DT"]
    algoritmos=["RL"]
    #objetivos=["m_id","alarms","m_subid"]
    objetivos=["m_id"]
    drifts_artificiales=[None,"ALEATORIO","CRUCE"]
    #drifts_artificiales=[None]

    modelosST={}
    for drift_generado in drifts_artificiales:
        for objetivo in objetivos:
            for algoritmo in algoritmos:
                #creacion
                st=streamPropio.streamModel(modoDatos=seleccionDatos, aplicar_drift=drift_generado,
                                            algoritmo=algoritmo, objetivo=objetivo)
                #config
                st.parametrosConfig(_fin_train, _it_drift, _dur_drift,_n_var_driftAL,
                         _noise_driftAL, _n_var_driftCR, _algoritmoAnomalia, 
                         _umbral_deteccion, _algoritmo_drift, consigna_drift=_consigna_drift)
                #ejecucion
                modelosST[algoritmo+"_"+objetivo+"_"+str(drift_generado)]=st.ejecucion()
