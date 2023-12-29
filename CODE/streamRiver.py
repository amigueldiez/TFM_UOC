# -*- coding: utf-8 -*-
"""
@author: pablo
"""

#%%
import pandas as pd
import os
from river import stream, preprocessing, compose, linear_model, multiclass, metrics, naive_bayes, tree
import itertools
import plotPerformance
import joblib

import reloxo
import driftGeneration


#%%

def modoTrabajo(modo):
    if modo=="individual":
        w_size=600
        print("Trabajando con datos individuales.")
        files_in_local=os.listdir("IN/")    
        files_in_local=["IN/"+i for i in files_in_local if i.endswith(".csv")]
        iterPrint=25000
        dur_drift=300000
        return files_in_local, w_size, iterPrint, dur_drift
    elif modo=="agrupado":
        w_size=120
        print("Trabajando con datos agregados.")
        iterPrint=2000
        dur_drift=5000
        return ["PROCESS/datos_agregados.csv"], w_size, iterPrint, dur_drift
    else:
        print("Selecciona una fuente de datos correcta.")
        return None

def valorNumerico(valor):
    try:
        return float(valor)
    except:
        return valor

def extract_features(data, admite_cualitativo):
    X=dict(itertools.islice(data[0].items(), len(data[0])-3))
    if not admite_cualitativo:
        X.pop("FEATURE76")
        X.pop("FEATURE87")
    X.pop("unix")
    X.update((k, valorNumerico(v)) for k, v in X.items())
    return X

def get_target(data, target):
    #y=[data[0]["m_id"],data[0]["m_subid"],data[0]["alarms"]]
    #y=dict(itertools.islice(data[0].items(), len(data[0])-1, len(data[0])))
    y=dict(itertools.islice(data[0].items(), len(data[0])-3, len(data[0])))
    y=y[target]
    return y
    #return list(y.values())[0]
    
def selectorModelo(modelo, objetivo, tipo_drift, duracion_drift):
    if tipo_drift=="ALEATORIO":
        var=100
    elif tipo_drift=="CRUCE":
        var=100
    else:
        var=0
    if modelo=="RL":
        admite_cualitativo=False
        scaler=None
        model = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
        drift=aplicarDrift(tipo_drift, var, duracion_drift)
        return scaler, model, admite_cualitativo, drift
    elif modelo=="NB":
        admite_cualitativo=False
        scaler = preprocessing.MinMaxScaler()
        model = naive_bayes.MultinomialNB()
        drift=aplicarDrift(tipo_drift, var, duracion_drift)
        return scaler, model, admite_cualitativo, drift
    elif modelo=="DT":
        admite_cualitativo=False
        #admite_cualitativo=True
        scaler=None
        model = tree.HoeffdingTreeClassifier()
        drift=aplicarDrift(tipo_drift, var, duracion_drift)
        return scaler, model, admite_cualitativo, drift
    else:
        print("INTRODUZCA UN MODELO CORRECTO.")
        return None
        
    #seleccion de métricas según objetivo cuantitativo o cualtitativo (y binario o multilevel)?
    #acc = metrics.Accuracy()
    #conf_matrix=metrics.ConfusionMatrix()
    
#def aplicarDrift(tipo, num_var, duracion):
def aplicarDrift(tipo, num_var, duracion):
    if tipo==None:
        print("NO SE GENERA DRIFT")
        return None
    elif tipo=="ALEATORIO":
        print("GENERA DRIFT ALEATORIO")
        ruido=driftGeneration.randomDrift(num_var,0.9, duracion)
        return ruido
    elif tipo=="CRUCE":
        print("GENERA DRIFT CRUZADO")
        ruido=driftGeneration.crossDrift(num_var, duracion)
        return ruido
    else:
        print("INTRODUZCA UN MODELO DE DRIFT CORRECTO.")
        return None

def streaming(files, algoritmo, objetivo, drift_artificial=None, dur_drift=0, it=10000):
    
    #llamar selector
    mod=selectorModelo(algoritmo, objetivo, drift_artificial, dur_drift)
    if mod is not None:
        tempo=reloxo.ElapsedTimer()
        print(f"MODELO {algoritmo} - {objetivo.upper()} - {tempo.current_time()}")
        
        #metricas
        acc = metrics.Accuracy()
        conf_matrix=metrics.ConfusionMatrix()
        
        #modelo
        model=mod[1]
        
        #escalado        
        scaler=mod[0]

        
        #drift articial
        drift=mod[3]    
    
        
        #grafica dinamica    
        grafica=plotPerformance.plotMetrica(f"MODELO {algoritmo} - "+objetivo.upper(), "accuracy")
    
        muestras=0
    
        for file in files:
            # Simular iteración
            dataset = stream.iter_csv(file)
        
            for data_point in dataset:
                
                features = extract_features(data_point, mod[2])
                
                if drift is not None:
                    if drift.activated:
                
                        features=drift.addNoise(features)
                
                if scaler is not None:
                
                    scaler.learn_one(features)
                
                    features = scaler.transform_one(features)
        
                target = get_target(data_point, objetivo)
                                
                model.learn_one(features, target)
                
                if muestras >0:
                    y_pred = model.predict_one(features)
                    conf_matrix.update(target, y_pred)
                    acc.update(target, y_pred)
                if muestras % it==0:
                    print(f'MODELO {algoritmo} - Accuracy para {file} - Objetivo: {objetivo.upper()}: \n \t {round(acc.get(),2)} - Muestras analizadas: {muestras}')
                if ((muestras >0) and (muestras % it==0)):
                    grafica.processDataPoint(muestras, acc.get())
                muestras=muestras + 1
                if ((muestras==12000) and (drift is not None)):
                    drift.disparador()
            
            grafica.processDataPoint(muestras, acc.get())
            grafica.printFile(muestras, file.split("/")[-1])
            
                    
            print(f'MODELO {algoritmo} - Modelo {algoritmo} - Accuracy final para {file} - Objetivo: {objetivo.upper()}: \n \t {round(acc.get(),2)}')
                    
        print(f'Accuracy final MODELO {algoritmo} - Objetivo: {objetivo.upper()}: \n \t {round(acc.get(),2)} - {tempo.elapsed_time()}')
        
        if drift is None:
            grafica.savePlot("streaming_"+algoritmo+"_"+objetivo, round(acc.get(),2), len(files), tempo.elapsed_time(), "No")
            joblib.dump(model,"OUT/MODELS/streaming_"+algoritmo+"_"+objetivo+".joblib")
        else:
            grafica.savePlot("streaming_"+algoritmo+"_"+objetivo+"_dr"+drift_artificial, round(acc.get(),2), len(files), tempo.elapsed_time(),drift_artificial)
            joblib.dump(model,"OUT/MODELS/streaming_"+algoritmo+"_"+objetivo+"_dr"+drift_artificial+".joblib")
        
        print(f'Verdaderos positivos (TP): {conf_matrix.total_true_positives*100/conf_matrix.n_samples} %')
        print(f'Verdaderos negativos (TN): {conf_matrix.total_true_negatives*100/conf_matrix.n_samples} %')
        print(f'Falsos positivos (FP): {conf_matrix.total_false_positives*100/conf_matrix.n_samples} %')
        print(f'Falsos negativos (FN): {conf_matrix.total_false_negatives*100/conf_matrix.n_samples} %')
    
        return model, conf_matrix
    
    