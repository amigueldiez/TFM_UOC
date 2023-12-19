# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:01:11 2023

@author: pablo
"""

#%%
import pandas as pd
import os
from river import stream, preprocessing, compose, linear_model, multiclass, metrics
import itertools

def modoTrabajo(modo):
    if modo=="individual":
        print("Trabajando con datos individuales.")
        files_in_local=os.listdir()    
        files_in_local=[i for i in files_in_local if i.endswith(".csv")]
        if "datos_agregados.csv" in files_in_local:
            files_in_local.remove("datos_agregados.csv")
        return files_in_local
    elif modo=="agrupado":
        print("Trabajando con datos agregados.")
        return ['datos_agregados.csv']
    else:
        print("Selecciona una fuente de datos correcta.")
        return None

def extract_features(data):
    X=dict(itertools.islice(data[0].items(), len(data[0])-3))
    X.pop("FEATURE76")
    X.pop("FEATURE87")
    X.pop("unix")
    X.update((k, float(v)) for k, v in X.items())
    return X

def get_target(data):
    #y=[data[0]["m_id"],data[0]["m_subid"],data[0]["alarms"]]
    #y=dict(itertools.islice(data[0].items(), len(data[0])-3, len(data[0])))
    y=dict(itertools.islice(data[0].items(), len(data[0])-1, len(data[0])))
    #return y
    return list(y.values())[0]

def modeloLogistico(files, objetivo):

    model = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(linear_model.LogisticRegression())

    acc = metrics.Accuracy()
    conf_matrix=metrics.ConfusionMatrix()


    for file in files:
        # Simular iteración
        dataset = stream.iter_csv(file)
    
        for count,data_point in enumerate(dataset):
            # Extract features from the data
            features = extract_features(data_point)
    
            target = get_target(data_point)
                
            y_pred = model.predict_one(features)
    
            model.learn_one(features, target)
    
            acc.update(target, y_pred)
            conf_matrix.update(target, y_pred)
                
            if count % 10000==0:
                print(f'Accuracy para {file}: {acc.get()} - Muestras analizadas: {count}')
    
        #print(f'Accuracy final para {file}: {acc.get()}')
        
        #print(f'Matriz de confusión final para {file}: {conf_matrix}')
        
    #print(f'Accuracy final modelo: {acc.get()}')
    
    #print(f'Matriz de confusión final modelo: {conf_matrix}')
        
    return model

