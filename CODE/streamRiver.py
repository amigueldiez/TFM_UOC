# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:01:11 2023

@author: pablo
"""

#%%
import pandas as pd
import os
from river import stream, preprocessing, compose, linear_model, multiclass, metrics, naive_bayes, tree
import itertools

#%%

def modoTrabajo(modo):
    if modo=="individual":
        w_size=600
        print("Trabajando con datos individuales.")
        files_in_local=os.listdir()    
        files_in_local=[i for i in files_in_local if i.endswith(".csv")]
        if "datos_agregados.csv" in files_in_local:
            files_in_local.remove("datos_agregados.csv")
        return files_in_local, w_size
    elif modo=="agrupado":
        w_size=120
        print("Trabajando con datos agregados.")
        return ['datos_agregados.csv'], w_size
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

def modeloLogistico(files, objetivo, admite_cualitativo=False):
    
    print(f"MODELO LOGÍSTICO - {objetivo.upper()}")

    model = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(linear_model.LogisticRegression())

    acc = metrics.Accuracy()
    conf_matrix=metrics.ConfusionMatrix()
    muestras_analizadas=0

    for file in files:
        # Simular iteración
        dataset = stream.iter_csv(file)
    
        for count,data_point in enumerate(dataset):

            features = extract_features(data_point, admite_cualitativo)
    
            target = get_target(data_point, objetivo)
            
            #target = 'OK' if target == 'none' else target
                
            y_pred = model.predict_one(features)
    
            model.learn_one(features, target)
            
            if count >0:
                conf_matrix.update(target, y_pred)
                acc.update(target, y_pred)
            if count % 10000==0:
                print(f'Accuracy para {file} - Objetivo: {objetivo.upper()}: {acc.get()} - Muestras analizadas: {muestras_analizadas + count}')
        muestras_analizadas=muestras_analizadas + count
                
        print(f'Accuracy final para {file} - Objetivo: {objetivo.upper()}: {acc.get()}')
                
    print(f'Accuracy final modelo - Objetivo: {objetivo.upper()}: {acc.get()}')
    
    print(f'Verdaderos positivos (TP): {conf_matrix.total_true_positives*100/conf_matrix.n_samples} %')
    print(f'Verdaderos negativos (TN): {conf_matrix.total_true_negatives*100/conf_matrix.n_samples} %')
    print(f'Falsos positivos (FP): {conf_matrix.total_false_positives*100/conf_matrix.n_samples} %')
    print(f'Falsos negativos (FN): {conf_matrix.total_false_negatives*100/conf_matrix.n_samples} %')

    return model, conf_matrix

def modeloProbabilistico(files, objetivo, admite_cualitativo=False):
    
    print(f"MODELO PROBABILÍSTICO - {objetivo.upper()}")
    
    scaler = preprocessing.MinMaxScaler()

    model = naive_bayes.MultinomialNB()

    acc = metrics.Accuracy()
    conf_matrix=metrics.ConfusionMatrix()
    muestras_analizadas=0

    for file in files:
        # Simular iteración
        dataset = stream.iter_csv(file)
    
        for count,data_point in enumerate(dataset):

            features = extract_features(data_point, admite_cualitativo)
            
            features = scaler.learn_one(features).transform_one(features)
    
            target = get_target(data_point, objetivo)
            
            #target = 'OK' if target == 'none' else target
                
            model.learn_one(features, target)
            
            if count >0:
                y_pred = model.predict_one(features)
                conf_matrix.update(target, y_pred)
                acc.update(target, y_pred)
            if count % 10000==0:
                print(f'Accuracy para {file} - Objetivo: {objetivo.upper()}: {acc.get()} - Muestras analizadas: {muestras_analizadas + count}')
        muestras_analizadas=muestras_analizadas + count
                
        print(f'Accuracy final para {file} - Objetivo: {objetivo.upper()}: {acc.get()}')
                
    print(f'Accuracy final modelo - Objetivo: {objetivo.upper()}: {acc.get()}')
    
    print(f'Verdaderos positivos (TP): {conf_matrix.total_true_positives*100/conf_matrix.n_samples} %')
    print(f'Verdaderos negativos (TN): {conf_matrix.total_true_negatives*100/conf_matrix.n_samples} %')
    print(f'Falsos positivos (FP): {conf_matrix.total_false_positives*100/conf_matrix.n_samples} %')
    print(f'Falsos negativos (FN): {conf_matrix.total_false_negatives*100/conf_matrix.n_samples} %')

    return model, conf_matrix

def modeloArbol(files, objetivo, admite_cualitativo=True):
    
    print(f"MODELO DT - {objetivo.upper()}")
    #model = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()
    model = tree.HoeffdingTreeClassifier()

    acc = metrics.Accuracy()
    conf_matrix=metrics.ConfusionMatrix()
    muestras_analizadas=0

    for file in files:
        # Simular iteración
        dataset = stream.iter_csv(file)
    
        for count,data_point in enumerate(dataset):

            features = extract_features(data_point, admite_cualitativo)
    
            target = get_target(data_point, objetivo)
                            
            y_pred = model.predict_one(features)
    
            model.learn_one(features, target)
            
            if count >0:
                conf_matrix.update(target, y_pred)
                acc.update(target, y_pred)
            if count % 10000==0:
                print(f'Accuracy para {file} - Objetivo: {objetivo.upper()}: {acc.get()} - Muestras analizadas: {muestras_analizadas + count}')
        muestras_analizadas=muestras_analizadas + count
                
        print(f'Accuracy final para {file} - Objetivo: {objetivo.upper()}: {acc.get()}')
                
    print(f'Accuracy final modelo - Objetivo: {objetivo.upper()}: {acc.get()}')
    
    print(f'Verdaderos positivos (TP): {conf_matrix.total_true_positives*100/conf_matrix.n_samples} %')
    print(f'Verdaderos negativos (TN): {conf_matrix.total_true_negatives*100/conf_matrix.n_samples} %')
    print(f'Falsos positivos (FP): {conf_matrix.total_false_positives*100/conf_matrix.n_samples} %')
    print(f'Falsos negativos (FN): {conf_matrix.total_false_negatives*100/conf_matrix.n_samples} %')

    return model, conf_matrix



