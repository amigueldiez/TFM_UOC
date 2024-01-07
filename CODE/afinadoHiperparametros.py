# -*- coding: utf-8 -*-
"""
@author: pablo
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import joblib
import seaborn as sns

import reloxo

#%% Tuning de hiperparámetros

def seleccionModelo (algoritmo):
    if algoritmo=="RL":
        model= LogisticRegression(multi_class='ovr', max_iter=10000)
        scaler=None
        parameters = {'C':[0.1,0.5,1,2,5,10,12,15]}
        return model, parameters, scaler
    elif algoritmo=="NB":
        model= MultinomialNB()
        scaler=MinMaxScaler()
        parameters = {'alpha':[0.05,0.1,0.3,0.5,0.7,1]}
        return model, parameters, scaler
    elif algoritmo=="DT":
        model= DecisionTreeClassifier()
        scaler=None
        parameters = {'max_depth':[2,5, 10,20,35], "min_samples_split":[2,5,10,20], "min_samples_leaf":[1,3,5,10],"max_features":[3,5,10,20,35]}
        return model, parameters, scaler
    else:
        print("INTRODUZCA UN MODELO CORRECTO.")
        return None, None, None
    

def afinado(dataset, objetivo, algoritmo):
    tempo=reloxo.ElapsedTimer()
    
    modelo, parametros, scaler = seleccionModelo(algoritmo)
    
    if modelo is not None:
        print("AFINADO %s - %s - %s" %(algoritmo, objetivo, tempo.current_time()))
        
        X=dataset.select_dtypes(include=['number'])
        X=X.drop("unix", axis=1)
        y=dataset[objetivo]
        #X=dataset.drop(["unix","m_id","m_subid", "alarms"], axis=1)
        
        if scaler is not None:
            X=scaler.fit_transform(X)        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        gs = GridSearchCV(modelo, parametros)
        gs.fit(X_train, y_train)
        
        print("MEJOR SCORE - TRAIN: ", round(gs.best_score_,2))
        print("MEJORES PARÁMETROS - TRAIN: ", gs.best_params_)
        
        y_pred=gs.best_estimator_.predict(X_train)
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train, y_pred), display_labels=gs.classes_)
        
        fig, ax = plt.subplots(figsize=(40,40))
        plt.title("Matriz de Confusión: "+algoritmo+" - " + objetivo)
        plt.tight_layout()
        sns.heatmap(confusion_matrix(y_train, y_pred), cmap="crest")
        plt.xticks(rotation=90)
        plt.savefig("OUT/PLOTS/mConf_"+algoritmo+"_" + objetivo+".png")
        plt.show()
        
        plt.figure(figsize=(25, 25))
        plt.tight_layout()
        
        y_pred=gs.best_estimator_.predict(X_test)
        print("ACCURACY SCORE - TEST: ", round(accuracy_score(y_test, y_pred),2))
        print("MODELADO %s FINALIZADO - %s" % (modelo, tempo.elapsed_time()))
        joblib.dump(gs,"OUT/MODELS/afinado_"+algoritmo+"_"+objetivo+".joblib")
        return gs

