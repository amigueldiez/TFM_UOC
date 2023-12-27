# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:51:09 2023

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

import reloxo

#%% Tuning de hiperparámetros

def afinadoRL(dataset, objetivo):
    tempo=reloxo.ElapsedTimer()
    print("AFINADO RL CATEGÓRICO - %s - %s" %(objetivo, tempo.current_time()))
    
    X=dataset.select_dtypes(include=['number'])
    X=X.drop("unix", axis=1)
    y=dataset[objetivo]
    #X=dataset.drop(["unix","m_id","m_subid", "alarms"], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    cLR= LogisticRegression(multi_class='ovr', max_iter=1000)
    parameters = {'C':[0.1,0.5,1,2,5]}

    gs = GridSearchCV(cLR, parameters)
    gs.fit(X_train, y_train)
    
    print("MEJOR SCORE - TRAIN: ", gs.best_score_)
    print("MEJORES PARÁMETROS - TRAIN: ", gs.best_params_)
    
    y_pred=gs.best_estimator_.predict(X_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train, y_pred), display_labels=gs.classes_)
    
    fig, ax = plt.subplots(figsize=(40,40))
    plt.title("Matriz de Confusión: RL - " + objetivo)
    plt.tight_layout()
    disp.plot(ax=ax, cmap="viridis")
    plt.xticks(rotation=90)
    plt.savefig("OUT/PLOTS/mConf_RL_" + objetivo+".png")
    plt.show()
    
    y_pred=gs.best_estimator_.predict(X_test)
    print("ACCURACY SCORE - TEST: ", accuracy_score(y_test, y_pred))
    print("MODELADO RL FINALIZADO - %s" % (tempo.elapsed_time()))
    joblib.dump(gs,"OUT/MODELS/Afinado_RL_"+objetivo+".joblib")
    return gs

def afinadoNB(dataset, objetivo):
    tempo=reloxo.ElapsedTimer()
    print("AFINADO NB CATEGÓRICO - %s - %s" %(objetivo, tempo.current_time()))
    
    X=dataset.select_dtypes(include=['number'])
    X=X.drop("unix", axis=1)
    y=dataset[objetivo]
    #X=dataset.drop(["unix","m_id","m_subid", "alarms"], axis=1)
    
    X=MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    cNB= MultinomialNB()
    parameters = {'alpha':[0.1,0.3,0.5,0.7,1]}

    gs = GridSearchCV(cNB, parameters)
    gs.fit(X_train, y_train)
    
    print("MEJOR SCORE: ", gs.best_score_)
    print("MEJORES PARÁMETROS: ", gs.best_params_)
    
    y_pred=gs.best_estimator_.predict(X_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train, y_pred), display_labels=gs.classes_)
    
    fig, ax = plt.subplots(figsize=(40,40))
    plt.title("Matriz de Confusión: NB - " + objetivo)
    plt.tight_layout()
    disp.plot(ax=ax, cmap="viridis")
    plt.xticks(rotation=90)
    plt.savefig("OUT/PLOTS/mConf_NB_" + objetivo+".png")
    plt.show()
    
    y_pred=gs.best_estimator_.predict(X_test)
    print("ACCURACY SCORE - TEST: ", accuracy_score(y_test, y_pred))
    print("MODELADO NB FINALIZADO - %s" % (tempo.elapsed_time()))
    joblib.dump(gs,"OUT/MODELS/Afinado_NB_"+objetivo+".joblib")
    return gs

def afinadoDT(dataset, objetivo):
    tempo=reloxo.ElapsedTimer()
    print("AFINADO DT CLASIFICADOR - %s - %s" %(objetivo, tempo.current_time()))
    
    X=dataset.select_dtypes(include=['number'])
    X=X.drop("unix", axis=1)
    y=dataset[objetivo]
    #X=dataset.drop(["unix","m_id","m_subid", "alarms"], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    cDT= DecisionTreeClassifier()
    parameters = {'max_depth':[2,5, 10,20], "min_samples_split":[2,5,10,20], "min_samples_leaf":[1,3,5,10,20],"max_features":[3,5,10,20]}

    gs = GridSearchCV(cDT, parameters)
    gs.fit(X_train, y_train)
    
    print("MEJOR SCORE: ", gs.best_score_)
    print("MEJORES PARÁMETROS: ", gs.best_params_)
    
    y_pred=gs.best_estimator_.predict(X_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train, y_pred), display_labels=gs.classes_)
    
    fig, ax = plt.subplots(figsize=(40,40))
    plt.title("Matriz de Confusión: DT - " + objetivo)
    plt.tight_layout()
    disp.plot(ax=ax, cmap="viridis")
    plt.xticks(rotation=90)
    plt.savefig("OUT/PLOTS/mConf_DT_" + objetivo+".png")
    plt.show()
    
    y_pred=gs.best_estimator_.predict(X_test)
    print("ACCURACY SCORE - TEST: ", accuracy_score(y_test, y_pred))
    print("MODELADO DT FINALIZADO - %s" % (tempo.elapsed_time()))
    joblib.dump(gs,"OUT/MODELS/Afinado_DT_"+objetivo+".joblib")
    return gs

