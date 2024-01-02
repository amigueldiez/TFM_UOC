# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:40:47 2023

@author: pablo
"""
import os
from river import stream, preprocessing, compose, linear_model, multiclass, metrics, naive_bayes, tree
import itertools
import plotPerformance
import joblib

import reloxo
import driftGeneration
import anomaliaPropio
import driftPropio

class streamModel:
    def __init__(self, modoDatos, algoritmo, objetivo, aplicar_drift):
        
        self.modoDatos=modoDatos
        self.algoritmo = algoritmo
        self.objetivo = objetivo
        self.aplicar_drift=aplicar_drift
        
        self.files=None
        #indexa archivos
        self.modoTrabajo(self.modoDatos)
        
        #calcula numero datos
        self.num_datos=self.numDatos(self.files)
        print("DATOS: ", self.num_datos)
        self.n_modelo=0
        self.modelos={}
        
        self.admite_cualitativo=False
        
        #self.modalidad=modalidad #◘cuantitativo o cualitativo
        #self.tipo_reentrenamiento=tipo_reentrenamiento
        
        self.iterPrint=0
        self.fin_train=0
        self.it_anomalia=0
        self.it_drift=0
        self.dur_drift=0
        self.n_var_driftCR=0
        self.noise_driftAL=0
        self.n_var_driftCR=0
        self.algoritmoAnomalia=None
        self.umbral_deteccion=0
        self.algoritmo_drift=None
        self.w_size=0
        self.consigna_drift=None
    
    def parametrosConfig(self, fin_train, it_anomalia=30000, it_drift=80000, dur_drift=10000,n_var_driftAL=90,
                         noise_driftAL=0.8, n_var_driftCR=50, algoritmoAnomalia=None,  
                         umbral_deteccion=None, algoritmo_drift=None, wsize_adwin=3000, consigna_drift=None):
        if self.num_datos <200_000:
            self.iterPrint=int(self.num_datos*0.02)
        elif self.num_datos<500_000:
            self.iterPrint=int(self.num_datos*0.05)
        elif self.num_datos<1_000_000:
            self.iterPrint=int(self.num_datos*0.1) 
        else:
            self.iterPrint=int(self.num_datos*0.15) 
        
        self.fin_train=int(fin_train*self.num_datos)
        #anomalias
        self.it_anomalia=int(it_anomalia*self.num_datos) #inicio deteccion anomalia
        #generacion drift
        self.it_drift=int(it_drift*self.num_datos)+1 #inicio drift
        self.dur_drift=int(dur_drift*self.num_datos) #duracion drift
        self.n_var_driftCR=n_var_driftCR
        self.noise_driftAL=noise_driftAL
        self.n_var_driftCR=n_var_driftCR
        #deteccion anomalias
        self.algoritmoAnomalia=algoritmoAnomalia
        self.umbral_deteccion=umbral_deteccion
        # deteccion drift
        self.algoritmo_drift=algoritmo_drift
        self.w_size=wsize_adwin
        #actuacion_drift
        self.consigna_drift=consigna_drift
        
    def modoTrabajo(self, modo):
        if modo=="individual":
            print("Trabajando con datos individuales.")
            files_in_local=os.listdir("IN/")    
            files_in_local=["IN/"+i for i in files_in_local if i.endswith(".csv")]
            #self.files=files_in_local
            self.files=files_in_local[0:100]
        elif modo=="agrupado":
            self.files=["PROCESS/datos_agregados.csv"]
        else:
            print("Selecciona una fuente de datos correcta.")
            self.files=None
            
    def numDatos(self, files):
        datos=0
        for _ in files:
            with open(_, 'r') as file:
                n = sum(1 for line in file)
            datos=datos+n
        return datos
    
    def selectorDrift(self, tipo):
        if tipo==None:
            print("NO SE GENERA DRIFT")
            return None
        elif tipo=="ALEATORIO":
            print("GENERA DRIFT ALEATORIO")
            return driftGeneration.randomDrift(self.n_var_driftCR, self.noise_driftAL, self.dur_drift)
        elif tipo=="CRUCE":
            print("GENERA DRIFT CRUZADO")
            return driftGeneration.crossDrift(self.n_var_driftCR, self.dur_drift)
        else:
            print("INTRODUZCA UN MODELO DE DRIFT CORRECTO. NO SE GENERARÁ DRIFT.")
            return None
        
    def leerHiperparametros(self, algoritmo, objetivo):
        try:
            af_mod=joblib.load("OUT/MODELS/afinado_"+algoritmo+"_"+objetivo+".joblib")
        except:
            print("No existe un modelo de afinado para el presente modelo.")
            return None
        else:
            print("Parámetros de afinado cargados.")
            return af_mod.best_params_
            
        
    def selectorModelo(self, modelo, objetivo):
        #params=self.leerHiperparametros(modelo, objetivo)
        if modelo=="RL":
            self.admite_cualitativo=False
            scaler=None
            model = compose.Pipeline(preprocessing.StandardScaler(), multiclass.OneVsRestClassifier(linear_model.LogisticRegression()))
            return scaler, model
        elif modelo=="NB":
            self.admite_cualitativo=False
            scaler = preprocessing.MinMaxScaler()
            model = naive_bayes.MultinomialNB(alpha=0.05)
            return scaler, model
        elif modelo=="DT":
            self.admite_cualitativo=False
            #admite_cualitativo=True #da problemas con el drift
            scaler=None
            model = tree.HoeffdingTreeClassifier()
            return scaler, model
        else:
            print("INTRODUZCA UN MODELO CORRECTO.")
            return None, None
        
        #seleccion de métricas según objetivo cuantitativo o cualtitativo (y binario o multilevel)?
        #acc = metrics.Accuracy()
        #conf_matrix=metrics.ConfusionMatrix()
        
    
    def actuacionDrift(self, consigna, muestra, modelo, grafica):
        if consigna=="reentrenar":
            print("Detectado drift. Se reentrena.")
            muestras_restantes = self.num_datos - muestra
            if muestras_restantes >100_000:
                self.fin_train=muestra +50_000
                return modelo, 50_000
            elif muestra <0.89*self.num_datos:
                self.fin_train=muestra + int(0.5*muestras_restantes)
                return modelo, int(0.1*muestras_restantes)
            else:
                print("Queda menos del 10% de datos no se reentrena.")
                return modelo, 0
            
        elif consigna=="nuevo":
            print("Detectado drift. Se genera un nuevo modelo.")
            muestras_restantes = self.num_datos - muestra
            if muestras_restantes >100_000:
                self.modelos[self.n_modelo]=modelo
                self.n_modelo=self.n_modelo+1
                _, modelo=self.selectorModelo(self.algoritmo, self.objetivo)
                self.fin_train=muestra +50_000
                grafica.plotTrain(muestra)
                return modelo, 50_000
            elif muestra <0.89*self.num_datos:
                self.modelos[self.n_modelo]=modelo
                self.n_modelo=self.n_modelo+1
                _, modelo=self.selectorModelo(self.algoritmo, self.objetivo)
                self.fin_train=muestra + int(0.5*muestras_restantes)
                grafica.plotTrain(muestra)
                return modelo, int(0.5*muestras_restantes)
            else:
                print("Queda menos del 10% de datos no se genera nuevo modelo.")
                return modelo, 0
        else:
            return modelo, 0
    
    def valorNumerico(self, valor):
        try:
            return float(valor)
        except:
            return valor

    def extract_features(self, data, admite_cualitativo):
        X=dict(itertools.islice(data[0].items(), len(data[0])-3))
        if not self.admite_cualitativo:
            X.pop("FEATURE76")
            X.pop("FEATURE87")
        X.pop("unix")
        X.update((k, self.valorNumerico(v)) for k, v in X.items())
        return X

    def get_target(self, data, target):
        #y=[data[0]["m_id"],data[0]["m_subid"],data[0]["alarms"]]
        #y=dict(itertools.islice(data[0].items(), len(data[0])-1, len(data[0])))
        y=dict(itertools.islice(data[0].items(), len(data[0])-3, len(data[0])))
        y=y[target]
        return y
        #return list(y.values())[0]
    
    def ejecucion(self):
                
        #comprobnar que hay archivos
        if self.files is not None:
            
            scaler, model=self.selectorModelo(self.algoritmo, self.objetivo)
            if model is not None:
                tempo=reloxo.ElapsedTimer()
                print(f"MODELO {self.algoritmo} - {self.objetivo.upper()} - {tempo.current_time()}")
           
                #metricas. en el futuro selector de métricas en función de tipo de variable
                acc = metrics.Accuracy()
                conf_matrix=metrics.ConfusionMatrix()
           
                #drift articial
                drift=self.selectorDrift(self.aplicar_drift)
                if drift is not None:
                    self.admite_cualitativo=False
                    
                #detector anomalias

                detectorAN=anomaliaPropio.anomalyDetection(self.algoritmoAnomalia, self.umbral_deteccion)
                hay_anomalia=False
                
                #dectector drift
                detectorDR=driftPropio.driftDetection(self.algoritmo_drift)
                hay_drift=False
                
                #grafica dinamica    
                grafica=plotPerformance.plotMetrica(f"MODELO {self.algoritmo} - "+self.objetivo.upper(), "accuracy")
       
                muestras=0
                _fin_train=self.fin_train
                
                for file in self.files:
                    # Simular iteración
                    dataset = stream.iter_csv(file)
           
                    for data_point in dataset:
                   
                        features = self.extract_features(data_point, self.admite_cualitativo)
                        
                        if ((drift is not None) and (drift.activated)):
                        
                            features=drift.addNoise(features)
                        
                        if scaler is not None:
                        
                            scaler.learn_one(features)
                        
                            features = scaler.transform_one(features)
           
                        target = self.get_target(data_point, self.objetivo)
                        
                        #entrenamiento y deteccion de anomalías                   
                        if _fin_train>0:
                            if detectorAN is not None:
                                detectorAN.entrenarDetector(features)
                            if ((muestras > self.it_anomalia) and (detectorAN is not None)):
                                # se le pasa X, y, y_pred antes de actualizar modelo. para que diga si hay anomalía
                                hay_anomalia=detectorAN.anomalia_TF(features,target, model.predict_one(features))
                            else:
                                hay_anomalia==False
                            if hay_anomalia==False:
                                model.learn_one(features, target)
                            _fin_train=_fin_train-1
                            if _fin_train==0:
                                grafica.plotTest(muestras)

                        
                        if muestras >0:
                            # entrenamiento y detección de drift
                            if detectorDR is not None:
                                detectorDR.entrenarDetector(detectorAN.anomalia_score(features))
                                if muestras > self.fin_train:
                                    hay_drift=detectorDR.driftDetectado
                                    if hay_drift:
                                        #grafica.plotDriftDetected(muestras)
                                        #print("Drift Detectado.")
                                        #actuacion en base a drift
                                        model, _fin_train=self.actuacionDrift(self.consigna_drift, muestras, model, grafica)
                                    
                            
                            y_pred = model.predict_one(features)
                            conf_matrix.update(target, y_pred)
                            acc.update(target, y_pred)
                        if ((muestras >0) and (muestras % self.iterPrint==0)):
                            print(f'MODELO {self.algoritmo} - Accuracy para {file} - Objetivo: {self.objetivo.upper()}: \n \t {round(acc.get(),2)} - Muestras analizadas: {muestras}')
                            grafica.processDataPoint(muestras, acc.get())

                        #inicio generacion drift
                        if ((muestras==self.it_drift) and (drift is not None)):
                            drift.disparador()
                            grafica.plotDriftStart(muestras)
                        #fin generacion drift
                        if drift is not None:
                            if muestras==(self.it_drift+drift.duracion):
                                grafica.plotDriftEnd(muestras)
                        muestras=muestras + 1

                    grafica.processDataPoint(muestras, acc.get())
                    grafica.printFile(muestras, file.split("/")[-1])
               
                       
                    print(f'MODELO {self.algoritmo} - Accuracy final para {file} - Objetivo: {self.objetivo.upper()}: \n \t {round(acc.get(),2)}')
                       
                print(f'Accuracy final MODELO {self.algoritmo} - Objetivo: {self.objetivo.upper()}: \n \t {round(acc.get(),2)} - {tempo.elapsed_time()}')
                #guardar ultimo modelo en dicc
                self.modelos["MODELO_"+str(self.n_modelo)]=[model, conf_matrix]
                
                if drift is None:
                    grafica.savePlot("streaming_"+self.algoritmo+"_"+self.objetivo, round(acc.get(),2), len(self.files), tempo.elapsed_time(), "No")
                    joblib.dump(self.modelos,"OUT/MODELS/streaming_"+self.algoritmo+"_"+self.objetivo+".joblib")
                else:
                    grafica.savePlot("streaming_"+self.algoritmo+"_"+self.objetivo+"_dr"+self.aplicar_drift, round(acc.get(),2), len(self.files), tempo.elapsed_time(),self.aplicar_drift)
                    joblib.dump(self.modelos,"OUT/MODELS/streaming_"+self.algoritmo+"_"+self.objetivo+"_dr"+self.aplicar_drift+".joblib")

                print(f'Verdaderos positivos (TP): {conf_matrix.total_true_positives*100/conf_matrix.n_samples} %')
                print(f'Verdaderos negativos (TN): {conf_matrix.total_true_negatives*100/conf_matrix.n_samples} %')
                print(f'Falsos positivos (FP): {conf_matrix.total_false_positives*100/conf_matrix.n_samples} %')
                print(f'Falsos negativos (FN): {conf_matrix.total_false_negatives*100/conf_matrix.n_samples} %')

        return self.modelos, detectorAN.num_anomalias