# -*- coding: utf-8 -*-
"""
@author: pablo
"""

from river import preprocessing, compose

from river import anomaly
from river import metrics



class anomalyDetection:
    def __init__(self, algoritmo, umbral_deteccion):
        self.algoritmo=algoritmo
        self.umbral_deteccion=umbral_deteccion
        self.num_anomalias=0
        self.detector=None
        
        self.metric=None
        
        self.selectorDetector()
        
        
    def selectorDetector(self):
        if self.algoritmo=="HST":
            print("DETECCIÓN ANOMALÍA- ALGORITMO HST")
            self.detector = compose.Pipeline(preprocessing.MinMaxScaler(), anomaly.HalfSpaceTrees())
        elif self.algoritmo=="LOF":
            print("DETECCIÓN ANOMALÍA- ALGORITMO LOF")
            self.detector = compose.Pipeline(anomaly.LocalOutlierFactor(n_neighbors=15))
        elif self.algoritmo=="OCSVM":
            print("DETECCIÓN ANOMALÍA - ALGORITMO OCSVM")
            self.detector = anomaly.QuantileFilter(anomaly.OneClassSVM(nu=0.5),q=0.995)
            self.metric=metrics.ROCAUC()
        else:
            print("ANOMALÍA - INTRODUZCA UN ALGORITMO CORRECTO.")

            
    def entrenarDetector(self, x):
        if self.detector is not None:
            self.detector.learn_one(x)

            

    def anomalia_TF(self, x, y, y_pred):
        if self.detector is not None:
            if self.algoritmo=="OCSVM":
                score = self.detector.score_one(x)
                is_anomaly = self.detector.classify(score)
                self.metric.update(y==y_pred, is_anomaly)
                print("AUCROC", self.metric)
                if is_anomaly:
                    self.num_anomalias=self.num_anomalias+1
                return is_anomaly
            else:
                score=self.detector.score_one(x)
                if score > self.umbral_deteccion:
                    self.num_anomalias=self.num_anomalias+1
                    return True
                    
                else:
                    return False
        else:
            return False
        
    def anomalia_score(self, x):
        if self.detector is not None:       
            score=self.detector.score_one(x)
        return score


