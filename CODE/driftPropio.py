# -*- coding: utf-8 -*-
"""
@author: pablo
"""

from river import drift

class driftDetection:
    def __init__(self, detector_drift):
        self.detector_drift=detector_drift
        
        self.w_size=0
        
        self.detector=None
                
        self.selectorDetector()
            
            
    def selectorDetector(self):
        if self.detector_drift=="ADWIN":
            print("DETECCIÓN DRIFT - ADWIN")
            self.detector = drift.ADWIN()
        elif self.detector_drift=="KSWIN":
            print("DETECCIÓN DRIFT - KSWIN")
            self.detector = drift.KSWIN()
        elif self.detector_drift=="PH":
            print("DETECCIÓN DRIFT - PH")
            self.detector = drift.PageHinkley()
        else:
            print("DRIFT - INTRODUZCA UN OBJETIVO CORRECTO.")
            self.detector = None
            
    def entrenarDetector(self, X):
        if self.detector is not None:
            self.detector.update(X)

            
    def driftDetectado(self):
        return self.detector.drift_detected