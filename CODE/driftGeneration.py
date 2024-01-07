# -*- coding: utf-8 -*-
"""
@author: pmartinez
"""
import random

random.seed(7)

class randomDrift():

    def __init__(self, n_var=10, factor=0.1, duracion=500000):
        
        self.n_var=n_var
        self.keys_affected = []
        self.noise_factor = factor
        self.duracion=duracion
        self.activated=False
        self.count=0

    def addNoise(self, X):
        if self.count==0:
            self.keys_affected = self.randomKeys(X, self.n_var)
        if self.activated:
            Xn=X.copy()
            for key in self.keys_affected:
                Xn[key] *= (1+ random.uniform(-self.noise_factor, self.noise_factor))
            self.count=self.count+1
            if self.count==self.duracion:
                self.reset()
            return Xn
    
    def randomKeys(self, dictionary, n):
        return random.sample(dictionary.keys(), n)
    
    def reset(self):
        self.keys_affected = []
        self.activated=False
        self.count=0
        print("TERMINA DRIFT ALEATORIO.")
        
    def disparador(self):
        #Aquí estaría bien preparar un disparo aleatorio
        self.activated=True
        print("ARRANCA DRIFT ALEATORIO.")


class crossDrift:
    
    def __init__(self, n_var=10, duracion=500000):
        
        self.n_var=n_var
        self.keys_affected = []
        self.keys_changed=[]
        self.duracion=duracion
        self.activated=False
        self.count=0

    def addNoise(self, X):
        if self.count==0:
            self.keys_affected = self.randomKeys(X, self.n_var)
            self.keys_changed = list(self.keys_affected)
            random.shuffle(self.keys_changed)
        if self.activated:
            Xn=X.copy()
            for key_orig, key_ch in zip(self.keys_affected, self.keys_changed):
                Xn[key_orig] = X[key_ch]
            self.count=self.count+1
            if self.count==self.duracion:
                self.reset()
            return Xn
    
    def randomKeys(self, dictionary, n):
        return random.sample(dictionary.keys(), n)
    
    def reset(self):
        self.keys_affected = []
        self.keys_changed=[]
        self.activated=False
        self.count=0
        print("TERMINA DRIFT CRUCE VARIABLES.")
        
    def disparador(self):
        #Aquí estaría bien preparar un disparo aleatorio, una vez que se inicializa.
        self.activated=True
        print("ARRANCA DRIFT CRUCE VARIABLES.")