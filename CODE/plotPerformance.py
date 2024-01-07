# -*- coding: utf-8 -*-
"""
@author: pmartinez
"""

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output

class plotMetrica:
    def __init__(self, model_name, metric_name, metric_name2):

        sns.set(style="darkgrid")
        plt.tight_layout()
        
        self.model_name=model_name
        self.metric_name=metric_name
        self.metric_name2=metric_name2

        self.fig, self.axs = plt.subplots(figsize = (20, 15))
        self.axs.set_title(f'Rendimiento Modelo - {model_name} - {metric_name}', fontsize=30)
        if self.metric_name2 is not None:
                self.axs.set_title(f'Rendimiento Modelo - {model_name} - {metric_name} - {metric_name2}', fontsize=30)
        self.axs.set_xlabel("Muestras analizadas (n)",fontsize=20)
        self.axs.set_ylabel("Métrica",fontsize=20)
        self.axs.set_ylim(0,1.05)

        self.line = sns.lineplot(x=[0], y=[0], ax=self.axs, color='blue', label=self.metric_name, linewidth=3)
        if self.metric_name2 is not None:
            self.line2 = sns.lineplot(x=[0], y=[0], ax=self.axs, color='grey', label=self.metric_name2, linewidth=3)
        
        self.axs.legend(prop = {'size' : 30}, loc = 'upper left')
        self.iterations =[0]
        self.metrics = [0]
        self.metrics2 = [0]
        
    def processDataPoint(self, it, met, met2):
        self.iterations.append(it)
        self.metrics.append(met)
        if met2 is not None:
            self.metrics2.append(met2)
        
        self.updatePlot()
    
    def plotAnomalyDetected(self, it):
        self.axs.axvline(x=it, ymax=0.2, color='red', linestyle='-')
        self.updatePlot()
        
    def plotDriftDetected(self, it):
        self.axs.axvline(x=it, ymax=0.2, color='gold', linestyle='-')
        self.updatePlot()
        
    def plotDriftStart(self, it):
        self.axs.axvline(x=it, color='orange', linestyle='-')
        self.updatePlot()
        
    def plotDriftEnd(self, it):
        self.axs.axvline(x=it, color='orange', linestyle='-.')
        self.updatePlot()
        
    def plotTrain(self, it):
        self.axs.axvline(x=it, color='green', linestyle='-')
        self.updatePlot()
        
    def plotTest(self, it):
        self.axs.axvline(x=it, color='green', linestyle='-.')
        self.updatePlot()
        
    def printFile(self, it, txt):
        self.axs.axvline(x=it, color='black', linestyle=':')
        #self.axs.text(it, 0.5, txt, rotation=90, fontsize=18)
        self.updatePlot()
    
    def updatePlot(self):
        # Aquí conseguir actualizar no crear otro gráfico
        #
        #
        # self.line.set_data(self.iterations, self.metrics)
        # self.line.x=self.iterations
        # self.line.y=self.metrics
        # self.line.ax=self.axs
        self.line = sns.lineplot(x=self.iterations, y=self.metrics, ax=self.axs, color='blue', linewidth=3)
        if self.metric_name2 is not None:
            self.line2 = sns.lineplot(x=self.iterations, y=self.metrics2, ax=self.axs, color='grey', linewidth=3)
        

        self.axs.relim()
        self.axs.autoscale_view(scaley=False)

        # Redraw and pause the figure
        clear_output(wait=True)
        display(self.fig)
        
    def savePlot(self, name, metric, num, datos, tiempo, dr):
        if self.metric_name2 is None:
            self.axs.set_title(f"""Rendimiento Modelo - {self.model_name} {self.metric_name}, : {metric}
                               \t Archivos: {num} - Datos: {datos} - Tiempo: {tiempo} - Drift: {dr}""", fontsize=30)
        else:
            self.axs.set_title(f"""Rendimiento Modelo - {self.model_name} 
                               \t {self.metric_name} - {self.metric_name2}: {metric}
                               \t Archivos: {num} - Datos: {datos} - Tiempo: {tiempo} - Drift: {dr}""", fontsize=30)
        #self.fig.suptitle('Main title')
        plt.savefig("OUT/PLOTS/"+name+".png",dpi=300)


