# -*- coding: utf-8 -*-
"""
@author: pmartinez
"""

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output

class plotMetrica:
    def __init__(self, model_name, metric_name):

        sns.set(style="darkgrid")
        plt.tight_layout()
        
        self.model_name=model_name
        self.metric_name=metric_name

        self.fig, self.axs = plt.subplots(figsize = (20, 15))
        self.axs.set_title(f'Rendimiento Modelo - {model_name} - {metric_name}', fontsize=30)
        self.axs.set_xlabel("Muestras analizadas (n)",fontsize=20)
        self.axs.set_ylabel(metric_name,fontsize=20)
        self.axs.set_ylim(0,1.05)

        self.line = sns.lineplot(x=[0], y=[0], ax=self.axs, color='blue', linewidth=3)

        self.iterations =[0]
        self.metrics = [0]
        
    def processDataPoint(self, it, met):
        self.iterations.append(it)
        self.metrics.append(met)
        
        self.updatePlot()
    
    def plotAnomaly(self, it):
        self.axs.axvline(x=it, color='red', linestyle='-')
        self.updatePlot()
        
    def plotDrift(self, it):
        self.axs.axvline(x=it, color='orange', linestyle='-.')
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

        self.axs.relim()
        self.axs.autoscale_view(scaley=False)

        # Redraw and pause the figure
        clear_output(wait=True)
        display(self.fig)
        
    def savePlot(self, name, metric, num, tiempo, dr):
        self.axs.set_title(f"""Rendimiento Modelo - {self.model_name} - {self.metric_name}: {metric}
                           \t Archivos: {num} - Tiempo: {tiempo} - Drift: {dr}""", fontsize=30)
        #self.fig.suptitle('Main title')
        plt.savefig("OUT/PLOTS/"+name+".png",dpi=300)
        

