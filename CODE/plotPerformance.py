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

        self.fig, self.axs = plt.subplots(figsize = (20, 15))
        self.axs.set_title(f'Rendimiento Modelo - {model_name} - {metric_name}', fontsize=30)
        self.axs.set_xlabel("Muestras analizadas (n)",fontsize=20)
        self.axs.set_ylabel(metric_name,fontsize=20)

        self.line = sns.lineplot(x=[0], y=[0], ax=self.axs)

        self.iterations =[0]
        self.metrics = [0]
        
    def process_data_point(self, it, met):
        self.iterations.append(it)
        self.metrics.append(met)
        
        self.update_plot()
    
    def plot_anomaly(self, it):
        self.axs.axvline(x=it, color='red', linestyle='--')
        self.update_plot()
    
    def update_plot(self):
        # Aquí conseguir actualizar no crear otro gráfico
        #
        #
        #self.line.set_data(self.iterations, self.metrics)
        # self.line.x=self.iterations
        # self.line.y=self.metrics
        # self.line.ax=self.axs
        self.line = sns.lineplot(x=self.iterations, y=self.metrics, ax=self.axs)

        self.axs.relim()
        self.axs.autoscale_view()

        # Redraw and pause the figure
        clear_output(wait=True)
        display(self.fig)
        
    def save_plot(self, name):
        plt.savefig("OUT/PLOTS/"+name+".png",dpi=300)
        

