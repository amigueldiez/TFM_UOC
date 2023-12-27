# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:39:07 2023

@author: pmartinez
"""

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output

class plotMetrica:
    def __init__(self, model_name, metric_name):
        # Set Seaborn style
        sns.set(style="darkgrid")
        plt.tight_layout()
        # Create a figure and axis for the plot
        self.fig, self.axs = plt.subplots(figsize = (20, 15))
        self.axs.set_title(f'Rendimiento Modelo - {model_name} - {metric_name}')

        # Initialize a Seaborn line plot
        self.line = sns.lineplot(x=[0], y=[0], ax=self.axs)

        # Lists to store metrics for plotting
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
        #self.line.set_data(self.iterations, self.metrics)
        self.line = sns.lineplot(x=self.iterations, y=self.metrics, ax=self.axs)
        self.axs.relim()
        self.axs.autoscale_view()

        # Redraw and pause the figure
        clear_output(wait=True)
        display(self.fig)
    def save_plot(self, name):
        plt.savefig(name+".png",dpi=300)
        

