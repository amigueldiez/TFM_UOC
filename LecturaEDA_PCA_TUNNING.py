# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:51:09 2023

@author: pablo
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from matplotlib.pyplot import cm
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import Reloxo


#%% Agregación
def agregarDatos(file_paths, chunk_size=150):
    datos_numericos = pd.DataFrame()
    datos_categoricos = pd.DataFrame()
    
    for file_path in file_paths:
    
        data_chunks = pd.read_csv(file_path, chunksize=chunk_size)
        tempo=Reloxo.ElapsedTimer()
        for chunk in data_chunks:
            
            numerical_columns = chunk.select_dtypes(include=['number'])
            string_columns = chunk.select_dtypes(include=['object'])
    
            numerical_chunk_mean = numerical_columns.mean()
    
            datos_numericos = pd.concat([datos_numericos,numerical_chunk_mean],axis=1, ignore_index=True)
            
            #string_chunk_mode = string_columns.mode().iloc[0]
            string_chunk_mode = selectorNiveles(string_columns)
    
            datos_categoricos = pd.concat([datos_categoricos, string_chunk_mode],axis=1, ignore_index=True)
        
        #os.remove(file_path)
        print("ARQUIVO %s AGREGADO...(%s)" % (file_path, tempo.elapsed_time()))

            
    pd.concat([datos_numericos.transpose(),datos_categoricos.transpose()],axis=1).to_csv('datos_agregados.csv', index=False)
    
def agregarDatosTandas(file_paths, chunk_size=150):
    datos_numericos = pd.DataFrame()
    datos_categoricos = pd.DataFrame()
    
    if os.path.exists('datos_agregados.csv'):
        os.remove('datos_agregados.csv')
    
    for n, file_path in enumerate(file_paths):
    
        data_chunks = pd.read_csv(file_path, chunksize=chunk_size)
        tempo=Reloxo.ElapsedTimer()
        for chunk in data_chunks:
            
            numerical_columns = chunk.select_dtypes(include=['number'])
            string_columns = chunk.select_dtypes(include=['object'])
    
            numerical_chunk_mean = numerical_columns.mean()
    
            datos_numericos = pd.concat([datos_numericos,numerical_chunk_mean],axis=1, ignore_index=True)
            
            #string_chunk_mode = string_columns.mode().iloc[0]
            string_chunk_mode = selectorNiveles(string_columns)
    
            datos_categoricos = pd.concat([datos_categoricos, string_chunk_mode],axis=1, ignore_index=True)
    
        if n==20:
            pd.concat([datos_numericos.transpose(),datos_categoricos.transpose()],axis=1).to_csv('datos_agregados.csv', index=False)
            datos_numericos = pd.DataFrame()
            datos_categoricos = pd.DataFrame()
        elif n>20 and n%20==0:
            agregado_previo=pd.read_csv('datos_agregados.csv')
            num_previo=agregado_previo.select_dtypes(include=['number'])
            cat_previo=agregado_previo.select_dtypes(include=['object'])
            
            datos_numericos = pd.concat([num_previo,datos_numericos.transpose()],axis=0, ignore_index=True)
            datos_categoricos = pd.concat([cat_previo,datos_categoricos.transpose()],axis=0, ignore_index=True)
            pd.concat([datos_numericos,datos_categoricos],axis=1).to_csv('datos_agregados.csv', index=False)
            datos_numericos = pd.DataFrame()
            datos_categoricos = pd.DataFrame()
            
        #os.remove(file_path)
        print("ARCHIVO %s AGREGADO...(%s) - %s de %s" % (file_path, tempo.elapsed_time(), n+1, len(file_paths)))
        
    #agregación de los últimos
    agregado_previo=pd.read_csv('datos_agregados.csv')
    num_previo=agregado_previo.select_dtypes(include=['number'])
    cat_previo=agregado_previo.select_dtypes(include=['object'])
    
    datos_numericos = pd.concat([num_previo,datos_numericos.transpose()],axis=0, ignore_index=True)
    datos_categoricos = pd.concat([cat_previo,datos_categoricos.transpose()],axis=0, ignore_index=True)
    pd.concat([datos_numericos,datos_categoricos],axis=1).to_csv('datos_agregados.csv', index=False)    
        
            
#%% EDA
def EDA(dataset):
    
    fecha=dataset["unix"]
    dataset=dataset.drop("unix", axis=1)
    
    #para variables cuantitativas
    cuantitativa=dataset.describe()
    
    #para variables cualitativas
    cualitativa=dataset.select_dtypes(include=['object'])
    cualitativa=[cualitativa[c].value_counts() for c in cualitativa.columns]
    
    #graficas iniciales
    for n, columna in enumerate(dataset.columns):
        if n%9==0:
            plt.figure(figsize=(20, 20))
            plt.tight_layout()
        if dataset[columna].dtype=="object":
            plt.subplot(3, 3,n%9+1)
            plt.bar(dataset[columna].value_counts().index, height= dataset[columna].value_counts(), color="red")
            #plt.hist(dataset[columna], color='red', bins=len(dataset[columna].unique()))
            if(len(dataset[columna].unique()))>10:
                plt.xticks(rotation=45, fontsize=5)
            plt.title(f'{columna} - CATEGÓRICA')
            #plt.xlabel(columna)
            plt.ylabel('Count')
        else:
            plt.subplot(3, 3,n%9+1)
            plt.hist(dataset[columna], color='skyblue', bins=9)
            plt.title(f'{columna} - Númerica')
            #plt.xlabel(columna)
            plt.ylabel('Count')
            
    #series temporales
    fecha=pd.to_datetime(fecha,unit='s')
    for n, columna in enumerate(dataset.select_dtypes(include=['number']).columns):
        if n%3==0:
            plt.figure(figsize=(20, 20))
            plt.tight_layout()
        plt.subplot(3, 1,n%3+1)
        plt.plot(list(fecha), list(dataset[columna]), color='orange')
        plt.title(f'{columna} - Númerica')
        plt.xticks(rotation=45, fontsize=5)
        #plt.xlabel(columna)
        plt.ylabel('Value')
        
    return cuantitativa, cualitativa
  
#%% PCA      
def pca2D(dataset, numPC=2):
    
    num = dataset.select_dtypes(include=['number'])
    num= num.drop("unix", axis=1)
    
    cat = dataset.select_dtypes(include=['object'])
    
    x = StandardScaler().fit_transform(num)
    
    pca = PCA(n_components=numPC)
    
    principalComponents = pca.fit_transform(x)
    
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ["PC_"+str(i+1) for i in range(numPC)])
    
    #colores=cm.rainbow(range(len(leyenda)))
    #punto_c
    plt.figure(figsize=(15, 15))
    plt.xlabel("PCA1", fontsize=20 )
    plt.ylabel("PCA2", fontsize=20 )
    plt.scatter(principalDf["PC_1"],principalDf["PC_2"])
    plt.title(f"PCA - Componentes: {pca.n_components} - Varianza explicada: {sum(pca.explained_variance_ratio_)}", fontsize=30 )
    plt.tight_layout()
    plt.show()
    
    for columna in cat.columns:
        leyenda=cat[columna].unique()
        #colores=cm.rainbow(range(len(leyenda)))
        #punto_c
        plt.figure(figsize=(15, 15))
        plt.xlabel("PCA1", fontsize=20 )
        plt.ylabel("PCA2", fontsize=20 )
        plt.scatter(principalDf["PC_1"],principalDf["PC_2"], c=LabelEncoder().fit_transform(cat[columna]))
        plt.title(f"PCA VS {columna} - Componentes: {pca.n_components} - Varianza explicada: {sum(pca.explained_variance_ratio_)}", fontsize=30 )
        plt.legend(labels=leyenda,loc="upper left", fontsize=20 )
        plt.tight_layout()
        plt.show()
    return pca
    
    
#%% LDA
def lda2D(dataset, numC=2):
    
    num = dataset.select_dtypes(include=['number'])
    num= num.drop("unix", axis=1)
    
    # Separating out the target
    cat = dataset.select_dtypes(include=['object'])
    
    # Standardizing the features
    x = StandardScaler().fit_transform(num)
    
    for columna in cat.columns:
        
        y=cat[columna]
        lda = LinearDiscriminantAnalysis(n_components=numC)
        principalComponents = lda.fit_transform(x, y)
        
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ["LD_"+str(i+1) for i in range(numC)])
        
        plt.figure(figsize=(15, 15))
        plt.xlabel("LD1A", fontsize=20 )
        plt.ylabel("LDA2", fontsize=20 )
        leyenda=cat[columna].unique()
        plt.scatter(principalDf["LD_1"],principalDf["LD_2"], c=LabelEncoder().fit_transform(y))
        plt.title(f"LDA VS {columna} - Componentes: {lda.n_components} - Varianza explicada: {sum(lda.explained_variance_ratio_)}", fontsize=30 )
        plt.legend(labels=leyenda,loc="upper left", fontsize=20 )
        plt.tight_layout()
        plt.show()
    

#%% FiltradoCuantitativo

def selectorNiveles(lote):
    valores=[]
    for columna in lote.columns:
        if "none" not in lote[columna].value_counts():
            valores.append(lote[columna].value_counts().index[0])
        elif not lote[columna].value_counts().drop("none").empty:
            valores.append(lote[columna].value_counts().drop("none").index[0])
        else:
            valores.append("none")
    #print(valores)
    return pd.Series(valores, index=lote.columns)

#%% Tuning de hiperparámetros


