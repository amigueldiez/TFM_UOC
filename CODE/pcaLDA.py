# -*- coding: utf-8 -*-
"""
@author: pablo
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from matplotlib.pyplot import cm
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

  
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
    plt.savefig("OUT/PLOTS/PCA.png", dpi=300, bbox_inches='tight')
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
        plt.savefig("OUT/PLOTS/PCA_"+columna+".png", dpi=300, bbox_inches='tight')
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
        plt.savefig("OUT/PLOTS/LDA_"+columna+".png", dpi=300, bbox_inches='tight')
        plt.show()
    
