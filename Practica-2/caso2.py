#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 01:32:34 2017

@author: zes
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering,Birch,MeanShift,MiniBatchKMeans
from sklearn import metrics, preprocessing
from math import floor
import seaborn as sns
import time
import latex_generator as ltxgen



#devuelve las metricas CH y SH
def calcMetrics(prediction,data_norm):
    metric_CH = metrics.calinski_harabaz_score(data_norm, prediction)
    
    #el cálculo de Silhouette consume mucha RAM, se selecciona una muestra del 10%
    metric_SC = metrics.silhouette_score(data_norm, prediction, metric='euclidean', sample_size=floor(0.1 * len(data_norm)),
                                        random_state=123456)

    return [metric_CH,metric_SC]

#obtiene el data frame a partir de un conjunto de datos y los clusters
def toDataFrame(data,prediction):
    DF = pd.DataFrame(prediction,index=data.index,columns=['cluster'])
    X_DF = pd.concat([data,DF],axis=1)

    return X_DF

#dataframe normalizado a partir de un data frame
def toDataFrameNormalized(dataFrame):
    vars = list(dataFrame)
    vars.remove('cluster')

    norm = preprocessing.normalize(dataFrame,norm='l2')
    DF_norm = pd.DataFrame(norm,columns=vars, index=dataFrame.index)

    return DF_norm

#calcula los clusters, devolviendo: las métricas(CH,SH), el dataFrame, tiempo empleado y el cluster. En ese orden.
def computeClustering(data, data_norm, algorithim):
    time_start = time.time()
    prediction = algorithim.fit_predict(data_norm)
    time_finish = time.time() - time_start

    X_DF = toDataFrame(data,prediction)
    
    return [calcMetrics(prediction,data_norm),X_DF,time_finish,prediction]



#Graficas----------------------------------------------------------------------
def doScatterMatrix(data, color_var, path, muestra=False):
    sns.set()
    variables = list(data)
    variables.remove(color_var)
    sns_plot = sns.pairplot(data, vars=variables, hue=color_var, palette='Paired', 
                            plot_kws={'s': 25}, diag_kind='hist') 
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    
    plt.savefig(path)
    if muestra == False:
        plt.clf()
    
def doHeatMap(data, color_measure, path):
    sns.heatmap(data, linewidths=.1, cmap='Blues_r')
    
def doClusterHeatMapopt(DF, path, muestra = False):
    DF.pop('cluster')
    sns.clustermap(DF)
    plt.savefig(path+'simple.png')
    if muestra == False:
        plt.clf()
    sns.clustermap(DF, z_score = 0)
    plt.savefig(path+'row_normalized.png')
    if muestra == False:
        plt.clf()
    sns.clustermap(DF, metric="correlation")
    plt.savefig(path+'correlation.png')
    if muestra == False:
        plt.clf()

def doClusterHeatMap(DF, path, muestra = False):
    
    sns.clustermap(DF)
    plt.savefig(path+'simple.png')
    if muestra == False:
        plt.clf()
    sns.clustermap(DF, z_score = 0)
    plt.savefig(path+'row_normalized.png')
    if muestra == False:
        plt.clf()
    sns.clustermap(DF, metric="correlation")
    plt.savefig(path+'correlation.png')
    if muestra == False:
        plt.clf()
    
def saveAlgResults(metricas, tiempo, X_DF, path, prediction):
    f = open(path, 'w')
    k = len(set(prediction))
    f.write("número de clusters: {:}".format(k))
    f.write("     Número de muestras en cada cluster:\n{:}".format(X_DF['cluster'].value_counts()))
    f.write("     tiempo: {:.5f} segundos".format(tiempo))
    f.write("     Calinski-Harabaz Index: {:.3f}".format(metricas[0]))
    f.write("     Silhouette Coefficient: {:.5f}".format(metricas[1]))
    f.write("     número de muestras: {:}".format(len(X_DF)))
    f.close()
    
    print("     número de clusters: {:}".format(k))
    print("     Número de muestras en cada cluster:\n{:}".format(X_DF['cluster'].value_counts()))
    print("     tiempo: {:.5f} segundos".format(tiempo))
    print("     Calinski-Harabaz Index: {:.3f}".format(metricas[0]))
    print("     Silhouette Coefficient: {:.5f}".format(metricas[1]))
    print("     número de muestras: {:}".format(len(X_DF)))
    
    return X_DF['cluster'].value_counts()
    


#main CASE 2 --------------------------------------------------------------------------------------------------------------
#preparación de datos----------------------------------------------------------
accidentes = pd.read_csv('accidentes_2013.csv')

subset = accidentes

#salidas de via los fines de semana
subset = accidentes.loc[(accidentes['DIASEMANA']>=5) & (accidentes['DIASEMANA']<=7)]
subset = subset[subset['TIPO_ACCIDENTE'].str.contains('Salida de la vía')]


#seleccionar variables de interÃ©s para clustering
var_interes = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 
               'HORA']
X = subset[var_interes]

#Sampleado
#n = 3000
#X  = X.sample(n,random_state=123456)

#normalizamos
X_normal = preprocessing.normalize(X, norm='l2')
#END preparacion de datos------------------------------------------------------

#algoritmos elegidos-----------------------------------------------------------
n_Clust = 12
k_means = KMeans(init='k-means++', n_clusters=n_Clust, n_init=5)
Agg = AgglomerativeClustering(n_clusters=n_Clust, linkage='ward')
birch = Birch(n_clusters=n_Clust, threshold=0.2)
meanshift = MeanShift(bin_seeding=True)
mbkm = MiniBatchKMeans(n_clusters=n_Clust)

algorithms = (
        ('KMeans', k_means),
        ('Agglomerative_Clustering', Agg),
        ('Birch', birch),
        ('Mean_Shift', meanshift),
        ('MiniBatchKMeans', mbkm)
        )
#END algoritmos elegidos-------------------------------------------------------

#Sampleado especificos---------------------------------------------------------
#Agglomerative_Clustering
#Xward = X.sample(20000, random_state=123456) #20 000
#Xward_norm = preprocessing.normalize(Xward, norm='l2')

#Mean_Shift
#Xmns = X.sample(10000, random_state=123456) #10 000
#Xmns_norm = preprocessing.normalize(Xmns, norm='l2')

#END Sampleado especificos ----------------------------------------------------

#generar endograma/heatmap s.a. clusterHeatMap con ward
#Xclhm = X.sample(3000, random_state=123456)
print('generando clusters heatmaps',end='')
path = 'Caso_2/clustermaps/'
doClusterHeatMap(X, path)
print(' <---done, saved files in: {:}'.format(path))

#Clustering--------------------------------------------------------------------
#guardar resultados simplificados globales
resGlobales = list()

for name, algorithm in algorithms:
    print("---------->{:}".format(name)) 
    print('calculando clusters',end='')
    #if name == "Agglomerative_Clustering":
       # metricas, X_DF, tiempo, prediction = computeClustering(Xward, Xward_norm, algorithm)
    #elif name == 'Mean_Shift':
        #metricas, X_DF, tiempo, prediction = computeClustering(Xmns, Xmns_norm, algorithm)        
   # else:
    metricas, X_DF, tiempo, prediction = computeClustering(X, X_normal, algorithm) 
    print(' <---done, results:')
     
    path = "Caso_2/results" + name + '{:}'.format(n_Clust)+'.txt'
    statCluster = saveAlgResults(metricas, tiempo, X_DF, path, prediction)
    print('Results saved as {:}'.format(path))
    
    resGlobales.append(name)
    resGlobales.append('{:.3f}'.format(metricas[0]))
    resGlobales.append('{:.3f}'.format(metricas[1]))
    resGlobales.append('{:.5f}'.format(tiempo))
    
    
    
    print('\ngenerando ScatterMatrix', end='')
    path = 'Caso_2/scatter_plots/'+name+'{:}'.format(n_Clust)+'SP.png'
    doScatterMatrix(X_DF, 'cluster', path)
    print(' <---done, saved as: {:}'.format(path))
    
    
    
    print('\n\n')
    
#guardamos los resultados globales
path = "Caso_2/globresultsTable"+ '{:}'.format(n_Clust)+'.txt'
tablenames = ['Algoritmo', 'CH', 'SC', 'Tiempo']
tableGlobal = ltxgen.genLatexTable(tablenames, resGlobales)
f = open(path, 'w')
f.write(tableGlobal)
f.close()
#END Clustering----------------------------------------------------------------
#END main CASE 2 --------------------------------------------------------------------------------------------------------------




