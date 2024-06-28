#%% Instalando pacotes e importando

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import open
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pd.options.display.max_columns = 150

#%% Importando base de dados

data_set = pd.read_csv('data-final.csv', sep='\t')
#fonte: https://www.kaggle.com/datasets/tunguz/big-five-personality-test

#%% Entendendo estrutura da base de dados

data_set.info()
data_set.describe()
data_head = data_set.head()

data_set = data_set.drop(data_set.columns[50:110], axis=1)

# Analisando estatisticas da base de dados
describe = data_set.describe()

data_set['EXT1'].value_counts()

#%% Filtrando base de dados para valores que são diferentes de 0

data_set = data_set[(data_set > 0.00).all(axis=1)]

#%% 
""" A regra de negócio dessa aplicação, 
é separar as pessoas por caracteristicas, formando 5 grupos,
portanto foi definido 5 cluster, 
porém também utilizei elbow.
"""



data_sample = data_set.sample(n=5000, random_state=1)
elbow = []


K = range(1,11) 

for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(data_sample)
    elbow.append(kmeanElbow.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11))
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% adicionando cluster na base de dados
kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(data_set)

data_kmeans = KMeans(n_clusters=5,
                       init='random',
                       random_state=100).fit(data_set)

predicoes = data_kmeans.labels_

data_set['cluster'] = predicoes

#%% Analisando os clusters

data_set['cluster'].value_counts()
data_gp_cluster = data_set.groupby(['cluster'])
mean_gp = data_gp_cluster.mean().T
describe_gp = data_gp_cluster.describe().T

#%% Extraindo padroes
col_list = list(data_set)
ext = col_list[0:10]
est = col_list[10:20]
agr = col_list[20:30]
csn = col_list[30:40]
opn = col_list[40:50]

data_soma = pd.DataFrame()
data_soma['extrovertido'] = data_set[ext].sum(axis=1)/10
data_soma['neurotico'] = data_set[est].sum(axis=1)/10
data_soma['agradavel'] = data_set[agr].sum(axis=1)/10
data_soma['diligente'] = data_set[csn].sum(axis=1)/10
data_soma['aberto'] = data_set[opn].sum(axis=1)/10
data_soma['cluster'] = predicoes

data_clusters = data_soma.groupby('cluster').mean()

#%% Plotando gráfico

plt.figure(figsize=(22,3))
for i in range(0, 5):
    plt.subplot(1,5,i+1)
    plt.bar(data_clusters.columns, data_clusters.iloc[:, i], color='green', alpha=0.2)
    plt.plot(data_clusters.columns, data_clusters.iloc[:, i], color='red')
    plt.title('Grupo ' + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0,4);
    
#%% Testando quando tem novas respostas

data_set[:0].to_excel("perguntas2.xlsx",index=False)

meus_dados = pd.read_excel("perguntas2.xlsx")

meus_dados.info()
data_set.info()

grupo_personalidade = k_fit.predict(meus_dados)[0]
print('Meu grupo de personalidade é: ', grupo_personalidade)
