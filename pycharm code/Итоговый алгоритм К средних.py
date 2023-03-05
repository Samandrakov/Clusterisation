import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_excel('500 comp.xlsx')
df = df.to_excel('res1.xlsx')

df = pd.read_excel('res1.xlsx')

df = df.drop(['Company','Sector',"Country","neto",'oper', 'Ventas'],  axis=1)
df.dropna(inplace=True)
print(df.head())



X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
# print(Clus_dataSet)

    #Функция оптимального количества кластеров

def optimise_k_means(data, max_k):

    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    #График плеча (не обращаем внимание на предупреждения в ходе запуска)
    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
optimise_k_means(df[['Activo', 'Roe']],10)

    #Создание кластеров - Проведение кластеризации

kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['Activo', 'Roe']])
df['kmeans_3'] = kmeans.labels_

    #Готовый к визуализации дб с средними значениями

print(df)

    #Среднее значение (можно заигнорить)

print(df.groupby('kmeans_3').mean())

    #Кластерный график

plt.scatter(x=df['Roe'], y=df['Activo'], c=df['kmeans_3'])
plt.xlim(-200, 1000, 100)
plt.xlabel('Roe', fontsize=16)
plt.ylim(0.5,40)
plt.ylabel('Активы компании', fontsize=16)
plt.show()