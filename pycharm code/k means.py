import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_excel('500 comp.xlsx')


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

plt.scatter(x=df['Activo'], y=df['Roe'], c=df['kmeans_3'])
plt.xlim(-0.1, 30)
plt.xlabel('Activo', fontsize=16)
plt.ylim(-100,350)
plt.ylabel('Roe', fontsize=16)
plt.show()





# area = np.pi * (X[:, 1])**2
# plt.scatter(X[:,0], X[:,1], s=area, c=labels, alpha=0.5)
# plt.xlabel('Empl', fontsize=18)
# plt.ylabel('Activo', fontsize=16)
# #
# plt.show()

from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(8, 6))
# plt.clf()
#
# ax = fig.add_subplot(projection='3d')
# ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
#
# ax.set_xlabel('Да')
# ax.set_ylabel('Два')
# ax.set_zlabel('ТРИ')
#
# ax.scatter(X[:, 0],X[:, 1],X[:, 2], c= labels)




# fig = plt.figure(figsize=(7, 4))
# ax_3d = fig.add_subplot(projection='3d')
#
# x = np.arange
# y = np.arange
#
# x = np.arange(-2*np.pi, 2*np.pi, 0.2)
# y = np.arange(-2*np.pi, 2*np.pi, 0.2)
# xgrid, ygrid = np.meshgrid(x,y)
# zgrid = np.sin(xgrid) * np.sin(ygrid) / (xgrid * ygrid)
#
# ax_3d.set_xlabel('x')
# ax_3d.set_ylabel('y')
# ax_3d.set_zlabel('z')
#
# ax_3d.scatter(xgrid, ygrid, zgrid, s=1, color='g')
#
# plt.show()

# X = df[['Activo','Empl']].copy()
# wcss = []
# #the elbow method plots whithin-cluster-sum-of-squares (WCSS) vs the number of clusters
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
#
# sns.set()
# plt.plot(range(1,11,1),wcss)
# plt.title('Selecting the number of clusters using the elbow method')
# plt.xlabel('clusters')
# plt.ylabel('WCSS')
# print(plt.show())
#
# plt.scatter(df["Activo"],df['Empl'])
# plt.title('Activo and Empl')
# plt.xlabel('Activo')
# plt.ylabel('Employers')
# print(plt.show())