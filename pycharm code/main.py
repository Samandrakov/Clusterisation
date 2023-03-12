import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

empres = pd.read_excel('500 comp.xlsx')
activo = pd.read_excel('activo.xlsx')
# print(empres.head())
# print(empres.shape)
print(len(activo))
    #Ссылка на статью с хабра тут:  https://habr.com/ru/post/585034/
    #кластеризация методом k средних
    # 1) Случайным образом создаются k точек, в дальнейшем будем называть их центрами кластеров;
    # 2) Для каждой точки ставится в соответствии ближайший к ней центр кластера;
    # 3) Вычисляются средние арифметические точек, принадлежащих к определённому кластеру.
    #    Именно эти значения становятся новыми центрами кластеров;
    # 4) Шаги 2 и 3 повторяются до тех пор, пока пересчёт центров кластеров будет приносить плоды.
    #    Как только высчитанные центры кластеров совпадут с предыдущими, алгоритм будет окончен.

    # n - количество строк
    # k - количество кластеров
    # dim - размерность точек (пространства)
    # cluster - двумерный массив размерностью dim * k, содержащий k точек — центры кластеров
    # cluster_content - массив, содержащий в себе k массивов — массивов точек принадлежащих соответствующему кластеру


def clusterization(array, k):
    n = len(array)
    dim = len(array[0])

    cluster = [[0 for i in range(dim)] for q in range(k)]
    cluster_content = [[] for i in range(k)]

    for i in range(dim):
        for q in range(k):
            cluster[q][i] = random.randint(0, max_cluster_value)

    cluster_content = data_distribution(array, cluster)

    privious_cluster = copy.deepcopy(cluster)
    while 1:
        cluster = cluster_update(cluster, cluster_content, dim)
        cluster_content = data_distribution(array, cluster)
        if cluster == privious_cluster:
            break
        privious_cluster = copy.deepcopy(cluster)

    def visualisation_2d(cluster_content):

        k = len(cluster_content)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("y")

        for i in range(k):
            x_coordinates = []
            y_coordinates = []
            for q in range(len(cluster_content[i])):
                x_coordinates.append(cluster_content[i][q][0])
                y_coordinates.append(cluster_content[i][q][1])
            plt.scatter(x_coordinates, y_coordinates)
        plt.show()

