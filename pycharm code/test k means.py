import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('Empl-Roe Orig.xlsx', index_col="Roe")
# print(df.head())



df.drop([0], inplace= True)

# print(df)

df = df.to_excel('result1.xlsx')
df = pd.read_excel('result1.xlsx', index_col="Empl")
df.drop([0], inplace= True)

# print(df)
df = df.to_excel('result2.xlsx')
df1 = pd.read_excel('result2.xlsx')



# sns.set()
# plt.plot(range(1,11,1),wcss)
# plt.title('Selecting the number of clusters using the elbow method')
# plt.xlabel('clusters')
# plt.ylabel('WCSS')
# print(plt.show())

plt.scatter(df1['Roe'],df1['Empl'])
plt.title('Roe and Empl')
plt.xlabel('Roe')
plt.ylabel('Empl')
print(plt.show())