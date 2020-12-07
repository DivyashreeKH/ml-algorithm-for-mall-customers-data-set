import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('Mall_Customers_lyst8965.csv')
#print(data)
x=data.iloc[:,[3,4]].values
#print(x)
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=1)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)#contains all the varience for each value of k i.e, 10 in this case

plt.plot(range(1,11),wcss)
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=1)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='orange',label='Cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='yellow',label='Cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='pink',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='cyan',label='centroid')

plt.show()
