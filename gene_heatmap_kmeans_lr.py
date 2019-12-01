# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:32:24 2019

@author: Chenyu Sun
"""
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


with open('BRCA.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt','r') as f:
    a = f.readlines()
#print(len(a))
#print(type(float(a[32])))

with open('gene_list.txt','r') as f:
    b = f.read()
b = b[:len(b)-1]
#print(b)
b = list(b.split("\n"))
#print(b)

l = []

for i in range(len(a)):
    index = a[i].find('|')
    k = a[i][:index]
    #print(k)
    if k in b:
        #print(k)
        l.append(a[i][:(len(a[i])-1)]) #don't want \n
print(len(l), "genes match")

#convert string to list
m = []
for i in range(len(l)):
    new_l = list(l[i].split("\t")) 
    m.append(new_l)
#print(m)

#f(x) = log2(x + 1)
m = np.array(m)
n = m[:,1:] #pick out the numbers only
#print(n)
n = np.array(n, dtype=np.float32) #convert to float
n = np.log2(n+1)
#print(n)
#k = np.hstack((np.asarray([m[:,0]]).transpose(),n))
#print(k.shape)

g = sns.clustermap(n)

###kmeans###

X = n.transpose()

for i in range(X.shape[1]):
    index = m[i][0].find('|')
    gene = m[i][0][:index]
    if gene == 'ESR1':
        a = i
        print('ESR1',i)
    if gene == 'ERBB2':
        b = i
        print('ERBB2',i)

plt.scatter(
   X[:, a], X[:, b],
   c='white', marker='o',
   edgecolor='black', s=50,
)
plt.xlabel("ESR1")
plt.ylabel("ERBB2")
plt.show()

km = KMeans(
    n_clusters=5, init='random',
    n_init=10, max_iter=50, 
    tol=1e-04, random_state=0
)

X_group = X[:,(a,b)]

y_km = km.fit_predict(X_group)

#plot the clusters

plt.scatter(
    X_group[y_km == 0, 0], X_group[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X_group[y_km == 1, 0], X_group[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X_group[y_km == 2, 0], X_group[y_km == 2, 1],
    s=50, c='lightblue',
    marker='o', edgecolor='black',
    label='cluster 3'
)

plt.scatter(
    X_group[y_km == 3, 0], X_group[y_km == 3, 1],
    s=50, c='pink',
    marker='o', edgecolor='black',
    label='cluster 4'
)

plt.scatter(
    X_group[y_km == 4, 0], X_group[y_km == 4, 1],
    s=50, c='gray',
    marker='o', edgecolor='black',
    label='cluster 5'
)


#plot centroid

plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.xlabel("ESR1")
plt.ylabel("ERBB2")
plt.show()

###bineray classifier###

#make a list of ground truth
y = []
for i in range(X.shape[0]):
    if X[i][16] < 9.0 and X[i][15] > 15.0:
        y.append(1)
    else:
        y.append(0)

#plt.scatter(
#    X[np.asarray(y) == 0, 0], X[np.asarray(y) == 0, 1],
#    s=50, c='lightgreen',
#    marker='s', edgecolor='black',
#    label='group 1'
#)
#
#plt.scatter(
#    X[np.asarray(y) == 1, 0], X[np.asarray(y) == 1, 1],
#    s=50, c='orange',
#    marker='o', edgecolor='black',
#    label='group 2'
#)
#plt.grid()
#plt.xlabel("ESR1")
#plt.ylabel("ERBB2")
#plt.show()
        
#train the classifier
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',max_iter=500).fit(X, y)
y_lr = LR.predict(X)


coef = abs(LR.coef_[0])
coef.sort()
thre = coef[-5]

w = LR.coef_[0]
for i in range(len(w)):
    index = m[i][0].find('|')
    gene = m[i][0][:index]
    if abs(w[i]) >= thre:
        print(gene, w[i])




