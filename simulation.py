# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:38:45 2021

@author: Juanyam
"""

## Problem 1##

#pip install pycairo
#pip install cairocffi
#load library
from igraph import *
import igraph as ig 
import csv,igraph
import cairocffi
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.io as spio


###function remove###
def edge_removal(A,phi):
    '''Remove edges by the trick developed by Mollison and Grassberger
    Args:
        A: numpy.ndarray
            Adjacency matrix of the dataset
        phi: float
            prob. of the edge is present

    Returns:
        A_new: numpy.ndarray
            Adjacency matrix after edge removal
    '''
    P = np.triu(np.random.random_sample(A.shape),1)
    Prob = P+P.T 
    A_new = ((A * Prob)> 1-phi).astype(int)
    return A_new


#pycairo

#import the dataset
annots = loadmat('facebook-ego.mat')                   #load data from mat.
resource=annots['A']                                   #get matrix from the dict
#print(data)

gra_data=Graph.Adjacency(resource.tolist())            #transform format from matrix to graph adjacency

Gx = nx.Graph(resource)

##### Problem 1 (a) #####
num_v=gra_data.vcount()                                #caculated number of vertex
#num_e=gra_data.ecount()/2                             #caculated number of edges
num_e=Gx.number_of_edges()
mean_C=2*num_e/num_v                                   #caculated number of mean degree
max_c=gra_data.maxdegree()/2                           #maximal degree
g_dia=gra_data.diameter(directed=False)                #diameter
avg_clust_coff=nx.average_clustering(Gx)

# show the detail information for this network
#summary(gra_data)
print("Number of nodes={0},\nNumber of edges ={1},\nMean degree. ={2},\nMaximum degree={3}".format(num_v,num_e,mean_C,max_c))
print('Diameter={0}'.format(g_dia))
print('Average clustering cofficient={0}'.format(avg_clust_coff))


##### Promblem 1(b) #####
#save as png named graph
#network_graph=ig.plot(gra_data,"graph.png",bbox=(0,0,600,600))
network_graph=ig.plot(gra_data,"graph_10.png",bbox=(0,0,1000,1000))            #visiualize the grapgh in different frame(600,1000,5000) 
#network_graph=ig.plot(gra_data,"graph_50.png",bbox=(0,0,5000,5000))


##### Problem 1 (c) #####
def plotDegDistLogLog(G, loglog = True):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    frac = [n/G.number_of_nodes() for n in cnt]
    fig, ax = plt.subplots()
    plt.plot(deg, frac, '.')
    if loglog:
        ax.set_yscale('log', nonposy='clip')
        ax.set_xscale('log', nonposx='clip')
    plt.ylabel("Fraction of nodes")
    plt.xlabel("Degree")
    
plotDegDistLogLog(Gx, loglog = True)

##### Problem 1 (d) #####

#degree centrality
degree_nx=nx.degree_centrality(Gx)
print('degree centrality top 10 node =\n')
print(sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)[0:10])

##katz 
eigenvalue=nx.adjacency_spectrum(Gx)

k=max(eigenvalue)
kk=round(k.real,4)

o=1/kk
alpha=o-0.005
if alpha<0 or alpha >=o:
    print("error")
else:
    katzcent=nx.katz_centrality(Gx,alpha)
    KN=sorted(katzcent.items(),key=lambda s:s[1], reverse=True)
    print("katz centrality:")
    for i in range(10):
        print(KN[i])


#eigenvector centrality

eig_nx = nx.eigenvector_centrality(Gx)
print('eigenvector centrality top 10 node =\n')
#sort the eigenvalue centrality from max to min
print(sorted(eig_nx.items(), key=lambda d: d[1], reverse = True)[0:10])

#betweeness 
between_nx=nx.betweenness_centrality(Gx)
print('betweenness centrality top 10 node =\n')
#sort the eigenvalue centrality from max to min
print(sorted(between_nx.items(), key=lambda d: d[1], reverse = True)[0:10])

#closeness centrality
closeness_nx = nx.closeness_centrality(Gx)
print('closeness centrality top 10 node=\n')
print(sorted(closeness_nx.items(), key=lambda d: d[1], reverse = True)[0:10])


##### Problem 2 (a) #####

deg=np.zeros(6)
eig=np.zeros(6)
kat=np.zeros(6)
bet=np.zeros(6)
clo=np.zeros(6)
#Remove 0%
#degree /eigenvector/katz/betweeness/closeness

y=np.zeros(100)         #Run 100 times
n_samp=range(0,100)

for c in n_samp:
    
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
        #print(i)
    #print(min(seed_index))
    
    x=temp_x                                                        #get vector x for |S|=5
    
    ## Remove 0% node##
    Rem_A=edge_removal(resource, 0.1)                               #Remove edge
    D=gra_data.diameter(directed=False) 
    I=np.identity(num_v)                                            #identity
    y_temp=((Rem_A+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y[c]=num_inflect


#prevalence for 0% Remove
deg[0]=(mean(y))/num_v
eig[0]=(mean(y))/num_v
kat[0]=(mean(y))/num_v
bet[0]=(mean(y))/num_v
clo[0]=(mean(y))/num_v

## Remove 10% node ##

##Degree centrality
num_remove=math.floor(num_v*0.1)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
deg_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
degree_nx=nx.degree_centrality(Gx)

count=range(0,num_remove)

sort_degree=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_degree]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_deg_10=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=deg_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_deg_10[c]=num_inflect
    
    

##Katz centrality
num_remove=math.floor(num_v*0.1)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
katz_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
#katz_nx=nx.katz_centrality(Gx,alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight='weight')

eigenvalue=nx.adjacency_spectrum(Gx)

k=max(eigenvalue)
kk=round(k.real,4)

o=1/kk
alpha=o-0.005
if alpha<0 or alpha >=o:
    print("error")
else:
    katzcent=nx.katz_centrality(Gx,alpha)
    KN=sorted(katzcent.items(),key=lambda s:s[1], reverse=True)

count=range(0,num_remove)

sort_katz=KN
b = [i[0] for i in sort_katz]               #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_katz_10=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)                #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=katz_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_katz_10[c]=num_inflect
  
    
  
    
###eigenvector 
num_remove=math.floor(num_v*0.1)           #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
eig_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
eig_nx=nx.eigenvector_centrality(Gx)

count=range(0,num_remove)

sort_eig=sorted(eig_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_eig]                #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_eig_10=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)               #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=eig_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_eig_10[c]=num_inflect
    
    
###betweeness
num_remove=math.floor(num_v*0.1)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
bet_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
bet_nx=nx.betweenness_centrality(Gx)

count=range(0,num_remove)

sort_bet=sorted(bet_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_bet]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_bet_10=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=bet_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_bet_10[c]=num_inflect
    
    
### closeness 

num_remove=math.floor(num_v*0.1)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
clos_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
clos_nx=nx.closeness_centrality(Gx)

count=range(0,num_remove)

sort_clos=sorted(clos_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_clos]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_clo_10=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=clos_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_clo_10[c]=num_inflect
    


deg[1]=(mean(y_deg_10))/num_v
eig[1]=(mean(y_eig_10))/num_v
kat[1]=(mean(y_katz_10))/num_v
bet[1]=(mean(y_bet_10))/num_v
clo[1]=(mean(y_clo_10))/num_v


## Remove 20% node ##

##Degree centrality
num_remove=math.floor(num_v*0.2)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
deg_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
degree_nx=nx.degree_centrality(Gx)

count=range(0,num_remove)

sort_degree=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_degree]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_deg_20=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=deg_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_deg_20[c]=num_inflect
    
    

##Katz centrality
num_remove=math.floor(num_v*0.2)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
kat_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
#katz_nx=nx.katz_centrality(Gx,alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight='weight')

eigenvalue=nx.adjacency_spectrum(Gx)

k=max(eigenvalue)
kk=round(k.real,4)

o=1/kk
alpha=o-0.005
if alpha<0 or alpha >=o:
    print("error")
else:
    katzcent=nx.katz_centrality(Gx,alpha)
    KN=sorted(katzcent.items(),key=lambda s:s[1], reverse=True)
    
count=range(0,num_remove)

sort_katz=KN
b = [i[0] for i in sort_katz]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_katz_20=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=kat_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_katz_20[c]=num_inflect
  
    
  
    
###eigenvector 
num_remove=math.floor(num_v*0.2)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
eig_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
eig_nx=nx.eigenvector_centrality(Gx)

count=range(0,num_remove)

sort_eig=sorted(eig_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_eig]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_eig_20=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=eig_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_eig_20[c]=num_inflect
    
    
###betweeness
num_remove=math.floor(num_v*0.2)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
bet_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
bet_nx=nx.betweenness_centrality(Gx)

count=range(0,num_remove)

sort_bet=sorted(bet_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_bet]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_bet_20=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=bet_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_bet_20[c]=num_inflect
    
    
### closeness 

num_remove=math.floor(num_v*0.2)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
clos_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
clos_nx=nx.closeness_centrality(Gx)

count=range(0,num_remove)

sort_clos=sorted(clos_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_clos]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_clo_20=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=clos_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_clo_20[c]=num_inflect
    


deg[2]=(mean(y_deg_20))/num_v
eig[2]=(mean(y_eig_20))/num_v
kat[2]=(mean(y_katz_20))/num_v
bet[2]=(mean(y_bet_20))/num_v
clo[2]=(mean(y_clo_20))/num_v


## Remove 30% node ##

##Degree centrality
num_remove=math.floor(num_v*0.3)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
deg_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
degree_nx=nx.degree_centrality(Gx)

count=range(0,num_remove)

sort_degree=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_degree]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_deg_30=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=deg_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_deg_30[c]=num_inflect
    
    

##Katz centrality
num_remove=math.floor(num_v*0.3)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
kat_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
#katz_nx=nx.katz_centrality(Gx,alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight='weight')

eigenvalue=nx.adjacency_spectrum(Gx)

k=max(eigenvalue)
kk=round(k.real,4)

o=1/kk
alpha=o-0.005
if alpha<0 or alpha >=o:
    print("error")
else:
    katzcent=nx.katz_centrality(Gx,alpha)
    KN=sorted(katzcent.items(),key=lambda s:s[1], reverse=True)

count=range(0,num_remove)

sort_katz=KN
b = [i[0] for i in sort_katz]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_katz_30=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=kat_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_katz_30[c]=num_inflect
  
    
  
    
###eigenvector 
num_remove=math.floor(num_v*0.3)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
eig_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
eig_nx=nx.eigenvector_centrality(Gx)

count=range(0,num_remove)

sort_eig=sorted(eig_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_eig]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_eig_30=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=eig_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_eig_30[c]=num_inflect
    
    
###betweeness
num_remove=math.floor(num_v*0.3)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
bet_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
bet_nx=nx.betweenness_centrality(Gx)

count=range(0,num_remove)

sort_bet=sorted(bet_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_bet]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_bet_30=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=bet_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_bet_30[c]=num_inflect
    
    
### closeness 

num_remove=math.floor(num_v*0.3)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
clos_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
clos_nx=nx.closeness_centrality(Gx)

count=range(0,num_remove)

sort_clos=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_clos]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_clo_30=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=clos_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_clo_30[c]=num_inflect



deg[3]=(mean(y_deg_30))/num_v
eig[3]=(mean(y_eig_30))/num_v
kat[3]=(mean(y_katz_30))/num_v
bet[3]=(mean(y_bet_30))/num_v
clo[3]=(mean(y_clo_30))/num_v



## Remove 40% node ##

##Degree centrality
num_remove=math.floor(num_v*0.4)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
deg_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
degree_nx=nx.degree_centrality(Gx)

count=range(0,num_remove)

sort_degree=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_degree]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_deg_40=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=deg_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_deg_40[c]=num_inflect
    
    

##Katz centrality
num_remove=math.floor(num_v*0.4)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
kat_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
#katz_nx=nx.katz_centrality(Gx,alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight='weight')

eigenvalue=nx.adjacency_spectrum(Gx)

k=max(eigenvalue)
kk=round(k.real,4)

o=1/kk
alpha=o-0.005
if alpha<0 or alpha >=o:
    print("error")
else:
    katzcent=nx.katz_centrality(Gx,alpha)
    KN=sorted(katzcent.items(),key=lambda s:s[1], reverse=True)

count=range(0,num_remove)

sort_katz=KN
b = [i[0] for i in sort_katz]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_katz_40=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=kat_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_katz_40[c]=num_inflect
  
    
  
    
###eigenvector 
num_remove=math.floor(num_v*0.4)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
eig_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
eig_nx=nx.eigenvector_centrality(Gx)

count=range(0,num_remove)

sort_eig=sorted(eig_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_eig]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_eig_40=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=eig_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_eig_40[c]=num_inflect
    
    
###betweeness
num_remove=math.floor(num_v*0.4)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
bet_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
bet_nx=nx.betweenness_centrality(Gx)

count=range(0,num_remove)

sort_bet=sorted(bet_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_bet]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_bet_40=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=bet_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_bet_40[c]=num_inflect
    
    
### closeness 

num_remove=math.floor(num_v*0.4)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
clos_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
clos_nx=nx.closeness_centrality(Gx)

count=range(0,num_remove)

sort_clos=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_clos]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_clo_40=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=clos_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_clo_40[c]=num_inflect


deg[4]=(mean(y_deg_40))/num_v
eig[4]=(mean(y_eig_40))/num_v
kat[4]=(mean(y_katz_40))/num_v
bet[4]=(mean(y_bet_40))/num_v
clo[4]=(mean(y_clo_40))/num_v



## Remove 50% node ##

##Degree centrality
num_remove=math.floor(num_v*0.5)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
deg_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
degree_nx=nx.degree_centrality(Gx)

count=range(0,num_remove)

sort_degree=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_degree]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_deg_50=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=deg_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_deg_50[c]=num_inflect
    
    

##Katz centrality
num_remove=math.floor(num_v*0.5)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
kat_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
#katz_nx=nx.katz_centrality(Gx,alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight='weight')

eigenvalue=nx.adjacency_spectrum(Gx)

k=max(eigenvalue)
kk=round(k.real,4)

o=1/kk
alpha=o-0.005
if alpha<0 or alpha >=o:
    print("error")
else:
    katzcent=nx.katz_centrality(Gx,alpha)
    KN=sorted(katzcent.items(),key=lambda s:s[1], reverse=True)

count=range(0,num_remove)

sort_katz=KN
b = [i[0] for i in sort_katz]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_katz_50=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=kat_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_katz_50[c]=num_inflect
  
    
  
    
###eigenvector 
num_remove=math.floor(num_v*0.5)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
eig_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
eig_nx=nx.eigenvector_centrality(Gx)

count=range(0,num_remove)

sort_eig=sorted(eig_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_eig]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_eig_50=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=eig_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_eig_50[c]=num_inflect
    
    
###betweeness
num_remove=math.floor(num_v*0.5)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
bet_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
bet_nx=nx.betweenness_centrality(Gx)

count=range(0,num_remove)

sort_bet=sorted(bet_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_bet]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_bet_50=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=bet_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_bet_50[c]=num_inflect
    
    
### closeness 

num_remove=math.floor(num_v*0.5)        #計算要RM點的個數(無條件捨去)
RM_edge=edge_removal(resource,0.1)
clos_gra=Graph.Adjacency(RM_edge.tolist())  #igraph 
Gx = nx.Graph(RM_edge)     
clos_nx=nx.closeness_centrality(Gx)

count=range(0,num_remove)

sort_clos=sorted(degree_nx.items(), key=lambda d: d[1], reverse = True)
b = [i[0] for i in sort_clos]             #取值
Ndeg_centr=b[0:num_remove]                  #要RM的NODE    

Rm_node=RM_edge
##REMOVE
for i in count:
    Rm_node[:,Ndeg_centr[i]]=0
    Rm_node[Ndeg_centr[i],:]=0



#remove vertex by function

y_clo_50=np.zeros(100)              #看要RUN幾次調整
n_sample=range(0,100)        #看要RUN幾次調整
for c in n_sample:
    
    seed_index=np.random.choice(num_v-1,5,replace=False)              #initial inflected for five (No replace)
    temp_x=np.zeros(num_v)                                            #set S into zeros elements
    sequence=[0,1,2,3,4]
    for i in sequence:
       temp_x[seed_index[i]]=1
    x=temp_x   
       
    gra_data=Graph.Adjacency(Rm_node.tolist()) 
    D=deg_gra.diameter(directed=False) 
    I=np.identity(num_v) 
    y_temp=((Rm_node+I)**D).dot(x)
    y_temp[(y_temp>=1)]=1
    num_inflect=sum(y_temp)
    y_clo_50[c]=num_inflect
    
deg[5]=(mean(y_deg_50))/num_v
eig[5]=(mean(y_eig_50))/num_v
kat[5]=(mean(y_katz_50))/num_v
bet[5]=(mean(y_bet_50))/num_v
clo[5]=(mean(y_clo_50))/num_v


percent = np.array([0, 10, 20, 30, 40,50])
#pre=plt.plot(deg,percent)
plt.figure(figsize = (6, 4.5), dpi = 100)                 # 設定圖片尺寸
plt.xlabel('r (m)', fontsize = 16)                        # 設定坐標軸標籤
plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
plt.yticks(fontsize = 12)
plt.xlabel('rate')
plt.ylabel(' Persent (%)')

line1, = plt.plot(deg, percent, color = 'red', linewidth = 3, label = 'Degree Centrality')             
line2, = plt.plot(kat, percent, color = 'blue', linewidth = 3, label = 'Katz Centrality')
line3, = plt.plot(eig, percent, color = 'yellow', linewidth = 3, label = 'Eigenvector Centrality')             
line4, = plt.plot(bet, percent, color = 'black', linewidth = 3, label = 'Betweeness Centrality')
line5, = plt.plot(clo, percent, color = 'green', linewidth = 3, label = 'Closeness Centrality')


plt.legend(handles = [line1, line2,line3, line4, line5], loc='upper right')
plt.show()






    
