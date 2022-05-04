'''
author: Bodan Chen
Date: 2022-04-28 19:30:04
LastEditors: Bodan Chen
LastEditTime: 2022-05-01 20:13:15
Email: 18377475@buaa.edu.cn
'''
import numpy as np
import pandas as pd
import networkx as nx
#from c1 import read_node_label,Classifier
import c1
edges=pd.DataFrame()
edges['s']=[1,1,2]
edges['t']=[2,3,4]
edges['w']=[1,1,1]
print(edges)

G=nx.from_pandas_edgelist(edges,source='s',target='t',edge_attr='w')
# degree
print(nx.degree(G))
# liantongfenliang
print(list(nx.connected_components(G)))
# tuzhijing
print(nx.diameter(G))
# 