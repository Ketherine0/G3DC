import matplotlib.pyplot as plt
import numpy as np
import pandas
from singleCell2 import load_data
import pandas as pd
from matplotlib import pyplot as plt
import scanpy as sc
import torch


# f = pd.read_csv("ver5_3_tsne3_W/weight.csv")
# W = f.iloc[:,1].values
# W = np.loadtxt("ver5_3_tsne3_tr2_W/W 220 .txt")
W = np.loadtxt("ver5_3_tsne3_tr3_W/W 270_ori.txt")
W = np.transpose(W)
# f = pd.read_csv("ver5_3_tsne2_tr_W/W 100 .txt")
# f = pd.read_csv("ver5_3_tsne3_W/W 210 .txt")
# W = f.iloc[:,1].values

W_li = []
W = torch.tensor(W)
L2 = torch.mul(W, W).sum(1)
for i in L2:
    W_li.append(i)
W_li = np.array(W_li)

x1,x2,x3,x4,x5 = [],[],[],[],[]
x_ind = []

# for i in range(len(W_li)):
#     if W_li[i]<=0.25:
#         x1.append(W_li[i])
#     elif W_li[i]<=0.4:
#         x2.append(W_li[i])
#     elif W_li[i]<=0.5:
#         x3.append(W_li[i])
#     elif W_li[i]<= 0.65:
#         x4.append(W_li[i])
#     else:
#         x5.append(W_li[i])
#         x_ind.append(i)
# print(len(x_ind))
# x_name = ['>=0.55','>=0.45','>=0.35','>=0.25']
# x_len = [len(x5),len(x4)+len(x5),len(x3)+len(x4)+len(x5),len(x2)+len(x3)+len(x4)+len(x5)]
# plt.bar(x_name,x_len)
# plt.title('First layer weights of Segerstolpe pancreas network')
# plt.pause(0)

# nor_W = (W_li-min(W_li)/(max(W_li)-min(W_li)))
lar = np.array(pd.Series(W_li).sort_values().index[len(W_li)-200:len(W_li)])
x_ind = lar


x, y, adj, v, edge, a = load_data()
# impor_gene = v[lar]
impor_gene = v[x_ind]
# print(v[lar])
# hist, bin_edges = np.histogram(W_li,bins=40)
# print(hist)
# print(bin_edges)
#
# plt.hist(W_li,bins=40,rwidth=0.9)
# plt.title('histogram of Seg pancreas')
# plt.pause(0)

# adata = sc.read_10x_h5("cluster_compare/scDeepCluster-master/code/data_usoskin.h5")
# gene = sc.pl.highest_expr_genes(adata, n_top=200, )
# print(gene)

impor_gene2 = []
edge = pd.read_csv('../usoskin_edge.csv')
edge = edge.iloc[:, :]
edge = edge.values
edge1 = []
edge_ind = []
degree = [0]*len(impor_gene)
order2_edge = []
order2_ind = []
single = []

de = a.sum(axis=1)
for i in range(len(de)):
    if (i in x_ind and de[i][0]==0):
        single.append(i)
print(len(single))

impor_li = []
impo_name = []
for i in edge:
    if i[0] in impor_gene:
        if i[1] in impor_gene:
            if i[0] != i[1]:
                edge1.append((i[0], i[1]))
                edge_ind.append((list(v).index(i[0]),list(v).index(i[1])))
                # degree[list(impor_gene).index(i[0])] += 1
                # degree[list(impor_gene).index(i[1])] += 1
            if i[0] not in impo_name:
                a = list(v).index(i[0])
                impor_li.append([i[0],W_li[a]])
                impo_name.append(i[0])
                degree[list(impor_gene).index(i[0])] += 1
            if i[1] not in impo_name:
                b = list(v).index(i[1])
                impor_li.append([i[1], W_li[b]])
                impo_name.append(i[1])
                degree[list(impor_gene).index(i[1])] += 1
df = pd.DataFrame(impor_li, columns=['gene name', 'weight'])

all = []
for i in x_ind:
    all.append([v[i],W_li[i]])
df2 = pd.DataFrame(all,columns=['gene name', 'weight'])
print(df2.shape)
# df2.to_csv('impor_weight_all.csv')




# 二阶邻居
for i in edge:
    if (i[0] in impor_gene and i[1] in v):
        if (i[1] not in impor_gene and i[0]!=i[1]):
            impor_gene2.append(i[0])
            order2_edge.append((i[0], i[1]))
            order2_ind.append((list(v).index(i[0]), list(v).index(i[1])))
    if (i[1] in impor_gene and i[0] in v):
        if (i[0] not in impor_gene and i[0]!=i[1]):
            impor_gene2.append(i[1])
            order2_edge.append((i[0], i[1]))
            order2_ind.append((list(v).index(i[0]), list(v).index(i[1])))

impor_unique = np.unique(impor_gene2)
rank = []
sort = np.array(pd.Series(W_li).sort_values().index[0:len(W_li)])

for i in sort[::-1]:
    # if v[i] in impor_unique:
    rank.append([v[i],W_li[i]])
# print(rank)

impor_df = pd.DataFrame(rank, columns=['gene_name', 'weight'])
#将DataFrame存储为csv,index表示是否显示行名，default=True
# impor_df.to_csv("impor_GSEA5.csv",index=False,sep=',')

# second_neigh = []
# for i in order2_ind:
#     second_neigh.append(i[0])
#     second_neigh.append(i[1])
# second_neigh = np.unique(second_neigh)

if_impor_gene = []
for i in range(len(v)):
    if i in x_ind:
        if_impor_gene.append(1)
    else:
        if_impor_gene.append(0)

print(len(edge_ind))
# file_handle=open('Seg_cytoscape2.txt',mode='w')
# file_handle.writelines(['gene1', ',', 'gene2', '\n'])
# for i in range(len(edge1)):
#     file_handle.writelines([str(edge_ind[i][0]), ',', str(edge_ind[i][1]), '\n'])
# file_handle.close()
#
# file_handle=open('Seg_cytoscape_node2.txt',mode='w')
# file_handle.writelines(['gene', ',', 'name', ',','molecule', ',', 'degree', ',', 'if_important', '\n'])
# for i in range(len(impor_gene)):
#     file_handle.writelines([str(x_ind[i]), ',', impor_gene[i], ',','gene', ',', str(degree[i]), ',', str(1), '\n'])
# file_handle.close()
#
# print('----------------------------------')
# print(len(order2_ind))
# file_handle=open('Seg_total6_or2.txt',mode='w')
# file_handle.writelines(['gene1', ',', 'gene2', '\n'])
# for i in range(len(order2_ind)):
#     file_handle.writelines([str(order2_ind[i][0]), ',', str(order2_ind[i][1]), '\n'])
# for k in range(len(edge_ind)):
#     file_handle.writelines([str(edge_ind[k][0]), ',', str(edge_ind[k][1]), '\n'])
# for j in range(len(single)):
#     file_handle.writelines([str(single[j]),'\n'])
# file_handle.close()
#
file_handle=open('Seg_total_node6_or2.txt',mode='w')
file_handle.writelines(['gene', ',', 'name', ',','molecule', ',', 'if_important', ',','score','\n'])
for i in range(len(v)):
    file_handle.writelines([str(i), ',', v[i], ',','gene', ',', str(if_impor_gene[i]),',',str(W_li[i]),'\n'])
file_handle.close()
print('finish')

