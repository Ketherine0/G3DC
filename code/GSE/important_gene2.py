import pandas as pd
import torch
import sys
sys.path.append("..")
from metrics import *
from preprocess import load_data

W = np.loadtxt("../model_weight/GSE_weight/layer_weight/W3.txt")
W = np.transpose(W)


W_li = []
W = torch.tensor(W)
L2 = torch.mul(W, W).sum(1)
for i in L2:
    W_li.append(i)
W_li = np.array(W_li)

lar = np.array(pd.Series(W_li).sort_values().index[len(W_li)-400:len(W_li)])
print(len(lar))
x_ind = lar

x, y, adj, v, edge, a = load_data()
impor_gene = v[x_ind]

de = a.sum(axis=1)

edge = pd.read_csv('../../usoskin_edge.csv')
edge = edge.iloc[:, :]
edge = edge.values
edge1 = []
edge_ind = []
degree = [0]*len(impor_gene)
order2_edge = []
order2_ind = []
single = []

impor_li = []
impo_name = []

for i in edge:
    if i[0] in impor_gene:
        if i[1] in impor_gene:
            if i[0] != i[1]:
                edge1.append((i[0], i[1]))
                edge_ind.append((list(v).index(i[0]),list(v).index(i[1])))
                degree[list(impor_gene).index(i[0])] += 1
                degree[list(impor_gene).index(i[1])] += 1
                if i[0] not in impo_name:
                    a = list(v).index(i[0])
                    impor_li.append([i[0],W_li[a]])
                    impo_name.append(i[0])
                if i[1] not in impo_name:
                    b = list(v).index(i[1])
                    impor_li.append([i[1], W_li[b]])
                    impo_name.append(i[1])
# print(impor_li)
print("impor_li",len(impor_li))
print('name',len(impo_name))
df = pd.DataFrame(impor_li, columns=['gene name', 'weight'])
# df.to_csv('impor_weight_GSE.csv')

all = []
for i in x_ind:
    # if not v[i].endswith('RIK'):
    all.append([v[i],W_li[i]])
df2 = pd.DataFrame(all,columns=['gene name', 'weight'])
print(df2.shape)
# df2.to_csv('impor_weight_all_GSE2.csv')
print(len(edge_ind))


for i in range(len(de)):
    if (i in x_ind and de[i][0]==0):
        single.append(i)


for i in edge:
    if (i[0] in impor_gene and i[1] in v):
        if (i[1] not in impor_gene and i[0]!=i[1]):
            order2_edge.append((i[0], i[1]))
            order2_ind.append((list(v).index(i[0]), list(v).index(i[1])))
    if (i[1] in impor_gene and i[0] in v):
        if (i[0] not in impor_gene and i[0]!=i[1]):
            order2_edge.append((i[0], i[1]))
            order2_ind.append((list(v).index(i[0]), list(v).index(i[1])))

print(len(single))

rank = []
sort = np.array(pd.Series(W_li).sort_values().index[0:len(W_li)])

for i in sort[::-1]:
    # if v[i] in impor_unique:
    rank.append([v[i],W_li[i]])
# print(rank)

impor_df = pd.DataFrame(rank, columns=['gene_name', 'weight'])

second_neigh = []
for i in order2_ind:
    second_neigh.append(i[0])
    second_neigh.append(i[1])
second_neigh = np.unique(second_neigh)

if_impor_gene = []
for i in range(len(v)):
    if i in x_ind:
        if_impor_gene.append(1)
    else:
        if_impor_gene.append(0)

file_handle=open('GSE_total.txt',mode='w')
file_handle.writelines(['gene1', ',', 'gene2', '\n'])
for k in range(len(edge_ind)):
    file_handle.writelines([str(edge_ind[k][0]), ',', str(edge_ind[k][1]), '\n'])
for j in range(len(single)):
    file_handle.writelines([str(single[j]),'\n'])
file_handle.close()

file_handle=open('GSE_total_node.txt',mode='w')
file_handle.writelines(['gene', ',', 'name', ',','molecule', ',', 'if_important', ',','score', '\n'])
for i in range(len(v)):
    file_handle.writelines([str(i), ',', v[i], ',','gene', ',', str(if_impor_gene[i]),',',str(W_li[i]),'\n'])
file_handle.close()



