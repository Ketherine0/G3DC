import pandas as pd
import numpy as np
import scipy.sparse as sp

def encode_onehot(labels):
    unique = set(labels)
    seq = enumerate(unique)
    uni_dict = {id: np.identity(len(unique))[i,:] for i, id in seq}
    onehot_labels = np.array(list(map(uni_dict.get, labels)), dtype=np.int32)
    # print(onehot_labels)
    labels = []
    for i in onehot_labels:
        labels.append(int(np.where(i!=0)[0]))
    return labels

def pre_adj(adj, symmetric=True):
    adj = normalize_adj(adj, symmetric)
    return adj

def normalize_adj(adj, symmetric=True):
    # D^(-1/2)*A*D^(-1/2)
    if symmetric:
        D = sp.eye(adj.shape[0]) - sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        adj_norm = adj.dot(D).transpose().dot(D).tocsr()
    else:
        D = sp.eye(adj.shape[0]) - sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        adj_norm = D.dot(adj).tocsr()
    return adj_norm

def load_data(path1,path2,path3):
    # X -------------------------------------
    d = pd.read_csv(path1)
    gene_name = d.iloc[:,1]


    X_df = d.iloc[:,2:]
    X = X_df.values
    Xmax, Xmin = X.max(axis=0), X.min(axis=0)
    X = (X - Xmin) / (Xmax - Xmin)

    X = np.transpose(X)
    gene_name = gene_name.values


    # Adj -------------------------------------
    edge = pd.read_csv(path2)
    edge = edge.iloc[:,:]
    edge = edge.values

    gene_name_li = gene_name.tolist()
    edge_li = []
    for i in edge:
        if i[0] in gene_name_li:
            if i[1] in gene_name_li:
                if i[0] != i[1]:
                    ind1 = gene_name_li.index(i[0])
                    ind2 = gene_name_li.index(i[1])
                    edge_li.append(list((ind1,ind2)))

    edge_np = np.array(edge_li)
    adj = sp.coo_matrix((np.ones(len(edge_li)), (edge_np[:, 0], edge_np[:, 1])),
                        shape=(len(gene_name), len(gene_name)), dtype=np.float32)
    adj += adj.T - sp.diags(adj.diagonal())
    a = adj
    adj = pre_adj(adj)
    adj = adj.todense()

    # y -------------------------------------
    y_df = pd.read_csv(path3)
    y_len = len(y_df)
    y = y_df.values
    y_li = []
    for i in range(y_len):
        y_li.append(y[i][0])

    y_np = np.array(y_li)
    labels = encode_onehot(y_np)

    return X, labels, adj, gene_name, edge_li, a

# X, labels, adj, gene_name, edge_li, a  = load_data(p1,p2,p3)
