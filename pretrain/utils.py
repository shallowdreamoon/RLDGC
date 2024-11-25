import numpy as np
import torch
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
import copy


def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_ori = copy.deepcopy(dataset.adj)
    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj_label = copy.deepcopy(dataset.adj)
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj_1(adj):
    #adj_ori = adj - sp.eye(adj.shape[0])  #ACM数据集使用
    adj_ori = adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    mx = adj
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = sparse_mx_to_torch_sparse_tensor(mx)
    return mx, adj_ori

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    解释一下:
        scipy coordinate稀疏矩阵类型转换成tuple
        coords[i]是一个二元组，代表一个坐标
        values[i]代表sparse_matrix[coords[i][0], coords[i][1]]的value
        shape=(2708, 1433)是稀疏矩阵的维度
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


#获得用于gnn模型训练的数据
def data_prepare(dataset_all):
    dataset = data_preprocessing(dataset_all)
    adj = dataset.adj
    adj = adj.numpy()
    adj = normalize_adj(adj)
    adj = sparse_to_tuple(adj)
    adj = torch.sparse.FloatTensor(
        torch.LongTensor(adj[0].transpose()),
        torch.FloatTensor(adj[1]),
        torch.Size(adj[2])
    )
    adj = adj.to_dense()
    adj_label = dataset.adj_label
    adj_ori = dataset.adj_ori

    # features and label
    features = torch.Tensor(dataset.x)
    y = dataset.y.cpu().numpy()
    #num_classes = y.max()+1
    return features, adj_ori, adj, adj_label, y

def save_result(filename, results):
    results = map(str, results)
    content = " ".join(results)
    with open(filename, "a+") as f:
        f.write('\n')
        f.writelines(content)
    f.close()

import scipy.sparse
import scipy.io as sio
def load_mat_data(str):
    data = sio.loadmat('E:\study\pycharm\GRL_Study\RL_Policy-GNN_V2\data\{}\ACM3025.mat'.format(str))
    if(str == 'large_cora'):
        X = data['X']
        A = data['G']
        gnd = data['labels']
        gnd = gnd[0, :]
    else:
        X = data['feature']
        A = data['PAP']
        B = data['PLP']
        av=[]
        av.append(A)
        av.append(B)
        gnd = data['label']
        #gnd = gnd.T
        #gnd = np.argmax(gnd, axis=0)

    adj = av[0].astype(np.float32)
    adj = scipy.sparse.csc_matrix(adj)
    X=X.astype(np.float32)
    gnd=gnd.astype(np.float32)
    return adj, X, gnd

import scipy.sparse
def load_npz_data(filename):
    with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
        loader = dict(loader)
        #indices表示所在列
        adjacency = scipy.sparse.csr_matrix((loader['adj_data'],      loader['adj_indices'],
                                             loader['adj_indptr']),   shape=loader['adj_shape'])
        features = scipy.sparse.csr_matrix(( loader['feature_data'],   loader['feature_indices'],
                                             loader['feature_indptr']),shape=loader['feature_shape'])
        label_indices = loader['label_indices']
        labels = loader['labels']
    assert adjacency.shape[0] == features.shape[0], 'Adjacency and feature size must be equal!'
    assert labels.shape[0] == label_indices.shape[0], 'Labels and label_indices size must be equal!'
    return adjacency, features, labels, label_indices