import gym
from gym import spaces
import numpy as np
#from gnn_model import Reinforced_GAT
from gnn_model_1 import Reinforced_GAT
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from utils import data_preprocessing, normalize_adj, sparse_to_tuple
from clustering_loss import ClusteringLoss, SubClusteringLoss
#from evaluation import eva
from evaluation_1 import eva_metrics
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
import os.path as osp
import random
import os
import copy
from ae import AE
from sklearn.preprocessing import normalize

def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class MyModel(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, n_classes, premodel_path, v=1):
        super(MyModel, self).__init__()
        self.n_classes = n_classes
        self.v = v

        self.ae = AE(num_features, hidden_size, embedding_size, alpha)

        self.gat = Reinforced_GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(premodel_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_classes, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        set_seed()

    def forward(self, inputs, adj, action, breadth_policy=None, deepth_policy=None):
        ae_z, ae_x_bar, enc_h1 = self.ae(inputs)
        gnn_a_pred, gnn_z = self.gat(inputs, adj, action, breadth_policy)
        ae_q = self.get_Q(ae_z)
        gnn_q = self.get_Q(gnn_z)
        all_z = ae_z + gnn_z
        all_q = self.get_Q(ae_z+gnn_z)
        return gnn_a_pred, all_z, all_q, ae_x_bar, ae_q, gnn_q, gnn_z

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class CustomGNNEnv(gym.Env):
    def __init__(self, state_size, action_size, init_state, recent_step,
                 num_features, hidden_size, embedding_size, alpha, gnn_lr,
                 adj_ori, adj_norm, adj_lab, features, att_sim, label, policy, breadth_action, premodel_path="", label_indices = 0):
        super(CustomGNNEnv, self).__init__()

        self.observation_space = state_size
        self.action_space = action_size

        self.init_state = init_state
        self.max_steps = 10
        self.past_performance = [0]
        self.recent_step = recent_step

        self.adj_ori = adj_ori
        self.adj_norm = adj_norm
        self.adj_lab = adj_lab
        self.features = features
        self.label = label
        self.num_classes = label.max()+1
        self.reinforced_gnn_model = MyModel(num_features, hidden_size, embedding_size, alpha, label.max()+1, premodel_path, v=1)
        self.optimizer = torch.optim.Adam(self.reinforced_gnn_model.parameters(), lr=gnn_lr, weight_decay=5e-3)
        self.clustering_loss = ClusteringLoss(adj_ori, att_sim)
        self.subclustering_loss = SubClusteringLoss(adj_ori, att_sim)
        self.label_indices = label_indices
        self.criterion = nn.CrossEntropyLoss()
        self.policy = policy
        self.breadth_action = breadth_action

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)  # config for CPU
        torch.cuda.manual_seed(seed)  # config for GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
        # config for cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    #强化后的gnn模型训练，并得到评估结果
    def evaluate(self, action, past_res, i=1):
        if self.policy == "breadth":
            breadth_policy = True
            action = action
        else:
            breadth_policy = None

        self.reinforced_gnn_model.train()

        A_pred, all_z, Q, ae_x_bar, ae_Q, gnn_Q, gnn_z = self.reinforced_gnn_model(self.features, self.adj_norm, action, breadth_policy)
        p = target_distribution(Q.detach())
        U = F.softmax(Q, dim=1)
        membership = U.detach().data.cpu().numpy().argmax(1)
        clu_res = eva_metrics(self.label, membership[self.label_indices])

        #mse_loss
        kl_loss = F.kl_div((Q.log()+ae_Q.log()+gnn_Q.log())/3, p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), self.adj_norm.view(-1))
        clu_loss, structure_loss, attribute_loss, struc_and_attri = self.subclustering_loss(U)
        ae_loss = F.mse_loss(ae_x_bar, self.features)
        loss = kl_loss + 0.1*re_loss + clu_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        cur_res = -struc_and_attri.detach().data.cpu().numpy()
        reward = np.where(cur_res > past_res, 1, np.where(past_res > cur_res, -1, 0))

        return reward, cur_res, clu_res

def data_prepare(dataset_all, name=None):
    dataset = data_preprocessing(dataset_all, name)
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





