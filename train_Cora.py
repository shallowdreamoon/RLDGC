import numpy as np
from config import all_seed,  set_seed, agent_config, get_args
from multi_armed_bandit import BernoulliBandit, UCB
from clustering_specific_gnn import CustomGNNEnv, data_prepare
from torch_geometric.datasets import Planetoid
import os.path as osp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from evaluation import eva
from utils import save_result, get_action
import time
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def all_policy_train(cfg, gnn_env, action, action_size):
    best_res = [0]
    best_eva_res = [0]
    bandit_arm = BernoulliBandit(gnn_env)
    epsilon_greedy_solver = UCB(bandit_arm, action.numpy(), policy="breadth", action_size=action_size, coef=0.2)
    best_action = action
    cur_res = best_res
    best_gnn_model = gnn_env
    consecutive_step = 0
    accumulated_reward = 0
    best_res = best_eva_res
    for epoch in range(1):
        best_gnn_model.policy = "breadth"
        best_res, best_action, reward, best_model, cur_res = epsilon_greedy_solver.run(100, best_gnn_model, cur_res, best_res)
        accumulated_reward += reward
        if best_res[0] > best_eva_res[0]:
            best_eva_res = best_res
            consecutive_step = 0
        else:
            consecutive_step += 1

    return best_eva_res, best_action

def gnn_train(gnn_env, best_res, best_action, best_model_path = "", clu_eva_res_path="", save = False, patience = 50):
    consecutive_step = 0
    for i in range(100):
        temp_model = copy.deepcopy(gnn_env)
        reward, cur_res, clu_res = gnn_env.evaluate(best_action, best_res[0], i)
        print("epoch: {} res: {}".format(i, clu_res))
        if clu_res[0] > best_res[0]:
            best_res = clu_res
            best_model = temp_model
            consecutive_step = 0
        else:
            consecutive_step += 1
        if consecutive_step > patience:
            break
    return best_res


def get_data(dataset):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
    dataset_all = Planetoid(path, dataset)
    features, adj_ori, adj_norm, adj_lab, label = data_prepare(dataset_all[0], name="cora")
    label_indices = np.arange(features.shape[0])
    att_sim = cosine_similarity(features.numpy())
    att_sim = torch.from_numpy(att_sim)
    att_sim = adj_ori * att_sim
    return features, adj_ori, adj_norm, adj_lab, label, label_indices, att_sim

def task(cfg, dataset, policy = None):
    features, adj_ori, adj_norm, adj_lab, label, label_indices, att_sim = get_data(dataset)
    num_nodes = features.shape[0]
    state_size = features.shape[1]
    init_state = features
    recent_step = 5
    num_features = features.shape[1]
    hidden_size = 256
    embedding_size = 16
    alpha = 0.2
    breadth_init_action = adj_lab.sum(dim=1).int()
    # breadth_init_action  = torch.randint(1, 400, (adj_ori.shape[0], ))
    breadth_action_size = breadth_init_action.max()

    premodel_path = r"E:\study\pycharm\RLDGC\RLDGC\pretrain_res\cora\premodel_cora.pkl".format(dataset)
    gnn_env = CustomGNNEnv(state_size, breadth_action_size, init_state, recent_step,
                           num_features, hidden_size, embedding_size, alpha, cfg['gnn_lr'],
                           adj_ori, adj_norm, adj_lab, features, att_sim, label, policy=None,
                           breadth_action=breadth_init_action, premodel_path=premodel_path, label_indices=label_indices)

    # 5.初始化模型参数（聚类中心）
    agent_config(gnn_env, cfg)
    with torch.no_grad():
        _, z = gnn_env.reinforced_gnn_model.gat(features, adj_norm, breadth_init_action)
    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=label.max() + 1, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    gnn_env.reinforced_gnn_model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_)
    res = eva(label, y_pred)
    print("Epoch: {} acc: {} nmi: {} f1: {} ari: {}".format(-1, res[0], res[1], res[2], res[3]))



    # print("start training whole model...\n")
    best_eva_res, breadth_action = all_policy_train(cfg, gnn_env, breadth_init_action, breadth_action_size)
    print("\nbest_res of whole model: {}".format(best_eva_res))
    print("\nstart training gnn model...\n")
    gnn_env.policy = "ALL"
    best_action = torch.tensor(breadth_action)
    best_res = gnn_train(gnn_env, [0], best_action, patience=50)
    print(best_res)


    return None

if __name__ == '__main__':
    set_seed(seed=10)
    #1.获取参数
    cfg = get_args()
    dataset = "cora"
    start = time.time()
    task(cfg, dataset, policy="Breadth") 
    print("All time cose: {}s".format(time.time()-start))









