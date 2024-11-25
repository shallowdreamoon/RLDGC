"""
每一个节点选择的邻居个数不相同,也就是说强化策略是针对于每一个节点的
"""
import copy
import time

import torch
import random
import os
import numpy as np

def set_seed(seed=1):
    ''' 万能的seed函数
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """

    def __init__(self, gnn_env):
        set_seed()
        self.num_nodes = gnn_env.adj_ori.shape[0]
        self.action_size = gnn_env.action_space
        self.policy = gnn_env.policy

    def step(self, action, gnn_env, past_res):
        action = torch.tensor(action)
        reward, cur_res, clu_res = gnn_env.evaluate(action, past_res)
        return reward, cur_res, clu_res



class Solver:
    """ 多臂老虎机算法基本框架 """

    def __init__(self, bandit, init_action, policy="", action_size=""):
        set_seed()
        self.bandit = bandit
        bandit.policy = policy
        self.bandit.action_size = action_size
        self.policy = bandit.policy
        self.counts = np.zeros((self.bandit.num_nodes, self.bandit.action_size))  # 每个节点每根拉杆（聚合邻居个数）的尝试次数

        #以下代码是一个非常巧妙的设计，对于邻居个数的尝试次数而言：相当于把每个节点不可能的邻居个数的次数初始化为一个非常大的值
        #但是对于节点的聚合层数而言，这里表示什么呢？？？========【对于聚合层数而言，每个节点都有可能选择到任意的层数，因此以下代码在这个时候可能是不适用的】
        if self.policy == "breadth":
            row_indices = init_action[:, np.newaxis]
            column_indices = np.arange(self.bandit.action_size)
            self.counts = np.where(column_indices < row_indices, np.zeros((self.bandit.num_nodes, self.bandit.action_size)), 1000)
        else:
            self.counts = np.zeros((self.bandit.num_nodes, self.bandit.action_size))
        self.regret = 0.  # 当前步的累积懊悔,应该记录每一个节点的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作（每一步的动作应当是针对所有节点动作的集合）
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps, gnn_env, past_res, clu_res):
        # 运行一定次数,num_steps为总运行次数
        temp_time = time.time()

        best_clu_res = clu_res
        best_clu_obj = [0]
        consecutive_step = 0
        accumulated_reward_list = []
        accumulated_reward = 0
        for i in range(num_steps):
            temp_model = copy.deepcopy(gnn_env)
            action, reward, cur_res, clu_res = self.run_one_step(gnn_env, past_res)
            print("step: {} action: {} reward: {} clu_res: {}".format(i, action, reward, clu_res))
            self.counts[np.arange(self.bandit.num_nodes), action] += 1
            self.actions.append(action)
            if consecutive_step > 50:
                break
            if clu_res[0] >= best_clu_res[0]:
                best_clu_res = clu_res
                best_model = temp_model
                best_action = action
                consecutive_step = 0
            else:
                consecutive_step += 1
            past_res = cur_res
            accumulated_reward += reward
            accumulated_reward_list.append(accumulated_reward)
        return best_clu_res, best_action, reward, best_model, cur_res

#方案3：上置信界算法（UCB）
class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, init_action, policy="", action_size="", coef=0.5):
        super(UCB, self).__init__(bandit, init_action, policy, action_size)
        set_seed()
        self.policy = bandit.policy
        self.total_count = 0
        # 以下代码是一个非常巧妙的设计，对于邻居个数的尝试次数而言：相当于把每个节点不可能的邻居个数的回报概率置为0
        if self.policy == "breadth":
            row_indices = init_action[:, np.newaxis]
            column_indices = np.arange(self.bandit.action_size)
            self.estimates = np.where(column_indices < row_indices, np.random.rand(self.bandit.num_nodes, self.bandit.action_size), 0.0)
        self.coef = coef  #用来控制探索的权重

    def run_one_step(self, gnn_env, past_res):
        self.total_count += 1
        #uncertainty = np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))  #计算不确定性度量
        uncertainty = np.sqrt(np.log(self.total_count)/(2*self.counts+1))  #计算不确定性度量
        ucb = self.estimates + self.coef * uncertainty  # 计算上置信界
        #self.coef *= (1-self.total_count/10000)
        #print(self.coef)
        action = np.argmax(ucb, axis=1)  # 选出上置信界最大的拉杆
        reward, cur_res, clu_res = self.bandit.step(action, gnn_env, past_res)
        reward = np.mean(reward)
        #其实就是ucb的预估奖励概率的计算方式，只不过为了提高效率，进行了增量式的计算而已；
        self.estimates[np.arange(self.bandit.num_nodes), action] += 1. / (self.counts[np.arange(self.bandit.num_nodes), action] + 1) * (reward - self.estimates[np.arange(self.bandit.num_nodes), action])
        return action, reward, cur_res, clu_res

