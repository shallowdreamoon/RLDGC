import torch
import torch.nn as nn

class ClusteringLoss(nn.Module):
    def __init__(self, adj, att):
        super(ClusteringLoss, self).__init__()
        self.adj = adj
        self.att = att

    def forward(self, output):
        num_edges = torch.sum(self.adj)
        structure_loss = torch.trace(torch.matmul(torch.matmul(output.t(), self.adj), output))/num_edges
        attribute_loss = torch.trace(torch.matmul(torch.matmul(output.t(), self.att), output))/num_edges
        cluster_size = torch.sum(output, dim=0)
        regularization_loss = torch.norm(cluster_size)/output.shape[0]*torch.sqrt(torch.tensor(output.shape[1]))-1
        clu_loss = -(structure_loss+attribute_loss-regularization_loss)
        return clu_loss, structure_loss, attribute_loss, regularization_loss

class SubClusteringLoss(nn.Module):
    def __init__(self, adj, att):
        super(SubClusteringLoss, self).__init__()
        self.adj = adj
        self.att = att

    def forward(self, output):
        sub_num_edges = torch.sum(self.adj)

        adj_multi_out = torch.matmul(self.adj, output)
        sub_structure_loss = torch.diag(torch.matmul(output, adj_multi_out.t()))/sub_num_edges

        att_multi_out = torch.matmul(self.att, output)
        sub_attribute_loss = torch.diag(torch.matmul(output, att_multi_out.t()))/sub_num_edges

        cluster_size = torch.sum(output, dim=0)
        regularization_loss = torch.norm(cluster_size)/output.shape[0]*torch.sqrt(torch.tensor(output.shape[1]))-1

        clu_loss = -(torch.sum(sub_structure_loss)+torch.sum(sub_attribute_loss)-regularization_loss)
        return clu_loss, -sub_structure_loss, -sub_attribute_loss, -(sub_structure_loss+sub_attribute_loss)



# 示例用法
# adj = torch.rand((3,3))
# att = torch.rand((3,3))
# output = torch.rand((3,2))
#
# cluster_loss = ClusteringLoss(adj, att)
# clu_loss = cluster_loss(output)
# print(clu_loss)
#
# cluster_loss1 = SubClusteringLoss(adj, att)
# clu_loss1 = cluster_loss1(output)
# print(clu_loss1)
