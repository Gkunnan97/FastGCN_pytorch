import pdb
import math
import torch
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import norm as sparse_norm
from torch.nn.parameter import Parameter

from utils import sparse_mx_to_torch_sparse_tensor, load_data


class Sampler:
    def __init__(self, features, adj, **kwargs):
        allowed_kwargs = {'num_layers', 'input_dim', 'layer_sizes', 'scope'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, \
                'Invalid keyword argument: ' + kwarg

        self.input_dim = kwargs.get('input_dim', 1)
        self.layer_sizes = kwargs.get('layer_sizes', [1])
        self.scope = kwargs.get('scope', 'test_graph')

        self.num_layers = len(self.layer_sizes)

        self.adj = adj
        self.features = features

        self.train_nodes_number = self.adj.shape[0]

    def sampling(self, v_indices):
        raise NotImplementedError("sampling is not implimented")

    def _change_sparse_to_tensor(self, adjs):
        new_adjs = []
        for adj in adjs:
            new_adjs.append(sparse_mx_to_torch_sparse_tensor(adj))
        return new_adjs

    def _change_dense_to_tensor(self, features):
        new_feats = []
        for feats in features:
            new_feats.append(torch.FloatTensor(feats))
        return new_feats


class Sampler_FastGCN(Sampler):
    def __init__(self, pre_probs, features, adj, **kwargs):
        super().__init__(features, adj, **kwargs)
        col_norm = sparse_norm(adj, axis=0)
        self.probs = col_norm / np.sum(col_norm)

    def sampling(self, v):
        """
        Inputs:
            v: batch nodes list
        """
        all_support = [[]] * (self.num_layers - 1)
        all_x_u = [[]] * self.num_layers
        all_x_u[self.num_layers - 1] = self.features[v]

        u_sampled, support = self._one_layer_sampling(
            v, output_size=self.layer_sizes[1], layer_num=0)

        all_support = [self.adj[u_sampled, :]
                       for _ in range(self.num_layers - 2)]
        all_support.append(support)
        all_x_u[:-1] = [self.features for _ in range(self.num_layers - 1)]

        all_x_u = self._change_dense_to_tensor(all_x_u)
        all_support = self._change_sparse_to_tensor(all_support)

        return all_x_u, all_support, 0

    def _one_layer_sampling(self, v_indices, output_size, layer_num):
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   output_size, True, p1 / np.sum(p1))

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support


class Sampler_ASGCN(Sampler, torch.nn.Module):
    def __init__(self, pre_probs, features, adj, **kwargs):
        # features = torch.FloatTensor(features)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        super().__init__(features, adj, **kwargs)
        torch.nn.Module.__init__(self)
        # col_norm = sparse_norm(adj, axis=0)
        # self.probs = col_norm / np.sum(col_norm)
        self.feats_dim = features.shape[1]

        # attention weights w1 is also wg
        self.w1 = Parameter(torch.FloatTensor(self.feats_dim, 1))
        self.w2 = Parameter(torch.FloatTensor(self.feats_dim, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w1.size(0))
        self.w1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)

    def sampling(self, v):
        """
        Inputs:
            v: batch nodes list
        """
        v = torch.LongTensor(v)
        all_support = [[]] * (self.num_layers - 1)
        all_p_u = [[]] * (self.num_layers - 1)
        all_x_u = [[]] * self.num_layers

        # sample top-1 layer
        all_x_u[self.num_layers - 1] = self.features[v]
        cur_out_nodes = v
        # top-down sampling from top-2 layer to the input layer
        for i in range(len(all_x_u) - 2, -1, -1):
            u_sampled, support, var_need = \
                self._one_layer_sampling(cur_out_nodes,
                                         output_size=self.layer_sizes[i],
                                         layer_num=0)

            all_x_u[i] = self.features[u_sampled]
            all_support[i] = support
            all_p_u[i] = var_need

            cur_out_nodes = u_sampled

        all_x_u = self._change_dense_to_tensor(all_x_u)
        # all_support = self._change_sparse_to_tensor(all_support)

        loss = self._calc_variance(all_p_u)
        return all_x_u, all_support, loss

    def _calc_variance(self, var_need):
        # NOTE: it's useless in this implementation for the three datasets
        # only calc the variane of the last layer
        u_nodes, p_u = var_need[-1][0], var_need[-1][1]
        p_u = p_u.reshape(-1, 1)
        feature = torch.FloatTensor(self.features[u_nodes])
        means = torch.sum(feature, axis=0)
        feature = feature - means
        var = torch.mean(torch.sum(torch.mul(feature, feature) * p_u, 0))
        return var

    def _one_layer_sampling(self, v_indices, output_size, layer_num):
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        support = support[:, neis]
        # change the sparse support to dense
        support = support.todense()
        support = torch.FloatTensor(support)
        h_v = torch.FloatTensor(self.features[v_indices])
        h_u = torch.FloatTensor(self.features[neis])

        attention = torch.mm(h_v, self.w1) + \
            torch.mm(h_u, self.w2).reshape(1, -1) + 1
        attention = (1.0 / np.size(neis)) * torch.relu(attention)

        p1 = torch.sum(support * attention, 0)
        numpy_p1 = p1.data.numpy()
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   size=output_size,
                                   replace=True,
                                   p=numpy_p1 / np.sum(numpy_p1))

        u_sampled = neis[sampled]
        support = support[:, sampled]
        sampled_p1 = p1[sampled]

        t_diag = torch.diag(1.0 / (sampled_p1 * output_size))
        support = torch.mm(support, t_diag)

        return u_sampled, support, (neis, p1 / torch.sum(p1))


if __name__ == '__main__':
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data("cora")
    batchsize = 256
    layer_sizes = [128, 128, batchsize]
    input_dim = features.shape[1]

    sampler = Sampler_ASGCN(None, train_features, adj_train,
                            input_dim=input_dim,
                            layer_sizes=layer_sizes, scope="None")

    batch_inds = list(range(batchsize))
    sampled_feats, sampled_adjs, var_loss = sampler.sampling(batch_inds)
    pdb.set_trace()
