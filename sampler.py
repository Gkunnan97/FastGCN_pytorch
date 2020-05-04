import pdb
import torch
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import norm as sparse_norm
from utils import prepare_pubmed, sparse_mx_to_torch_sparse_tensor

def get_batches(train_ind, train_labels, batch_size=64, shuffle=True):
    """
    Inputs:
        train_ind: np.array 
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind =  train_ind[i:i + batch_size] 
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Sampler:
    def __init__(self, features, adj, **kwargs):
        allowed_kwargs = {'num_layers', 'input_dim', 'layer_sizes', 'scope'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        self.input_dim = kwargs.get('input_dim', 1)
        self.layer_sizes = kwargs.get('layer_sizes', [1])
        self.scope = kwargs.get('scope', 'test_graph')

        self.num_layers = len(self.layer_sizes)

        self.adj = adj
        self.x = features

        self.train_nodes_number = self.adj.shape[0]

    def one_layer_sampling(self, v_indices, output_size, layer_num):
        raise NotImplementedError("one_layer_sampling is not implimented")

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
        all_p_u = [[]] * (self.num_layers - 1)
        all_x_u = [[]] * self.num_layers

        # sample top-1 layer
        all_x_u[self.num_layers - 1] = self.x[v] 
        cur_out_nodes = v
        # top-down sampling from top-2 layer to the input layer

        u_sampled, support, p_u = self.one_layer_sampling(cur_out_nodes,
                                                            output_size=self.layer_sizes[0], layer_num=0)

        all_support = [self.adj[u_sampled, :] for _ in range(self.num_layers - 2)]
        all_support.append(support)
        all_p_u = [p_u for _ in range(self.num_layers - 1)]
        all_x_u[:-1] = [self.x for _ in range(self.num_layers - 1)]

        # pdb.set_trace()
        all_x_u = self._change_dense_to_tensor(all_x_u)
        all_support = self._change_sparse_to_tensor(all_support)
        # pdb.set_trace()
        return all_x_u, all_support        

    def one_layer_sampling(self, v_indices, output_size, layer_num):
        # pdb.set_trace()
        support = self.adj[v_indices, :]
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        # p1 = p1 / np.sum(p1)
        sampled = np.random.choice(np.array(np.arange(np.size(neis))), output_size, False, p1 / np.sum(p1))
        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]
        # pdb.set_trace()
        # support = support.todense()
        # re-normalize
        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support, None
    
    def update_distribution(self, outputs, *args):
        pass