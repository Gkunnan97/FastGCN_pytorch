import pdb
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # pdb.set_trace()
        outputs1 = F.relu(self.gc1(x[0], adj[0]))
        # pdb.set_trace()
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs2 = self.gc2(outputs1, adj[1])
        return F.log_softmax(outputs2, dim=1)
