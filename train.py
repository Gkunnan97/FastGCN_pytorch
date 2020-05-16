import argparse
import pdb
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import GCN
from sampler import Sampler_FastGCN, Sampler_ASGCN
from utils import load_data, get_batches, accuracy, sparse_mx_to_torch_sparse_tensor


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='AS',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=1,
                        help='the train epochs between two test')                       
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='Dropout rate (1 - keep probability).')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(model, sampler, train_ind, train_labels, batch_size, train_times):
    model.train()
    for epoch in range(train_times):
        t = time.time()
        for batch_inds, batch_labels in get_batches(train_ind, train_labels, batch_size):
            # change labels format (train_N, Class_N) -> (train_N,)
            batch_labels = torch.LongTensor(batch_labels)
            batch_labels = batch_labels.max(1)[1]

            # pdb.set_trace()
            sampled_feats, sampled_adjs, var_loss = sampler.sampling(batch_inds)
            optimizer.zero_grad()
            output = model(sampled_feats, sampled_adjs)
            # pdb.set_trace()
            loss_train = F.nll_loss(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()

    return loss_train.item(), acc_train.item(), time.time() - t


def test(model, test_adj, test_feats, test_labels, batch_size, epoch):
    t = time.time()
    # change data type to tensor
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj) for cur_adj in test_adj]
    test_feats = [torch.FloatTensor(cur_feats) for cur_feats in test_feats]
    test_labels = torch.LongTensor(test_labels).max(1)[1]

    model.eval()
    outputs = model(test_feats, test_adj)
    loss_test = F.nll_loss(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


if __name__ == '__main__':
    # set superpara and load data
    args = get_args()
    adj, features, adj_train, train_features, y_train, y_test, test_index = load_data(
        args.dataset)

    layer_sizes = [128, 128, args.batchsize]
    input_dim = features.shape[1]
    train_nums = adj_train.shape[0]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init the sampler
    if args.model == 'Fast':
        sampler = Sampler_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  scope="None")
    elif args.model == 'AS':
        sampler = Sampler_ASGCN(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                scope="None")
    else:
        print(f"model name error, no model named {args.model}")
        exit()

    # init model and optimizer
    nclass = y_train.shape[1]
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    test_gap = args.test_gap
    test_adj = [adj, adj[test_index, :]]
    test_feats = [features, features[test_index]]
    test_labels = y_test

    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(
            model, sampler, np.arange(train_nums), y_train, args.batchsize, test_gap)
        test_loss, test_acc, test_time = test(
            model, test_adj, test_feats, test_labels, None, args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, train_times: {train_time:.3f} "
              f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}, test_times: {test_time:.3f}")
