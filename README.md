# Pytorch Implementation of FastGCN and AS-GCN
PyTorch implementation of [FastGCN](https://arxiv.org/abs/1801.10247) and [AS-GCN](http://papers.nips.cc/paper/7707-adaptive-sampling-towards-fast-graph-representation-learning). The supported datasets are: cora, citeseer and pubmed. 
Mind that this implementation may differ from the original in some parts. Especially for the AS-GCN, the different methods of calculating variance did not bring better performance. So if you want to use it into the research, please cheak these details carefully.
## Requirements
    * PyTorch 1.14
    * Python 3.7

## Usage
    python train.py --dataset dataset_name --model model_name

## Reference
    FASTGCN: FAST LEARNING WITH GRAPH CONVOLUTIONAL NETWORKS VIA IMPORTANCE SAMPLING
    Adaptive Sampling Towards Fast Graph Representation Learning