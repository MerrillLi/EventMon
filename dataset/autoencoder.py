import torch as t
from torch.nn import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Dataset import NMMDataset

class AutoEncoder(Module):

    def __init__(self, in_size, hidden_size):
        super(AutoEncoder, self).__init__()

        self.enc = Sequential(
            Linear(in_size, 128),
            LayerNorm(128),
            Linear(128, hidden_size),
            Sigmoid()
        )

        self.dec = Sequential(
            Linear(hidden_size, 128),
            LayerNorm(128),
            Linear(128, in_size)
        )

    def forward(self, x):
        enc = self.enc(x)
        y = self.dec(enc)
        return y


def get_loader(args):
    collate = collate_graph
    data = get_tensor(args)

    trainTensor = None
    testTensor = None
    if args.dataset == 'abilene':
        data = data[:4000]
        thsh = np.percentile(data, q=args.quantile)
        data[data > thsh] = thsh
        data /= thsh
        trainTensor = data[:3000]
        testTensor = data[3000:4000]

    if args.dataset == 'geant':
        data = data[:3500]
        thsh = np.percentile(data, q=args.quantile)
        data[data > thsh] = thsh
        data /= thsh
        trainTensor = data[:3000]
        testTensor = data[3000:3500]

    if args.dataset == 'seattle':
        data = data[:650]
        thsh = np.percentile(data, q=args.quantile)
        data[data > thsh] = thsh
        data /= thsh
        trainTensor = data[:400]
        testTensor = data[400:650]

    if args.dataset == 'harvard':
        data = data[:800]
        thsh = np.percentile(data, q=args.quantile)
        data[data > thsh] = thsh
        data /= thsh
        trainTensor = data[:600]
        testTensor = data[600:800]

    if args.dataset == 'taxi':
        data = data[:1464]
        thsh = np.percentile(data, q=args.quantile)
        data[data > thsh] = thsh
        data /= thsh
        trainTensor = data[:960]
        testTensor = data[960:]

    num_nodes = data.shape[-1]

    trainLoader = DataLoader(NMM(trainTensor, args), batch_size=args.bs,
                             shuffle=True, collate_fn=partial(collate, num_nodes=num_nodes),
                             num_workers=2, persistent_workers=True)

    validLoader = DataLoader(GraphStreamDataset(trainTensor, args), batch_size=args.bs,
                             shuffle=True, collate_fn=partial(collate, num_nodes=num_nodes))

    testLoader = DataLoader(GraphStreamDataset(testTensor, args), batch_size=args.bs,
                            shuffle=True, collate_fn=partial(collate, num_nodes=num_nodes),
                            num_workers=2, persistent_workers=True)

    return trainLoader, validLoader, testLoader, thsh, num_nodes
