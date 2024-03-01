import dgl as d
import numpy as np
from torch.nn import functional as F
import torch as t
from modules.stream2batch import ReBatch
from functools import partial
from dataset.Dataset import *
from torch.utils.data import DataLoader
import copy


def Metrics(true, pred):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE


def LossFunc(label, pred):
    mse = F.mse_loss(label, pred)
    re = t.mean(t.abs(label - pred) / t.abs(label + pred + 1))
    # loss = a * mse + (1 - a) * (re ** 1.2)
    return mse + 0.1 * re


def get_tensor(args):
    tensor = None
    if args.dataset == 'abilene':
        tensor = np.load('./dataset/abilene.npy')[:4000].astype('float32')

    if args.dataset == 'geant':
        tensor = np.load('./dataset/geant.npy')[:3500].astype('float32')

    if args.dataset == 'seattle':
        tensor = np.load('./dataset/seattle.npy')[:650].astype('float32')

    if args.dataset == 'taxi':
        tensor = np.load('./dataset/taxi.npy')[:1464].astype('float32')

    return tensor


def collate_stream(data_list, num_nodes):
    batch_size = len(data_list)
    num_wins = len(data_list[0][0])
    win_streams = [[] for _ in range(num_wins)]
    curr_streams = []
    real_mats = []
    masks = []

    for ctx_stream, curr_stream, real_mat, mask in data_list:
        for i, stream in enumerate(ctx_stream):
            win_streams[i].append(stream)
        curr_streams += curr_stream
        real_mats += [t.FloatTensor(real_mat)]
        masks += [t.FloatTensor(mask)]

    real_mats = t.stack(real_mats)
    masks = t.stack(masks)

    ctxStreams = [ReBatch(stream, num_nodes) for stream in win_streams]
    currStreams = ReBatch(curr_streams, num_nodes)
    return (ctxStreams, batch_size), (currStreams, batch_size), real_mats, masks


def get_loader(args):
    collate = collate_mat
    data = get_tensor(args)

    trainTensor = None
    testTensor = None
    if args.dataset == 'abilene':
        data = data[:4000]
        thsh = np.percentile(data, q=args.quantile)
        data[data > thsh] = thsh
        data /= thsh
        trainTensor = data[:3000]
        testTensor = data[3000 - args.window:4000]

    if args.dataset == 'geant':
        data = data[:3500]
        thsh = np.percentile(data, q=args.quantile)
        data[data > thsh] = thsh
        data /= thsh
        trainTensor = data[:3000]
        testTensor = data[3000 - args.window:3500]

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

    trainLoader = DataLoader(GraphStreamDataset(trainTensor, args), batch_size=args.bs,
                             shuffle=True, collate_fn=partial(collate_graph, num_nodes=num_nodes))

    validLoader = DataLoader(GraphStreamDataset(trainTensor, args), batch_size=args.bs,
                             shuffle=True, collate_fn=partial(collate_graph, num_nodes=num_nodes))

    # args.density *= 0.75
    testLoader = DataLoader(GraphStreamDataset(testTensor, args), batch_size=args.bs,
                            collate_fn=partial(collate_graph, num_nodes=num_nodes))

    # args.density /= 0.75
    return trainLoader, validLoader, testLoader, thsh, num_nodes


def collate_graph(data_list, num_nodes):
    batch_size = len(data_list)
    num_wins = len(data_list[0][0])
    win_graphs = [[] for _ in range(num_wins)]
    streams = []
    real_mats = []
    masks = []

    for graphs, stream, real_mat, mask in data_list:
        for i, winGraph in enumerate(graphs):
            win_graphs[i].append(winGraph)
        streams += [stream]
        real_mats += [t.FloatTensor(real_mat)]
        masks += [t.FloatTensor(mask)]

    win_graphs = [d.batch(graphs) for graphs in win_graphs]
    real_mats = t.stack(real_mats)
    masks = t.stack(masks)
    streams = ReBatch(streams, num_nodes)
    return win_graphs, (streams, batch_size), real_mats, masks


def collate_mat(data_list, num_nodes):
    batch_size = len(data_list)
    win_mats = []
    streams = []
    real_mats = []
    masks = []

    for mats, stream, real_mat, mask in data_list:
        win_mats.append(t.tensor(mats))
        streams += [stream]
        real_mats += [t.FloatTensor(real_mat)]
        masks += [t.FloatTensor(mask)]

    win_mats = t.stack(win_mats)
    real_mats = t.stack(real_mats)
    masks = t.stack(masks)
    streams = ReBatch(streams, num_nodes)
    return win_mats, (streams, batch_size), real_mats, masks


class SequentialDataset:

    def __init__(self, dense, windows, density):

        self.dense = dense
        quantile = np.percentile(self.dense, q=99)
        self.dense[self.dense > quantile] = quantile
        self.dense /= quantile
        self.start = 0
        self.windows = windows
        self.density = density
        self.mask = np.random.rand(*dense.shape).astype('float32')
        self.mask[self.mask > self.density] = 1
        self.mask[self.mask < self.density] = 0

    def move_next(self):
        self.start += self.windows
        # self.start += 1
        return self.start < self.dense.shape[0]

    def reset(self):
        self.start = -self.windows

    def get_loaders(self):
        curr_tensor = self.dense[self.start:self.start + self.windows]
        curr_mask = self.mask[self.start:self.start + self.windows]
        trainTensor = curr_tensor * (1 - curr_mask)
        testTensor = curr_tensor * curr_mask
        trainset = TensorDataset(trainTensor)
        testset = TensorDataset(testTensor)
        trainLoader = DataLoader(trainset, batch_size=128, shuffle=True)
        testLoader = DataLoader(testset, batch_size=1024)
        return trainLoader, testLoader


class EarlyStopMonitor:

    def __init__(self, patient):
        self.model = None
        self.patient = patient
        self.counter = 0
        self.val = 1e10
        self.epoch = -1

    def early_stop(self):
        return self.counter >= self.patient

    def track(self, epoch, model, val):
        if val < self.val:
            self.model = copy.deepcopy(model)
            self.epoch = epoch
            self.val = val
            self.counter = 0
        else:
            self.counter += 1
