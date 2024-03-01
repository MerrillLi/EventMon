#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 11/12/2023 23:43
# @Author : YuHui Li(MerylLynch)
# @File : runEventMon.py
# @Comment : Created By Liyuhui,23:43
# @Completed : No
# @Tested : No
import sys
sys.path.append('.')
# TODO::这里去做一个在线版本的EventMon，设计一个在线错误报警的更新算法，实现更好的数据恢复精度
import torch as t
import dgl as d
import numpy as np
import argparse
from common.logger import setup_logger
from loguru import logger
from modules.ResNMMF import ResNMMFv2
from dataset.online_dataset import GraphStreamDataset, sparsify, collate_graph
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
from common.utils import Metrics
from modules.stream2batch import stream2batch
from tqdm import *
from collections import deque
import pickle as pkl


def LossFunc(label, pred):
    mse = F.mse_loss(label, pred)
    re = t.mean(t.abs(label - pred) / t.abs(label + pred + 1))
    # loss = a * mse + (1 - a) * (re ** 1.2)
    return mse + 0.1 * re

def get_tensor(args):
    tensor = None
    if args.dataset == 'abilene':
        tensor = np.load('./dataset/abilene.npy').astype('float32')

    if args.dataset == 'geant':
        tensor = np.load('./dataset/geant.npy').astype('float32')

    if args.dataset == 'seattle':
        tensor = np.load('./dataset/seattle.npy').astype('float32')
    return tensor


def mat2graph(mat: np.ndarray, num_nodes):
    srcIdx, dstIdx = mat.nonzero()
    edge_attrs = t.FloatTensor(mat[srcIdx, dstIdx]).reshape(-1, 1)

    graph = d.heterograph(data_dict={
        ('src', 'src2dst', 'dst'): (srcIdx, dstIdx),
        ('dst', 'dst2src', 'src'): (dstIdx, srcIdx)
    }, num_nodes_dict={
        'src': num_nodes,
        'dst': num_nodes}
    )  # type:d.DGLHeteroGraph
    graph.edata['v'] = {
        'src2dst': edge_attrs,
        'dst2src': edge_attrs
    }
    return graph


def construct_input(spNMMs, fuNMMs, masks, num_nodes):
    # Offline Part
    graphs = [ mat2graph(mat, num_nodes) for mat in spNMMs[:-1] ]

    # Online Part
    src, dst = spNMMs[-1].nonzero()
    p = np.random.permutation(len(src))
    src, dst = src[p], dst[p]
    messages = spNMMs[-1][src, dst]
    streams = (src, dst, messages)

    # For Validation
    real_mat = fuNMMs[-1]
    mask = masks[-1]
    streams = stream2batch(streams, max_nodes=num_nodes)
    return graphs, (streams, 1), real_mat, mask



class OnlineSimulator:

    def __init__(self, args):
        self.args = args
        self.tensor = get_tensor(args)

        thsh = np.percentile(self.tensor, args.quantile)
        self.tensor[self.tensor > thsh] = thsh
        self.tensor /= thsh
        self.spNMM, self.fuNMM, self.masks = sparsify(self.tensor, args.density)

        self.window = args.window
        self.dataWindow = 128
        self.currRound = 400
        self.online_size = 250
        self.model = ResNMMFv2(args).to(args.device)

        self.watch_size = 32

        self.record = dict()
        self.NMAEs = []
        self.queue = deque(maxlen=self.watch_size)

        self.preds = np.array([])
        self.reals = np.array([])


    def training(self, finetune):
        trainData = self.spNMM[self.currRound - self.dataWindow:self.currRound]
        validData = self.fuNMM[self.currRound - self.dataWindow:self.currRound]
        trainMask = self.masks[self.currRound - self.dataWindow:self.currRound]
        dataset = GraphStreamDataset(trainData, validData, trainMask, args)
        dataLoader = DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=partial(collate_graph, num_nodes=args.num_nodes))
        # self.model = ResNMMFv2(args)
        self.model.train()
        optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr)
        epochs = args.epochs if not finetune else 30
        for _ in trange(epochs):
            self.model.train()
            losses = []
            for trainBatch in dataLoader:
                ctxStreams, currStreams, testLabel, mask = trainBatch
                pred = self.model.forward(ctxStreams, currStreams, args)
                trainMask = mask == 1
                loss = LossFunc(testLabel[trainMask].to(args.device), pred[trainMask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += [loss.item()]
            # print(f'Epoch = {epoch}, Loss = {np.mean(losses)}')
        # print("Training Finished!")

    def start(self):

        # 1. 准备模型
        self.training(finetune=False)

        # 2. 开始模拟
        for i in range(self.online_size):

            currInputNMMs = self.spNMM[self.currRound - self.window:self.currRound]
            currInputMask = self.masks[self.currRound - self.window:self.currRound]
            currFullNMMs = self.fuNMM[self.currRound - self.window:self.currRound]

            graphs, streams, real_mat, mask = construct_input(currInputNMMs, currFullNMMs, currInputMask, args.num_nodes)

            # 2.1 模型预测
            with t.no_grad():
                pred_mat = self.model.forward(graphs, streams, self.args)
                pred_mat = pred_mat.reshape((args.num_nodes, args.num_nodes)).numpy()
            test_mask = mask == 0
            real_vec, pred_vec = real_mat[test_mask], pred_mat[test_mask]

            self.reals = np.append(self.reals, real_vec)
            self.preds = np.append(self.preds, pred_vec)

            NRMSE, NMAE = Metrics(pred_vec, real_vec)
            self.record[i] = {
                'real': real_vec,
                'pred': pred_vec
            }
            print(f'Round = {self.currRound}, NMAE = {NMAE}, NRMSE = {NRMSE}')

            # 2.2 模型更新控制器
            if len(self.queue) == self.watch_size:
                pop_NMAE = self.queue.popleft()
                self.NMAEs.append(pop_NMAE)
                self.queue.append(NMAE)
            else:
                self.queue.append(NMAE)

            if len(self.queue) == self.watch_size and len(self.NMAEs) >= self.watch_size:
                meanNMAE = np.mean(self.NMAEs)
                recentNMAE = np.mean(list(self.queue))
                if recentNMAE > meanNMAE * 1.2:
                    logger.info(f'Retraining on Round = {self.currRound}')
                    self.training(finetune=True)
                    self.NMAEs.extend(self.queue)
                    self.queue.clear()
            self.currRound += 1


    def summary(self):
        NRMSE, NMAE = Metrics(self.preds, self.reals)
        with open(f'./output/EventMon_OL_{args.dataset}_{args.density}.pkl', 'wb') as f:
            pkl.dump(self.record, f)

        return NRMSE, NMAE

def main(args):
    NRMSEs, NMAEs = [], []
    for roundId in range(args.rounds):
        simulator = OnlineSimulator(args)
        simulator.start()
        NRMSE, NMAE = simulator.summary()
        NRMSEs.append(NRMSE)
        NMAEs.append(NMAE)
        logger.info(f'Round = {roundId}, NRMSE = {NRMSE}, NMAE = {NMAE}')
    logger.info(f'EventMon OL: NRMSE = {np.mean(NRMSEs)}, NMAE = {np.mean(NMAEs)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--num_nodes', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--msg_dim', type=int, default=1)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--quantile', type=int, default=99)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--return_time', type=bool, default=False)
    parser.add_argument('--intermedia', type=bool, default=False)
    parser.add_argument('--timer', type=bool, default=False)
    args = parser.parse_args()

    for density in [0.25]:
        args.density = density
        setup_logger(args, f"./output/EventMon_OL_{args.dataset}_{args.density}")
        logger.info(args)
        main(args)
