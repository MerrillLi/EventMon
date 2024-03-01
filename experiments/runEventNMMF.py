import sys
sys.path.append('.')
import torch as t
import numpy as np
import logging
import argparse
global logger
from tqdm import *
from modules.exp_model import EventNMMF
from common.utils import LossFunc, Metrics, EarlyStopMonitor
from torch.utils.data import DataLoader, Dataset
from functools import partial
from modules.stream2batch import ReBatch, mats_to_streams


def to_events(mat):
    src, dst = mat.nonzero()
    p = np.random.permutation(len(src))
    src, dst = src[p], dst[p]
    messages = mat[src, dst]
    streams = (src, dst, messages)
    return streams


def sparsify(tensor, train_ratio, valid_ratio):

    timeIdx, srcIdx, dstIdx = tensor.nonzero()
    p = np.random.permutation(len(timeIdx))
    timeIdx, srcIdx, dstIdx = timeIdx[p], srcIdx[p], dstIdx[p]


    # Sparse Data
    sparseData = np.zeros_like(tensor)
    size = int(np.prod(tensor.shape) * train_ratio)
    tIdx, rIdx, cIdx = timeIdx[:size], srcIdx[:size], dstIdx[:size]
    sparseData[tIdx, rIdx, cIdx] = tensor[tIdx, rIdx, cIdx]

    # Valid Data
    validData = np.zeros_like(tensor)
    vsize = int(np.prod(tensor.shape) * valid_ratio)
    tIdx, rIdx, cIdx = timeIdx[size:size+vsize], srcIdx[size:size+vsize], dstIdx[size:size+vsize]
    validData[tIdx, rIdx, cIdx] = tensor[tIdx, rIdx, cIdx]

    # Test Data
    testData = np.zeros_like(tensor)
    tIdx, rIdx, cIdx = timeIdx[size+vsize:], srcIdx[size+vsize:], dstIdx[size+vsize:]
    testData[tIdx, rIdx, cIdx] = tensor[tIdx, rIdx, cIdx]

    return sparseData, validData, testData


class EventNMMDataset(Dataset):

    def __init__(self, dataTensor, args):
        self.window = args.window
        self.density = args.density
        self.spNMM, self.validNMM, self.testNMM = sparsify(dataTensor, args.density, 0.1)

    def __len__(self):
        return len(self.spNMM) - self.window - 1

    def __getitem__(self, idx):
        ctxNMMs = self.spNMM[idx:idx + self.window]
        assert len(ctxNMMs) == self.window
        currNMM = self.spNMM[idx + self.window]
        trainNMM = self.spNMM[idx + self.window]
        validNMM = self.validNMM[idx + self.window]
        testNMM = self.testNMM[idx + self.window]
        ctxStreams = ctxNMMs
        currStream = to_events(currNMM)
        return ctxStreams, currStream, trainNMM, validNMM, testNMM


def collate_fn(samples, num_nodes):
    bs = len(samples)
    ctxStreams = []
    currSteam = []
    trainMats = []
    validMats = []
    testMats = []

    for sample in samples:
        s_ctx_streams, s_curr_stream, train_mat, valid_mat, test_mat = sample
        ctxStreams.append(t.from_numpy(s_ctx_streams))
        currSteam += [s_curr_stream]
        trainMats.append(t.from_numpy(train_mat))
        validMats.append(t.from_numpy(valid_mat))
        testMats.append(t.from_numpy(test_mat))

    ctxStreams = t.stack(ctxStreams)
    ctxStreams = mats_to_streams(ctxStreams, num_nodes)
    trainMats = t.stack(trainMats)
    validMats = t.stack(validMats)
    testMats = t.stack(testMats)

    currSteam = ReBatch(currSteam, num_nodes)
    return ctxStreams, (currSteam, bs), trainMats, validMats, testMats



def get_loader(args):
    data = np.load(f'./dataset/{args.dataset}.npy')[:args.num_times].astype('float32')

    quantile = {
        'abilene': 99,
        'geant': 99,
        'seattle': 99,
        'taxi': 100
    }

    thsh = np.percentile(data, q=quantile[args.dataset])
    data[data > thsh] = thsh
    data /= thsh
    trainTensor = data[:args.offset]
    testTensor = data[args.offset:]

    collator = partial(collate_fn, num_nodes=data.shape[-1])
    train_valid_set = EventNMMDataset(trainTensor, args)
    testset = EventNMMDataset(testTensor, args)
    trainLoader = DataLoader(train_valid_set, batch_size=args.bs, shuffle=True, collate_fn=collator)
    validLoader = DataLoader(train_valid_set, batch_size=args.bs, collate_fn=collator)
    testLoader = DataLoader(testset, batch_size=args.bs, collate_fn=collator)
    return trainLoader, validLoader, testLoader


def model_training(model, optimizer, trainLoader, criterion):
    losses = []
    model.train()
    for trainBatch in trainLoader:
        ctxStreams, currStream, trainNMM, validNMM, testNMM = trainBatch
        pred = model.forward(ctxStreams, currStream, args)
        trainMask = trainNMM != 0
        loss = criterion(trainNMM[trainMask].to(args.device), pred[trainMask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += [loss.item()]

    return np.mean(losses)


def model_validation(model, validLoader):
    model.eval()
    labels = []
    preds = []
    t.set_grad_enabled(False)
    for validBatch in validLoader:
        ctxStreams, currStream, trainNMM, validNMM, testNMM = validBatch
        pred_mat = model.forward(ctxStreams, currStream, args)
        testMask = validNMM != 0
        labels += (validNMM[testMask]).flatten().numpy().tolist()
        preds += (pred_mat[testMask]).flatten().cpu().numpy().tolist()

    labels = np.array(labels)
    preds = np.array(preds)
    ER, NMAE = Metrics(labels, preds)
    t.set_grad_enabled(True)
    return ER, NMAE


def model_testing(model, testLoader):
    labels = []
    preds = []
    t.set_grad_enabled(False)
    for testBatch in testLoader:
        ctxStreams, currStream, trainNMM, validNMM, testNMM = testBatch
        pred_mat = model.forward(ctxStreams, currStream, args)
        testMask = testNMM != 0
        labels += (testNMM[testMask]).flatten().numpy().tolist()
        preds += (pred_mat[testMask]).flatten().cpu().numpy().tolist()
    labels = np.array(labels)
    preds = np.array(preds)
    ER, NMAE = Metrics(labels, preds)
    t.set_grad_enabled(True)
    return ER, NMAE


def run(runid, args):

    # Training Initialization
    trainLoader, validLoader, testLoader = get_loader(args)
    model = EventNMMF(args).to(args.device)
    optimizer = t.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = t.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    criterion = LossFunc
    monitor = EarlyStopMonitor(5)

    for epoch in trange(args.epochs):
        loss = model_training(model, optimizer, trainLoader, criterion)
        vNRMSE, vNMAE = model_validation(model, validLoader)
        # scheduler.step()

        print(f'Epoch {epoch}, Loss={loss}, vNRMSE={vNRMSE:.3f}, vNMAE={vNMAE:.3f}')

        monitor.track(epoch, model, vNRMSE)
        if monitor.early_stop():
            break

        best_model = monitor.model
        tNRMSE, tNMAE = model_testing(best_model, testLoader)
        print(f'tNRMSE={tNRMSE:.3f}, tNMAE={tNMAE:.3f}')

    best_model = monitor.model
    tNRMSE, tNMAE = model_testing(best_model, testLoader)
    print(f'tNRMSE={tNRMSE:.3f}, tNMAE={tNMAE:.3f}')
    return tNRMSE, tNMAE


def main(args):
    RunERs, RunNMAEs = [], []
    for runid in range(args.rounds):
        ERs, NMAEs = run(runid, args)
        RunERs += [ERs]
        RunNMAEs += [NMAEs]

    RunERs = np.array(RunERs)
    RunNMAEs = np.array(RunNMAEs)

    for i in range(4):
        logger.info(f'Data Input Rate={(i+1)*25}%, Run ER={np.mean(RunERs[:,i]):.3f}, Run NAME={np.mean(RunNMAEs[:,i]):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--num_nodes', type=int, default=12)
    parser.add_argument('--num_times', type=int, default=4000)
    parser.add_argument('--offset', type=int, default=3000)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--msg_dim', type=int, default=1)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--return_time', type=bool, default=False)
    parser.add_argument('--intermedia', type=bool, default=False)
    parser.add_argument('--timer', type=bool, default=False)



    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, filename=f'results/ResNMMF/RTNet-Inter_{args.dataset}_{args.density}.log', filemode='w')
    logger = logging.getLogger('RTNet')
    logger.info(f'Experiment Config = {args}')
    main(args)
