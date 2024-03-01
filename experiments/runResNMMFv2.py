import sys
sys.path.append('.')
import torch as t
import numpy as np
import logging
import argparse
global logger
from tqdm import *
from modules.ResNMMF import ResNMMFv2
from common.utils import get_loader, LossFunc, Metrics
import pickle as pkl

def run(runid, args):

    # Training Initialization
    trainLoader, validLoader, testLoader, thsh, num_nodes = get_loader(args)
    model = ResNMMFv2(args).to(args.device)
    optimizer = t.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = t.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    # Training
    for epoch in trange(args.epochs):
        model.train()
        losses = []
        # if epoch > 10:
        #     optimizer = t.optim.AdamW(model.parameters(), lr=2e-3)

        for trainBatch in trainLoader:
            ctxStreams, currStreams, testLabel, mask = trainBatch
            pred = model.forward(ctxStreams, currStreams, args)
            trainMask = mask == 1
            loss = LossFunc(testLabel[trainMask].to(args.device), pred[trainMask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
        # scheduler.step()

        # Evaluation
        model.eval()

        labels = []
        preds = []

        per_slice = dict()
        counter = 0
        with t.no_grad():
            for i, testBatch in enumerate(testLoader):
                try:
                    ctxStreams, currStreams, testLabel, mask = testBatch
                    pred_mat = model.forward(ctxStreams, currStreams, args)
                    testMask = mask != 1
                    labels += (testLabel[testMask]).flatten().numpy().tolist()
                    preds += (pred_mat[testMask]).flatten().cpu().numpy().tolist()

                    # for id in range(len(pred_mat)):
                    #     sliceMask = testMask[id]
                    #     sliceLabel = testLabel[id]
                    #     slicePred = pred_mat[id]
                    #     per_slice[counter] = {
                    #         'real': (sliceLabel[sliceMask]).flatten().numpy(),
                    #         'pred': (slicePred[sliceMask]).flatten().cpu().numpy()
                    #     }
                    #     counter += 1
                except Exception as ex:
                    print(ex)
                    pass

            labels = np.array(labels)
            preds = np.array(preds)
            ER, NMAE = Metrics(labels * thsh, preds * thsh)
            print(f'Epoch {epoch}, Loss={np.mean(losses):.6f}, ER={ER:.3f}, NMAE={NMAE:.3f}')

        # fp = open(f'{args.dataset}_perslice.pkl', 'wb')
        # pkl.dump(per_slice, fp)
        # fp.close()
    return ER, NMAE


def main(args):
    RunERs, RunNMAEs = [], []
    for runid in range(args.rounds):
        ERs, NMAEs = run(runid, args)
        RunERs += [ERs]
        RunNMAEs += [NMAEs]

    RunERs = np.array(RunERs)
    RunNMAEs = np.array(RunNMAEs)

    # for i in range(4):
    #     logger.info(f'Data Input Rate={(i+1)*25}%, Run ER={np.mean(RunERs[:,i]):.3f}, Run NAME={np.mean(RunNMAEs[:,i]):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--density', type=float, default=0.05)
    parser.add_argument('--num_nodes', type=int, default=12)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--msg_dim', type=int, default=1)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--window', type=int, default=6)
    parser.add_argument('--quantile', type=int, default=99)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--return_time', type=bool, default=False)
    parser.add_argument('--intermedia', type=bool, default=False)
    parser.add_argument('--timer', type=bool, default=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, filename=f'results/ResNMMF/RTNet-Inter_{args.dataset}_{args.density}.log', filemode='w')
    logger = logging.getLogger('RTNet')
    logger.info(f'Experiment Config = {args}')
    main(args)
