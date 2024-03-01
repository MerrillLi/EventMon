import torch as t
from torch.nn import *
from torch.nn import functional as F
import sys
sys.path.append('.')
import numpy as np
import logging
import math

import argparse
import time
import torch
from baselines.utils import get_nmm_loaders, ErrMetrics
from tqdm import *
global logger

def compute_output_size(in_size, padding, kernel=2, stride=1):
    out_size = (in_size - kernel + 2 * padding) // stride + 1
    return out_size


def abilene_block():
    enc_layers = Sequential(
        Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Flatten()
    )

    dec_layers = Sequential(
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        LeakyReLU(),
        ConvTranspose2d(4, 1, kernel_size=2, stride=2)
    )
    hsize = 3 * 3 * 4
    return enc_layers, dec_layers, hsize


def geant_block():
    enc_layers = Sequential(
        Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Flatten()
    )

    dec_layers = Sequential(
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        LeakyReLU(),
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        LeakyReLU(),
        Conv2d(4, 1, 2)
    )
    hsize = 6 * 6 * 4
    return enc_layers, dec_layers, hsize


def taxi_block():
    enc_layers = Sequential(
        Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Flatten()
    )

    dec_layers = Sequential(
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        LeakyReLU(),
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        Conv2d(4, 1, kernel_size=3)
    )
    hsize = 8 * 8 * 4
    return enc_layers, dec_layers, hsize


def seattle_block():
    enc_layers = Sequential(
        Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
        LeakyReLU(),
        Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
        Flatten()
    )

    dec_layers = Sequential(
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        Conv2d(4, 4, 2),
        LeakyReLU(),
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        LeakyReLU(),
        ConvTranspose2d(4, 4, kernel_size=2, stride=2),
        LeakyReLU(),
        Conv2d(4, 1, 2)
    )
    hsize = 13 * 13 * 4
    return enc_layers, dec_layers, hsize



class NMMF_Stream(Module):

    def __init__(self, args):
        super(NMMF_Stream, self).__init__()
        self.encoder, self.decoder, self.dim = abilene_block()

        self.rnn = GRUCell(self.dim, self.dim)
        self.ctx_proj = Sequential(LazyLinear(64), ReLU(), LazyLinear(32))
        self.cur_proj = Sequential(LazyLinear(64), ReLU(), LazyLinear(32))

    def forward(self, ctx_mats, neg_mats, curr_mat, neg=True):
        bs, win = ctx_mats.shape[:2]
        ctx_mats = t.chunk(ctx_mats, win, dim=1)
        mat_feats = [self.encoder(mat) for mat in ctx_mats]

        context = None
        for mat in mat_feats:
            context = self.rnn.forward(mat, context)

        current = self.encoder(curr_mat.unsqueeze(1))

        row_size = int(math.sqrt(current.numel() // bs // 4))
        dec_input = (current + context).reshape(bs, 4, row_size, row_size)
        recover = self.decoder(dec_input).sigmoid()

        if neg:
            neg_mats = t.chunk(neg_mats, win, dim=1)
            neg_feats = t.stack([self.encoder(mat) for mat in neg_mats])
            neg_feats = self.cur_proj(neg_feats)
            current = self.cur_proj(current)
            context = self.ctx_proj(context)

            pos = t.einsum('bk, bk->b', context, current)
            pos = t.exp(pos)

            neg = t.einsum('bk, nbk-> nb', context, neg_feats)
            neg = t.exp(neg)

            neg_loss = - t.log(pos / (pos + t.sum(neg, dim=0)))

            return recover.squeeze(), neg_loss.mean()

        return recover.squeeze()


def run(runid, args):

    model = NMMF_Stream(args)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    trainLoader, testLoader = get_nmm_loaders(args)

    for epoch in trange(args.epochs):
        losses = []

        for trainBatch in trainLoader:
            ctxNMMs, currNMM, fullNMM, negNMMs, currMask = trainBatch
            pred, neg_loss = model.forward(ctxNMMs, negNMMs, currNMM)
            loss = composite_loss(pred * currMask,
                                  fullNMM * currMask,
                                  epoch, args.epochs) + 0.01 * neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += [loss.item()]

        avg_loss = np.mean(losses)
        print(f'{args.model}, Epoch={epoch}, Loss={avg_loss}')
        # if avg_loss < past_loss:
        #     anchor = epoch
        #     past_loss = avg_loss
        # else:
        #     if epoch - anchor >= 3:
        #         break

        torch.set_grad_enabled(False)
        reals, preds = [], []
        for testBatch in tqdm(testLoader):
            ctxNMMs, currNMM, fullNMM, negNMMs, currMask = testBatch
            pred = model.forward(ctxNMMs, negNMMs, currNMM, False)
            nonzeroIdx = (1 - currMask) == 1
            mVal = fullNMM[nonzeroIdx]
            pred = pred[nonzeroIdx]
            reals += mVal.numpy().tolist()
            preds += pred.numpy().tolist()
        reals = np.array(reals)
        preds = np.array(preds)
        ER, NMAE = ErrMetrics(reals, preds)
        torch.set_grad_enabled(True)
        logger.info(f'Round ID={runid}, ER={ER:.3f}, NMAE={NMAE:.3f}')
        print(f'Round ID={runid}, ER={ER:.3f}, NMAE={NMAE:.3f}')

    return ER, NMAE


def main(args):
    RunERs, RunNMAEs = [], []
    for runid in range(args.rounds):
        ER, NMAE = run(runid, args)
        RunERs += [ER]
        RunNMAEs += [NMAE]

    logger.info(f'Run ER={np.mean(RunERs):.3f}, Run NAME={np.mean(RunNMAEs):.3f}')


def composite_loss(true, pred, epoch, emax):
    alpha = (1 - epoch / emax) ** 2
    mse = F.mse_loss(pred, true)
    rre = t.abs(pred - true) / t.abs(pred + true + 1)
    rre = rre.mean()
    composite = alpha * mse + (1 - alpha) * (rre ** 1.2)
    return composite


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NMMF-Stream')
    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--num_users', type=int, default=12)
    parser.add_argument('--num_items', type=int, default=12)
    parser.add_argument('--num_times', type=int, default=4000)
    parser.add_argument('--windows', type=int, default=12)
    parser.add_argument('--density', type=float, default=0.3)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=1)

    args = parser.parse_args()
    ts = time.asctime()
    logging.basicConfig(level=logging.INFO, filename=f'./baselines/experiments/results/{args.model}/{args.dataset}_{args.density}_{ts}.log', filemode='w')
    logger = logging.getLogger('NMMF-Stream')
    logger.info(f'----------------------------')
    logger.info(f'Params Info={args}')
    main(args)
