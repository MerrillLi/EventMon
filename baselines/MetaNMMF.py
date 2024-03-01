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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class TempAttn(Module):

    def __init__(self, dim):
        super(TempAttn, self).__init__()
        self.query = Sequential(Linear(dim, dim), Tanh())
        self.value = Sequential(Linear(dim, dim), Tanh())
        # self.value = Identity()
        self.act = Softmax(dim=-1)

    def forward(self, hiddens, enc_outputs):
        # x [bs, steps, dim]
        query = self.query(enc_outputs)
        key = self.value(hiddens)
        weight = t.einsum('bd, bkd->bk', query, key)
        weight = weight.unsqueeze(-1)
        weight = self.act(weight)
        value = (weight * key).mean(1)
        return value


class fMetaNMMF(Module):

    def __init__(self, args):
        super(fMetaNMMF, self).__init__()
        self.window = args.windows
        self.hidden = args.embed_dim
        in_size = 144
        hidden = args.embed_dim
        self.encoder = Sequential(
            Linear(in_size, hidden),
            LayerNorm(hidden),
            PReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            PReLU(),
            Linear(hidden, hidden)
        )

        self.decoder = Sequential(
            Linear(hidden, hidden),
            LayerNorm(hidden),
            PReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            PReLU(),
            Linear(hidden, in_size)
        )
        self.norm = LayerNorm(hidden)
        self.rnn = GRUCell(hidden, hidden)
        self.attn = TempAttn(hidden)
        self.context_proj = Sequential(LazyLinear(64), ReLU(), LazyLinear(32))
        self.current_proj = Sequential(LazyLinear(64), ReLU(), LazyLinear(32))


    def encode(self, target_x):
        bs, step, _, _ = target_x.shape
        target_x = target_x.reshape((bs, step, -1))

        if self.training:
            target_x = t.dropout(target_x, p=0.1, train=self.training)

        ctx_features = [self.encoder(target_x[:, i, :]) for i in range(target_x.shape[1])]
        ctx_vec = t.zeros((target_x.shape[0], self.hidden))
        for i in range(len(ctx_features)):
            ctx_vec = self.rnn.forward(ctx_features[i], ctx_vec)

        return ctx_vec


    def forward(self, contextNMMs, currentNMM, negativeNMMs=None):
        bs, step, _, _ = contextNMMs.shape
        contextNMMs = contextNMMs.reshape((bs, step, -1))

        if self.training:
            contextNMMs = t.dropout(contextNMMs, p=0.1, train=self.training)

        # Step 2: Encoding Spatial Information
        ctx_features = [self.encoder(contextNMMs[:, i, :]) for i in range(contextNMMs.shape[1])]
        if self.training:
            currentNMM = t.dropout(currentNMM, p=0.1, train=self.training)
        curr_feature = self.encoder(currentNMM.reshape((bs, -1)))

        # Step 3: Encoding Temporal Information
        ctx_vec = t.zeros((contextNMMs.shape[0], self.hidden))
        for i in range(len(ctx_features)):
            ctx_vec = self.rnn.forward(ctx_features[i], ctx_vec)

        ctx_vec = self.attn.forward(t.stack(ctx_features, dim=1), curr_feature)

        # Step 4: Recovery Full Matrix
        dec_input = ctx_vec + curr_feature
        dec_input = self.norm(dec_input)
        pred = self.decoder(dec_input)
        pred = pred.reshape(currentNMM.shape)

        neg_loss = 0
        if negativeNMMs is not None:
            neg_mats = t.chunk(negativeNMMs, self.window, dim=1)[:2]
            neg_features = t.stack([self.encoder(mat.reshape(bs, -1)) for mat in neg_mats])
            neg_features = self.current_proj(neg_features)
            current = self.current_proj(curr_feature)
            context = self.context_proj(ctx_vec)

            pos = t.einsum('bk, bk->b', context, current)
            pos = t.exp(pos) / 2

            neg = t.einsum('bk, nbk-> nb', context, neg_features)
            neg = t.exp(neg) / 2
            neg_loss = - t.log(pos / (pos + t.sum(neg, dim=0))).mean()

        return pred.sigmoid(), neg_loss


def plot(encode):
    encode = TSNE().fit_transform(encode)
    plt.scatter(encode[:, 0], encode[:, 1])
    plt.title('Latent Space Distribution!')
    plt.tight_layout()
    plt.show()


def run(runid, args):

    model = fMetaNMMF(args)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    trainLoader, testLoader = get_nmm_loaders(args)
    ER, NMAE = 0, 0
    for epoch in trange(args.epochs):
        losses = []
        for trainBatch in trainLoader:
            ctxNMMs, currNMM, fullNMM, negNMMs, currMask = trainBatch
            pred, neg_loss = model.forward(ctxNMMs, currNMM, negNMMs)
            loss = composite_loss(pred * currMask, fullNMM * currMask, epoch, args.epochs) + 0.001 * neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
        avg_loss = np.mean(losses)

        torch.set_grad_enabled(False)
        reals, preds = [], []
        neg_losses = []
        for testBatch in tqdm(testLoader):
            ctxNMMs, currNMM, fullNMM, negNMMs, currMask = testBatch
            pred, neg_loss = model.forward(ctxNMMs, currNMM, negNMMs)
            nonzeroIdx = (1 - currMask) == 1
            mVal = fullNMM[nonzeroIdx]
            pred = pred[nonzeroIdx]
            reals += mVal.numpy().tolist()
            preds += pred.numpy().tolist()
            neg_losses += [neg_loss.item()]
        reals = np.array(reals)
        preds = np.array(preds)
        ER, NMAE = ErrMetrics(preds, reals)
        torch.set_grad_enabled(True)
        avg_neg = np.mean(neg_losses)
        print(f'MetaNMMF, Epoch={epoch}, Loss={avg_loss} ER={ER:.3f}, NMAE={NMAE:.3f}, NegLoss={avg_neg:.3f}')

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
    parser.add_argument('--windows', type=int, default=6)
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args()
    ts = time.asctime()
    main(args)
