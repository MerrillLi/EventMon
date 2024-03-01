#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 28/02/2023 23:04
# @Author : YuHui Li(MerylLynch)
# @File : exp_model.py
# @Comment : Created By Liyuhui,23:04
# @Completed : No
# @Tested : No

import torch as t
from torch.nn import *
from modules.rnn import StructuredRNN
from modules.backbone import EventEncoder
from modules.event_encoder import PermInvarEventEncoder
from torch.nn import functional as F

class LNLayer(Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(LNLayer, self).__init__()
        self.lin = Linear(in_feats, out_feats, bias)
        self.act = GELU()

    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
        return x


class NormLayer(Module):
    def __init__(self, in_feats, out_feats):
        super(NormLayer, self).__init__()
        self.linear = Linear(in_feats, out_feats)
        self.norm = LayerNorm(out_feats)
        self.act = ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResMLPMixer(Module):

    def __init__(self, dim, depth):
        super(ResMLPMixer, self).__init__()

        self.hori_mix = LNLayer(dim, dim)
        self.vert_mix = LNLayer(depth, depth)
        self.linear = LNLayer(dim, dim)

    def forward(self, x:t.Tensor):
        res = x
        x = self.hori_mix(x) # type:t.Tensor
        # x = F.relu(x)
        x += res
        x = self.vert_mix(x.permute(0, 2, 1))
        # x = F.relu(x)
        x = x.permute(0, 2, 1)
        x += res
        return x



class EventNMMF(Module):

    def __init__(self, args):
        super(EventNMMF, self).__init__()

        self.num_nodes = args.num_nodes
        self.embed_dim = args.embed_dim

        self.row_recurrent = StructuredRNN(args.embed_dim, args.embed_dim)
        self.col_recurrent = StructuredRNN(args.embed_dim, args.embed_dim)

        self.row_init_hidden = Embedding(self.num_nodes, args.embed_dim)
        self.col_init_hidden = Embedding(self.num_nodes, args.embed_dim)

        self.row_rep = Sequential(LNLayer(2 * args.embed_dim, args.embed_dim), Linear(args.embed_dim, args.rank))
        self.col_rep = Sequential(LNLayer(2 * args.embed_dim, args.embed_dim), Linear(args.embed_dim, args.rank))

        self.stream_encoder = PermInvarEventEncoder(args.num_nodes, args.embed_dim)
        self.event_encoder = EventEncoder(args.num_nodes, args.embed_dim, args.msg_dim)

        self.row_mixer = ResMLPMixer(args.embed_dim, args.num_nodes)
        self.col_mixer = ResMLPMixer(args.embed_dim, args.num_nodes)

        self.row_norm = LayerNorm(args.embed_dim)
        self.col_norm = LayerNorm(args.embed_dim)


    def forward(self, mat_seqs, streams, args):
        # Initialize
        row_embed_seqs = []
        col_embed_seqs = []
        streams, bs = streams
        device = args.device

        # History Encoder
        for stream in mat_seqs:
            row, col = self.event_encoder.forward(stream, bs, device)
            row = row.reshape((bs, self.num_nodes, -1))
            col = col.reshape((bs, self.num_nodes, -1))
            row = self.row_mixer(row)
            col = self.col_mixer(col)
            row_embed_seqs += [row]
            col_embed_seqs += [col]

        hidden_idx = t.tensor([range(self.num_nodes) for _ in range(bs)], device=device)
        # row_hidden = self.row_init_hidden(hidden_idx)
        # col_hidden = self.col_init_hidden(hidden_idx)
        row_hidden = self.row_init_hidden(hidden_idx)
        col_hidden = self.col_init_hidden(hidden_idx)

        for trow_embed, tcol_embed in zip(row_embed_seqs, col_embed_seqs):
            row_hidden = self.row_recurrent.forward(trow_embed, row_hidden)
            col_hidden = self.col_recurrent.forward(tcol_embed, col_hidden)

        row_hidden = self.row_norm(row_hidden)
        col_hidden = self.col_norm(col_hidden)

        # Streaming
        stream_row, stream_col = self.stream_encoder.forward(streams, bs, device)

        row_rep = self.row_rep(t.cat([row_hidden, stream_row], dim=-1))
        col_rep = self.col_rep(t.cat([col_hidden, stream_col], dim=-1))

        recover = t.bmm(row_rep, col_rep.permute(0, 2, 1))
        recover = F.softplus(recover)

        return recover
