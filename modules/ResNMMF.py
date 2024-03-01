import time

import torch as t
from torch.nn import *
from modules.rnn import StructuredRNN
from modules.event_encoder import PermInvarEventEncoder, LNLayer
from modules.graph_encoder import GraphAE
from torch.nn import functional as F

# V1: PermInvarEncoder for ALL
# V2: GNN-Backbone & PermInvar
# V3: Sparse CNN & PermInvar

class ResNMMFv1(Module):

    def __init__(self, args):
        super(ResNMMFv1, self).__init__()

        self.num_nodes = args.num_nodes
        self.embed_dim = args.embed_dim

        self.row_recurrent = StructuredRNN(args.embed_dim, args.embed_dim)
        self.col_recurrent = StructuredRNN(args.embed_dim, args.embed_dim)

        self.row_init_hidden = Embedding(self.num_nodes, args.embed_dim)
        self.col_init_hidden = Embedding(self.num_nodes, args.embed_dim)

        self.row_rep = Sequential(
            LNLayer(2 * args.embed_dim, args.rank),
            # Linear(args.embed_dim, args.rank)
        )

        self.col_rep = Sequential(
            LNLayer(2 * args.embed_dim, args.rank),
            # Linear(args.embed_dim, args.rank)
        )

        self.ctxs_encoder = PermInvarEventEncoder(args.num_nodes, args.embed_dim)
        self.curr_encoder = PermInvarEventEncoder(args.num_nodes, args.embed_dim)

    def forward(self, ctx_streams, curr_streams, args):

        # Initialize
        device = args.device
        return_time = args.return_time
        intermedia = args.intermedia
        row_embed_seqs = []
        col_embed_seqs = []
        stageTimes = []
        startTime = time.time()

        # 1. Offline Part
        ctx_streams, bs = ctx_streams
        for stream in ctx_streams:
            row_embeds, col_embeds = self.ctxs_encoder.forward(stream, bs, device=device)
            row_embeds = row_embeds#  / t.norm(row_embeds, dim=1, keepdim=True).detach()
            col_embeds = col_embeds#  / t.norm(col_embeds, dim=1, keepdim=True).detach()
            row_embed_seqs += [row_embeds]
            col_embed_seqs += [col_embeds]

        hidden_idx = t.tensor([range(self.num_nodes) for _ in range(bs)], device=device)
        row_hidden = self.row_init_hidden(hidden_idx)
        col_hidden = self.col_init_hidden(hidden_idx)
        # row_hidden = t.rand((bs, self.num_nodes, self.embed_dim))
        # col_hidden = t.rand((bs, self.num_nodes, self.embed_dim))

        for trow_embed, tcol_embed in zip(row_embed_seqs, col_embed_seqs):
            row_hidden = self.row_recurrent.forward(trow_embed, row_hidden)
            col_hidden = self.col_recurrent.forward(tcol_embed, col_hidden)

        offlineTime = time.time()
        stageTimes.append(offlineTime - startTime)

        # 2. Online Part
        startTime = time.time()
        curr_streams, bs = curr_streams
        stream_row, stream_col = self.curr_encoder.forward(curr_streams, bs, device, test=intermedia)
        stream_row = stream_row # / t.norm(stream_row, dim=1, keepdim=True).detach()
        stream_col = stream_col # / t.norm(stream_col, dim=1, keepdim=True).detach()
        onlineTime = time.time()
        stageTimes.append(onlineTime - startTime)

        # 3. Decoder Part
        startTime = time.time()
        row_rep = self.row_rep(t.cat([row_hidden, stream_row], dim=-1))
        col_rep = self.col_rep(t.cat([col_hidden, stream_col], dim=-1))
        recover = t.bmm(row_rep, col_rep.permute(0, 2, 1)).sigmoid()
        decoderTime = time.time()
        stageTimes.append(decoderTime - startTime)

        if return_time:
            return recover, stageTimes

        return recover


class ResNMMFv2(Module):

    def __init__(self, args):
        super(ResNMMFv2, self).__init__()

        self.num_nodes = args.num_nodes
        self.embed_dim = args.embed_dim

        self.row_embed = Embedding(self.num_nodes, args.embed_dim)
        self.col_embed = Embedding(self.num_nodes, args.embed_dim)

        self.row_recurrent = StructuredRNN(args.embed_dim, args.embed_dim)
        self.col_recurrent = StructuredRNN(args.embed_dim, args.embed_dim)

        self.row_init_hidden = Embedding(self.num_nodes, args.embed_dim)
        self.col_init_hidden = Embedding(self.num_nodes, args.embed_dim)

        self.row_rep = Sequential(LNLayer(2 * args.embed_dim, args.embed_dim), Linear(args.embed_dim, args.rank))
        self.col_rep = Sequential(LNLayer(2 * args.embed_dim, args.embed_dim), Linear(args.embed_dim, args.rank))

        self.stream_encoder = PermInvarEventEncoder(args.num_nodes, args.embed_dim)
        self.graph_encoder = GraphAE(args.embed_dim, args.msg_dim, depth=args.depth)

        self.row_norm = LayerNorm(args.embed_dim)
        self.col_norm = LayerNorm(args.embed_dim)



    def forward(self, mat_seqs, streams, args):
        # Initialize
        row_embed_seqs = []
        col_embed_seqs = []
        times = []
        startTime = 0
        streams, bs = streams
        device = args.device
        timer = args.timer

        # ------Asynchronous Part----- #
        # Read Graph Static Embeddings
        embed_idx = t.tensor([i for _ in range(bs) for i in range(self.num_nodes)], device=device)
        row_embed = self.row_embed(embed_idx)
        col_embed = self.col_embed(embed_idx)

        # Static Graph Encoder
        if timer:
            startTime = time.perf_counter()
        for graph in mat_seqs:
            row, col = self.graph_encoder.forward(graph.to(device), (row_embed, col_embed))
            row = row.reshape((bs, self.num_nodes, -1))
            col = col.reshape((bs, self.num_nodes, -1))
            # row = self.row_norm(row)
            # col = self.col_norm(col)
            # row_pooling = t.mean(row, dim=1, keepdim=True)
            # col_pooling = t.mean(col, dim=1, keepdim=True)
            # row_embed_seqs += [row + row_pooling]
            # col_embed_seqs += [col + col_pooling]
            row_embed_seqs += [row]
            col_embed_seqs += [col]



        if timer:
            times += [time.perf_counter() - startTime]

        # RNN Temporal Aggregation
        hidden_idx = t.tensor([range(self.num_nodes) for _ in range(bs)], device=device)
        row_hidden = self.row_init_hidden(hidden_idx)
        col_hidden = self.col_init_hidden(hidden_idx)


        if timer:
            startTime = time.perf_counter()

        for trow_embed, tcol_embed in zip(row_embed_seqs, col_embed_seqs):
            row_hidden = self.row_recurrent.forward(trow_embed, row_hidden)
            col_hidden = self.col_recurrent.forward(tcol_embed, col_hidden)


        row_hidden = self.row_norm(row_hidden)
        col_hidden = self.col_norm(col_hidden)

        if timer:
            times += [time.perf_counter() - startTime]

        # ------Streaming Part----- #
        if timer:
            startTime = time.perf_counter()

        stream_row, stream_col = self.stream_encoder.forward(streams, bs, device)

        if timer:
            times += [time.perf_counter() - startTime]

        if timer:
            startTime = time.perf_counter()

        # stream_row = t.zeros_like(stream_row)
        # stream_col = t.zeros_like(stream_col)
        # stream_row = stream_row / stream_row.norm(p=2, dim=-1, keepdim=True).detach()
        # stream_col = stream_col / stream_col.norm(p=2, dim=-1, keepdim=True).detach()
        row_rep = self.row_rep(t.cat([row_hidden, stream_row], dim=-1))
        col_rep = self.col_rep(t.cat([col_hidden, stream_col], dim=-1))

        recover = t.bmm(row_rep, col_rep.permute(0, 2, 1))
        recover = F.softplus(recover)
        if timer:
            times += [time.perf_counter() - startTime]

        if timer:
            return recover, times
        return recover



