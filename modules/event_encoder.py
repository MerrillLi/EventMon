import torch as t
import math
from torch.nn import *

class LNLayer(Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(LNLayer, self).__init__()

        self.lin = Linear(in_feats, out_feats, bias)
        self.act = GELU()
        # self.norm = LayerNorm([out_feats])


    def forward(self, x):
        x = self.lin(x)
        # x = self.norm(x)
        x = self.act(x)
        return x


class LinearReLU(Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(LinearReLU, self).__init__()
        self.linear = Linear(in_feats, out_feats, bias=bias)
        self.act = LeakyReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x


class PermInvarEventEncoder(Module):

    def __init__(self, num_nodes, dim):
        super(PermInvarEventEncoder, self).__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.user_embeddings = Embedding(num_nodes, dim)
        self.item_embeddings = Embedding(num_nodes, dim)

        self.transform = Sequential(
            Linear(2 * dim + 1, 4 * dim),
        )

    def forward(self, streams, bs, device, test=False):

        user_state = t.zeros((self.num_nodes * bs, self.dim), device=device)
        item_state = t.zeros((self.num_nodes * bs, self.dim), device=device)

        user_global = t.zeros((self.num_nodes * bs, self.dim), device=device)
        item_global = t.zeros((self.num_nodes * bs, self.dim), device=device)

        fill_counter = 0
        # all_num = sum([len(stream) for stream in streams[0]])

        stream_rows = []
        stream_cols = []

        for users, items, messages in zip(*streams):
            # if len(users) <= 16:
            #     continue

            messages = t.tensor(messages, device=device).float().reshape(-1, 1)
            users = t.tensor(users).to(device)
            items = t.tensor(items).to(device)

            # Events
            user_embed = self.user_embeddings(users % self.num_nodes)
            item_embed = self.item_embeddings(items % self.num_nodes)
            event_inputs = t.cat([user_embed, item_embed, messages], dim=-1)

            enc = self.transform(event_inputs)
            user_update, item_update, user_gstate, item_gstate  = enc.chunk(4, -1)

            user_state[users] += user_update
            item_state[items] += item_update

            user_global[users] += user_gstate
            item_global[items] += item_gstate

            if test:
                fill_counter += len(users)

                if math.floor((fill_counter / all_num) / 0.25) != len(stream_rows):
                    uep = user_global.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2)
                    iep = item_global.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2)
                    uep = t.max(uep, dim=1, keepdim=True).values
                    iep = t.max(iep, dim=1, keepdim=True).values

                    u_im = user_state.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2) + uep
                    i_im = item_state.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2) + iep
                    stream_rows += [u_im]
                    stream_cols += [i_im]


        user_pooling = user_global.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2)
        item_pooling = item_global.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2)

        user_pooling = t.max(user_pooling, dim=1, keepdim=True).values
        item_pooling = t.max(item_pooling, dim=1, keepdim=True).values

        ret_uembeds = user_state.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2) + user_pooling
        ret_iembeds = item_state.reshape((self.num_nodes, bs, self.dim)).permute(1, 0, 2) + item_pooling

        if test:
            return stream_rows, stream_cols

        return ret_uembeds, ret_iembeds



