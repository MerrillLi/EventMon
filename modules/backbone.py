from torch.nn import *
import torch_scatter as ts
import torch as t

class EventEncoder(Module):

    def __init__(self, num_nodes, embed_dim, msg_dim):
        super(EventEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.user_embeddings = Embedding(num_nodes, embed_dim)
        self.item_embeddings = Embedding(num_nodes, embed_dim)
        self.transform = Sequential(Linear(2 * embed_dim + msg_dim, 4 * embed_dim), LayerNorm(4*embed_dim), Linear(4*embed_dim,4*embed_dim))

    def forward(self, streams, bs, device):

        users, items, messages = streams
        messages = messages.float().reshape(-1, 1)
        users = users.to(device)
        items = items.to(device)
        user_state = t.zeros((self.num_nodes * bs, self.embed_dim), device=device)
        item_state = t.zeros((self.num_nodes * bs, self.embed_dim), device=device)

        user_global = t.zeros((self.num_nodes * bs, self.embed_dim), device=device)
        item_global = t.zeros((self.num_nodes * bs, self.embed_dim), device=device)

        # Events
        user_embed = self.user_embeddings(users % self.num_nodes)
        item_embed = self.item_embeddings(items % self.num_nodes)
        event_inputs = t.cat([user_embed, item_embed, messages], dim=-1)

        enc = self.transform(event_inputs)
        user_update, item_update, user_gstate, item_gstate  = enc.chunk(4, -1)

        ts.scatter_add(user_update, users, dim=0, out=user_state)
        ts.scatter_add(item_update, items, dim=0, out=item_state)
        ts.scatter_add(user_gstate, users, dim=0, out=user_global)
        ts.scatter_add(item_gstate, items, dim=0, out=item_global)

        user_pooling = user_global.reshape((self.num_nodes, bs, self.embed_dim)).permute(1, 0, 2)
        item_pooling = item_global.reshape((self.num_nodes, bs, self.embed_dim)).permute(1, 0, 2)

        user_pooling = t.mean(user_pooling, dim=1, keepdim=True)
        item_pooling = t.mean(item_pooling, dim=1, keepdim=True)

        ret_uembeds = user_state.reshape((self.num_nodes, bs, self.embed_dim)).permute(1, 0, 2) + user_pooling
        ret_iembeds = item_state.reshape((self.num_nodes, bs, self.embed_dim)).permute(1, 0, 2) + item_pooling
        return ret_uembeds, ret_iembeds
