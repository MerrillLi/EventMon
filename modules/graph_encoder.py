from torch.nn import *
from modules.event_encoder import LNLayer
import dgl.function as fn
import dgl as d
import torch as t

class NodeEdgeConv(Module):

    def __init__(self, embed_dim, msg_dim):
        super(NodeEdgeConv, self).__init__()
        self.embed_dim = embed_dim
        self.src_transform = Linear(embed_dim, embed_dim)
        self.dst_transform = Linear(embed_dim, embed_dim)
        self.srcmsg_transform = Linear(msg_dim, embed_dim)
        self.dstmsg_transform = Linear(msg_dim, embed_dim)
        self.apply_row = Sequential(LNLayer(embed_dim, embed_dim), Linear(embed_dim, embed_dim))
        self.apply_col = Sequential(LNLayer(embed_dim, embed_dim), Linear(embed_dim, embed_dim))

    def message_func(self, edges):
        msg = edges.dst['h'] * edges.data['msg']
        return {'m': msg}

    def forward(self, graph: d.DGLHeteroGraph, n_feats):
        src_embed, dst_embed = n_feats
        with graph.local_scope():
            graph.ndata['h'] = {'src': self.src_transform(src_embed),
                                'dst': self.dst_transform(dst_embed)}
            graph.edata['msg'] = {
                'src2dst': self.srcmsg_transform(graph.edata['v'][('src', 'src2dst', 'dst')]),
                'dst2src': self.dstmsg_transform(graph.edata['v'][('dst', 'dst2src', 'src')])
            }

            graph.update_all(self.message_func, fn.sum('m', 'out'), etype='src2dst')
            graph.update_all(self.message_func, fn.sum('m', 'out'), etype='dst2src')

            row_embed = graph.ndata['out']['src']
            col_embed = graph.ndata['out']['dst']

            # row_embed = F.normalize(row_embed, p=2, dim=-1)
            # col_embed = F.normalize(col_embed, p=2, dim=-1)
            row_embed = src_embed + self.apply_row(row_embed)
            col_embed = dst_embed + self.apply_col(col_embed)
            return row_embed, col_embed


class GraphAE(Module):

    def __init__(self, emb_dim, msg_dim, depth=1):
        super(GraphAE, self).__init__()
        self.depth = depth
        self.conv1 = NodeEdgeConv(emb_dim, msg_dim)
        if depth == 2:
            self.conv2 = NodeEdgeConv(emb_dim, msg_dim)

    def forward(self, graph, n_feats):
        row_embed, col_embed = n_feats
        row_embed, col_embed = self.conv1.forward(graph, (row_embed, col_embed))
        if self.depth == 2:
            row_embed, col_embed = self.conv2.forward(graph, (row_embed, col_embed))
        return row_embed, col_embed


class LinearAttention(Module):

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.fc_q = Linear(dim, dim)
        self.fc_k = Linear(dim, dim)
        self.fc_v = Linear(dim, dim)
        self.dim = dim
        self.sigmoid = Sigmoid()
        self.linear = Linear(dim, dim)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, Linear):
    #             init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, prev, curr):
        # Current : Query
        # Previous: Key=Value
        bs, n, dim = curr.shape
        q = self.fc_q(curr)
        k = self.fc_k(prev).view(1, bs, n, dim)
        v = self.fc_v(prev).view(1, bs, n, dim)

        numerator = t.sum(t.exp(k) * v, dim=2)
        denominator = t.sum(t.exp(k), dim=2)

        out = (numerator / denominator)
        out = self.sigmoid(q) * (out.permute(1, 0, 2))
        out = out.squeeze()
        out = self.linear(out)
        return out
