import torch as t
from torch.nn import *

class StructuredRNN(Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(StructuredRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, -1)
        h_r, h_i, h_n = gate_h.chunk(3, -1)
        resetgate = t.sigmoid(i_r + h_r)
        inputgate = t.sigmoid(i_i + h_i)
        newgate = t.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy


class MatGRUCell(Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats, out_feats, Sigmoid())

        self.reset = MatGRUGate(in_feats, out_feats, Sigmoid())

        self.htilda = MatGRUGate(in_feats, out_feats, Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

