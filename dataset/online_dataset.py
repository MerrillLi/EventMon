import dgl as d
import torch as t
import numpy as np
from torch.utils.data import Dataset
from modules.stream2batch import ReBatch


def sparsify(data, density):

    tIdx, srcIdx, dstIdx = data.nonzero()
    p = np.random.permutation(len(tIdx))
    size = int(np.prod(data.shape) * density)
    tIdx, srcIdx, dstIdx = tIdx[p][:size], srcIdx[p][:size], dstIdx[p][:size]

    # Sparse Data
    sparseData = np.zeros_like(data)
    sparseData[tIdx, srcIdx, dstIdx] = data[tIdx, srcIdx, dstIdx]

    # Full Data
    fullData = data.copy()

    # Mask Indicates Sampled Location
    mask = np.zeros_like(data)
    mask[sparseData != 0] = 1
    return sparseData, fullData, mask


def collate_graph(data_list, num_nodes):
    batch_size = len(data_list)
    num_wins = len(data_list[0][0])
    win_graphs = [[] for _ in range(num_wins)]
    streams = []
    real_mats = []
    masks = []

    for graphs, stream, real_mat, mask in data_list:
        for i, winGraph in enumerate(graphs):
            win_graphs[i].append(winGraph)
        streams += [stream]
        real_mats += [t.FloatTensor(real_mat)]
        masks += [t.FloatTensor(mask)]

    win_graphs = [d.batch(graphs) for graphs in win_graphs]
    real_mats = t.stack(real_mats)
    masks = t.stack(masks)
    streams = ReBatch(streams, num_nodes)
    return win_graphs, (streams, batch_size), real_mats, masks


class GraphStreamDataset(Dataset):

    def __init__(self, spNMMs, fuNMMs, mask, args):
        self.num_nodes = spNMMs.shape[-1]
        self.window = args.window
        self.density = args.density
        self.spNMM, self.fullNMM, self.mask = spNMMs, fuNMMs, mask
        self.cache = dict()

    def __len__(self):
        return self.spNMM.shape[0] - self.window

    def mat2graph(self, mat: np.ndarray):
        srcIdx, dstIdx = mat.nonzero()
        edge_attrs = t.FloatTensor(mat[srcIdx, dstIdx]).reshape(-1, 1)

        graph = d.heterograph(data_dict={
            ('src', 'src2dst', 'dst'): (srcIdx, dstIdx),
            ('dst', 'dst2src', 'src'): (dstIdx, srcIdx)
        }, num_nodes_dict={
            'src': self.num_nodes,
            'dst': self.num_nodes}
        )  # type:d.DGLHeteroGraph
        graph.edata['v'] = {
            'src2dst': edge_attrs,
            'dst2src': edge_attrs
        }
        return graph

    def __getitem__(self, idx):
        # Get Graph Sequence of [T-window, T-1]
        if idx not in self.cache:
            graphs = []
            for shift in range(self.window - 1):
                mat = self.spNMM[idx + shift]
                graphs += [self.mat2graph(mat)]
            self.cache[idx] = graphs
        graphs = self.cache[idx]

        # Label and Mask for Training
        real_mat = self.fullNMM[idx + self.window - 1]
        mask = self.mask[idx + self.window - 1]

        # batch of streams
        current_mat = self.spNMM[idx + self.window - 1]

        src, dst = current_mat.nonzero()
        p = np.random.permutation(len(src))
        src, dst = src[p], dst[p]
        messages = current_mat[src, dst]
        streams = (src, dst, messages)
        return graphs, streams, real_mat, mask
