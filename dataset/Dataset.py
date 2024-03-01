import numpy as np
import torch as t
import dgl as d
from torch.utils.data import Dataset


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


def to_events(mat):
    src, dst = mat.nonzero()
    p = np.random.permutation(len(src))
    src, dst = src[p], dst[p]
    messages = mat[src, dst]
    streams = (src, dst, messages)
    return streams


class TensorDataset(Dataset):

    def __init__(self, sparseTensor):
        self.sparseTensor = sparseTensor
        self.tIdx, self.rIdx, self.cIdx = self.sparseTensor.nonzero()

    def __len__(self):
        return len(self.tIdx)

    def __getitem__(self, idx):
        tIdx = self.tIdx[idx]
        rIdx = self.rIdx[idx]
        cIdx = self.cIdx[idx]
        mVal = self.sparseTensor[tIdx, rIdx, cIdx]
        return tIdx, rIdx, cIdx, mVal


class NMMDataset(Dataset):

    def __init__(self, dataTensor, args):

        self.windows = args.windows
        self.density = args.density

        self.spNMM, self.fullNMM, self.mask = sparsify(dataTensor, args.density)

    def __len__(self):
        return len(self.spNMM) - self.windows - 1

    def __getitem__(self, idx):

        # Context Window = args.window
        ctxNMMs = self.spNMM[idx:idx + self.windows]
        currNMM = self.spNMM[idx + self.windows]
        fullNMM = self.fullNMM[idx + self.windows]
        currMask = self.mask[idx + self.windows]

        negNMM = np.random.randint(len(self), size=(5,))

        negNMM = self.spNMM[negNMM]

        return ctxNMMs, currNMM, fullNMM, negNMM, currMask


class EventNMMDataset(Dataset):

    def __init__(self, dataTensor, args):

        self.window = args.window
        self.density = args.density

        self.spNMM, self.fullNMM, self.mask = sparsify(dataTensor, args.density)

    def __len__(self):
        return len(self.spNMM) - self.window - 1

    def __getitem__(self, idx):

        # Context Window = args.window
        ctxNMMs = self.spNMM[idx:idx + self.window]
        assert len(ctxNMMs) == self.window
        currNMM = self.spNMM[idx + self.window]
        fullNMM = self.fullNMM[idx + self.window]
        currMask = self.mask[idx + self.window]

        ctxStreams = [to_events(nmm) for nmm in ctxNMMs]
        currStream = [to_events(currNMM)]
        return ctxStreams, currStream, fullNMM, currMask


class GraphStreamDataset(Dataset):

    def __init__(self, data, args):
        self.data = data.astype('float32')
        self.num_nodes = self.data.shape[-1]
        self.window = args.window
        self.density = args.density

        self.spNMM, self.fullNMM, self.mask = sparsify(data, args.density)

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
        real_mat = self.data[idx + self.window - 1]
        mask = self.mask[idx + self.window - 1]

        # batch of streams
        current_mat = self.spNMM[idx + self.window - 1]

        src, dst = current_mat.nonzero()
        p = np.random.permutation(len(src))
        src, dst = src[p], dst[p]
        messages = current_mat[src, dst]
        streams = (src, dst, messages)
        return graphs, streams, real_mat, mask


class NMMStreamDataset(Dataset):

    def __init__(self, data, args):
        self.data = data.astype('float32')
        self.num_nodes = self.data.shape[-1]
        self.window = args.window
        self.density = args.density
        self.spNMM, self.fullNMM, self.mask = sparsify(data, args.density)

    def __len__(self):
        return self.spNMM.shape[0] - self.window

    def __getitem__(self, idx):

        # Context NMM
        mat_seqs = self.spNMM[idx: idx + self.window - 1]

        # Label and Mask for Training
        real_mat = self.data[idx + self.window - 1]
        mask = self.mask[idx + self.window - 1]

        # batch of streams
        current_mat = self.spNMM[idx + self.window - 1]

        src, dst = current_mat.nonzero()
        p = np.random.permutation(len(src))
        src, dst = src[p], dst[p]
        messages = current_mat[src, dst]
        streams = (src, dst, messages)
        return mat_seqs, streams, real_mat, mask

