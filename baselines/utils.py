import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset.Dataset import NMMDataset

def ErrMetrics(pred, true):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE



class TensorDataset(Dataset):

    def __init__(self, tensor, offset=None):
        self.tensor = tensor
        self.offset = offset
        self.tIdx, self.rIdx, self.cIdx = self.tensor.nonzero()

    def __len__(self):
        return len(self.tIdx)

    def __getitem__(self, id):
        tIdx = self.tIdx[id]
        rIdx = self.rIdx[id]
        cIdx = self.cIdx[id]
        mVal = self.tensor[tIdx, rIdx, cIdx]
        if self.offset is not None:
            tIdx += self.offset
        return tIdx, rIdx, cIdx, mVal


def get_tensor(args):
    tensor = None
    if args.dataset == 'abilene':
        tensor = np.load('./dataset/abilene.npy').astype('float32')[:args.num_times]

    if args.dataset == 'geant':
        tensor = np.load('./dataset/geant.npy').astype('float32')[:args.num_times]
    return tensor


def get_loaders(args):

    tensor = get_tensor(args)
    quantile = np.percentile(tensor, q=99)
    tensor[tensor > quantile] = quantile
    tensor /= quantile
    density = args.density
    mask = np.random.rand(*tensor.shape).astype('float32')
    mask[mask > density] = 1
    mask[mask < density] = 0

    trainTensor = tensor * (1 - mask)
    testTensor = tensor * mask

    trainset = TensorDataset(trainTensor[:3500])
    testset = TensorDataset(testTensor[3000:3500], offset=args.offset)
    trainLoader = DataLoader(trainset, batch_size=64, shuffle=True)
    testLoader = DataLoader(testset, batch_size=64, shuffle=True)
    return trainLoader, testLoader


def get_nmm_loaders(args):

    tensor = get_tensor(args)
    quantile = np.percentile(tensor, q=99)
    tensor[tensor > quantile] = quantile
    tensor /= quantile
    # density = args.density
    # mask = np.random.rand(*tensor.shape).astype('float32')
    # mask[mask > density] = 1
    # mask[mask < density] = 0
    #
    # trainTensor = tensor * (1 - mask)
    # testTensor = tensor * mask

    trainset = NMMDataset(tensor[:3000], args)
    testset = NMMDataset(tensor[3000:4000], args)
    trainLoader = DataLoader(trainset, batch_size=32, shuffle=True)
    testLoader = DataLoader(testset, batch_size=32)
    return trainLoader, testLoader
