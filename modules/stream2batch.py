from collections import defaultdict
import numpy as np


def stream2batch(stream, max_bs=256, max_nodes=30):
    user_round_id = np.zeros((max_nodes, ), dtype=int) - 1
    item_round_id = np.zeros((max_nodes, ), dtype=int) - 1

    user_batch = defaultdict(list)
    item_batch = defaultdict(list)
    msg_batch = defaultdict(list)

    for user, item, msg in zip(*stream):
        uround_id = user_round_id[user]
        iround_id = item_round_id[item]
        round_id = max(uround_id, iround_id) + 1
        user_round_id[user] = round_id
        item_round_id[item] = round_id
        write_round_id = round_id
        while len(user_batch[write_round_id]) >= max_bs:
            user_round_id[user_round_id < round_id] = round_id
            item_round_id[item_round_id < round_id] = round_id
            write_round_id += 1
        user_batch[write_round_id].append(user)
        item_batch[write_round_id].append(item)
        msg_batch[write_round_id].append(msg)

    user_batch = list(user_batch.values())
    item_batch = list(item_batch.values())
    msg_batch = list(msg_batch.values())

    return user_batch, item_batch, msg_batch


def ReBatch(streams, num_nodes, max_bs=128):
    # Step 1: ReID & tBatch
    users, items, msgs = [], [], []
    for i, stream in enumerate(streams):
        user, item, msg = stream
        user += i * num_nodes
        item += i * num_nodes
        users += user.tolist()
        items += item.tolist()
        msgs += msg.tolist()

    # Step 2: tBatch
    bs = len(streams)
    batched_stream = stream2batch((users, items, msgs), max_bs, bs * num_nodes)
    return batched_stream


def mats_to_streams(mats, num_nodes):
    _, steps, nodes, _ = mats.shape
    streams = []
    for i in range(steps):
        batch_mats = mats[:,i,:,:]
        indices = batch_mats.nonzero()
        values = batch_mats[indices[:, 0], indices[:, 1], indices[:, 2]]
        source = (indices[:, 0]) * num_nodes + indices[:, 1]
        target = (indices[:, 0]) * num_nodes + indices[:, 2]
        streams.append((source, target, values))
    return streams


if __name__ == '__main__':

    import torch as t

    mat = t.rand((16, 12, 30, 30))
    mat[mat > 0.2] = 0

    streams = mats_to_streams(mat, 30)

    print(...)
