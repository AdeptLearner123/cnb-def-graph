import torch
from torch.nn.utils.rnn import pad_sequence

def batchify(tensors, padding_value):
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def batchify_matrices(tensors, padding_value):
    x = max([t.shape[0] for t in tensors])
    y = max([t.shape[1] for t in tensors])
    out_matrix = torch.zeros((len(tensors), x, y))
    out_matrix += padding_value
    for i, tensor in enumerate(tensors):
        out_matrix[i][0 : tensor.shape[0], 0 : tensor.shape[1]] = tensor
    return out_matrix