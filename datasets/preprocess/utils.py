import pickle

import torch


def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x): return todevice(x, 'numpy')
def to_cpu(x): return todevice(x, 'cpu')
def to_cuda(x): return todevice(x, 'cuda')


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor
    """
    return batch[0]


def load_file_pkl_file(file_path):
    """
    Load a .pkl file.

    :param file_path: The path to the .pkl file.
    :return: The loaded data.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_torch_pkl_file(file_path):
    """
    Load a .pkl file.

    :param file_path: The path to the .pkl file.
    :return: The loaded data.
    """

    with open(file_path, 'rb') as file:
        data = torch.load(file, weights_only=False)
    return data


def save_dict_to_pkl(dictionary, file_path):
    """
    Save a dictionary to a .pkl file.

    :param dictionary: The dictionary to save.
    :param file_path: The path to the .pkl file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)
