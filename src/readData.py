import h5py
import numpy as np
import utils
import torch

filename = "../sem_seg_h5/Door-2/test-00.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

    print(f['data'])
    print(np.max(f['label_seg']))
    print(f['data_num'])
    # for i in data:
    #     print(i, type(i))

    utils.viz_seg (torch.tensor(f['data'][0, :, :]), torch.tensor(f['label_seg'][0, :]), "test.gif", "cuda")