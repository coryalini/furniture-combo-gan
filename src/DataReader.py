import h5py
import numpy as np
import utils
import torch

# filename = "../sem_seg_h5/Chair-1/test-00.h5"
class DataReader:
    input_path ="../sem_seg_h5/"
    def __init__(self, input_path="../sem_seg_h5/"):
        self.input_path=input_path
    def read_data(self,data_name="/Chair-1/test-00.h5"):
        filename = self.input_path + data_name
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
            print("length ",torch.max( torch.tensor(f['label_seg'][2, :])))
            utils.viz_seg (torch.tensor(f['data'][2, :, :]), torch.tensor(f['label_seg'][2, :]), "test.gif", "cuda")
            return torch.tensor(f['data']), torch.tensor(f['label_seg'])
    def read_to_file(self):
        to = torch.load("gan_generator_voxels_chairs.to")

        # print(to.shape)
        print(to)
dataReader = DataReader()
# dataReader.read_to_file()
data, labels = dataReader.read_data("/Chair-1/test-00.h5")
np.savez("chair_ex",data[2, :, :])
