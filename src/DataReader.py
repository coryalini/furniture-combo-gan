import h5py
import numpy as np
import utils
import torch
from prepare_partnet_dataset import get_uniform_and_surface_points, get_uniform_filename, get_surface_filename,POINT_CLOUD_SAMPLE_SIZE,DIRECTORY_UNIFORM,DIRECTORY_SURFACE,DIRECTORY_BAD_MESHES,get_model_files
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
            # utils.viz_seg (torch.tensor(f['data'][2, :, :]), torch.tensor(f['label_seg'][2, :]), "test.gif", "cuda")
            return torch.tensor(f['data']), torch.tensor(f['label_seg'])
    def test_surface_ptcloud(self, surface_point_cloud):
        # ensure_directory(DIRECTORY_UNIFORM)
        # ensure_directory(DIRECTORY_SURFACE)
        # ensure_directory(DIRECTORY_BAD_MESHES)

        files = list(get_model_files())
        surface_point_cloud = surface_point_cloud[0,:,:].squeeze(0)
        for filename in files:
            print(filename)

            uniform_points, uniform_sdf, near_surface_points, near_surface_sdf = get_uniform_and_surface_points(
                surface_point_cloud, number_of_points=10000)

            combined_uniform = np.concatenate((uniform_points, uniform_sdf[:, np.newaxis]), axis=1)
            np.save(get_uniform_filename(filename), combined_uniform)

            combined_surface = np.concatenate((near_surface_points, near_surface_sdf[:, np.newaxis]), axis=1)
            np.save(get_surface_filename(filename), combined_surface)

    def read_to_file(self):
        to = torch.load("gan_generator_voxels_chairs.to")

        # print(to.shape)
        print(to)
dataReader = DataReader()
# dataReader.read_to_file()
data, labels = dataReader.read_data("/Chair-1/test-00.h5")
dataReader.test_surface_ptcloud(data)
# np.savez("chair_ex",data[2, :, :])
