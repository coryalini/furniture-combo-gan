from rendering import MeshRenderer
from datasets import VoxelDataset
from torch.utils.data import DataLoader
viewer = MeshRenderer()


dataset = VoxelDataset.from_split('../data/chair_table_combinations/voxels_{:d}/{{:s}}.npy'.format(32),
                                  '../data/chair_table_combinations/train.txt')
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
count = 0
for valid_sample in data_loader:
    print(dataset.files[count][len("../data/chair_table_combinations/voxels_32/"):-len(".npy")])
    viewer.set_voxels(valid_sample[0].squeeze().detach().cpu().numpy())
    input("Press Enter to continue...")
    count +=1

viewer.stop()
