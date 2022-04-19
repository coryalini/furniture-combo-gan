
import pytorch3d
import torch
import mcubes

class VoxelSwapper:

    def __init__(self):
        pass


    def swap_voxel_vertical(self, chair_voxel, table_voxel, res):

        vertices_chair, faces = mcubes.marching_cubes(mcubes.smooth(table_voxel), isovalue=0)
        vertices_table, faces = mcubes.marching_cubes(mcubes.smooth(table_voxel), isovalue=0)

        valid = True 
        if vertices_chair.shape[0] == 0 or vertices_table.shape[0] == 0:
            valid = False

        tmp_voxel = table_voxel
        # tmp_voxel[0:int(res/2),:,:] = torch.min(chair_voxel[0:int(res/2),:,:],table_voxel[0:int(res/2),:,:])
        tmp_voxel = torch.min(torch.Tensor(chair_voxel),torch.Tensor(table_voxel))

        return tmp_voxel, valid


# vox = VoxelSwapper()
# chair_voxel = torch.randint(8, (2, 2, 2))
# table_voxel = torch.randint(8, (2, 2, 2))
# print(chair_voxel)
# print(table_voxel)
# new_vox, validity = vox.swap_voxel_vertical(chair_voxel, table_voxel)
#
# print(new_vox)
# print(validity)

