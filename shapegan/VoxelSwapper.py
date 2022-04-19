
import pytorch3d
import torch
import mcubes

class VoxelSwapper:

    def __init__(self):
        pass


    def swap_voxel_vertical(self, chair_voxel, table_voxel, res):

        validity = {"chair":"valid", "table":"valid"}
        vertices, faces = mcubes.marching_cubes(mcubes.smooth(table_voxel), isovalue=0)

        if vertices.shape[0] == 0:
            validity["chair"] = "invalid"
        vertices, faces = mcubes.marching_cubes(mcubes.smooth(table_voxel), isovalue=0)
        if vertices.shape[0] == 0:
            validity["table"] = "invalid"

        tmp_voxel = table_voxel
        tmp_voxel[0:int(res/2),:,:] = chair_voxel[0:int(res/2),:,:]

        return tmp_voxel, validity


# vox = VoxelSwapper()
# chair_voxel = torch.randint(8, (2, 2, 2))
# table_voxel = torch.randint(8, (2, 2, 2))
# print(chair_voxel)
# print(table_voxel)
# new_vox, validity = vox.swap_voxel_vertical(chair_voxel, table_voxel)
#
# print(new_vox)
# print(validity)

