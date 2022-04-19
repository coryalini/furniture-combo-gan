
import pytorch3d
import torch

class VoxelSwapper:

    def __init__(self):
        pass


    def swap_voxel_vertical(self, chair_voxel, table_voxel):

        validity = {"chair":"valid", "table":"valid"}
        mesh, faces = pytorch3d.marching_cubes(chair_voxel)

        if mesh.shape[0] == 0:
            validity["chair"] = "invlid"

        mesh, faces = pytorch3d.marching_cubes(table_voxel)
        if mesh.shape[0] == 0:
            validity["table"] = "invlid"

        tmp_voxel = chair_voxel
        tmp_voxel[0:4,:,:] = table_voxel[0:4,:,:]

        return tmp_voxel, validity

    
vox = VoxelSwapper()
chair_voxel = torch.randint(512, (8, 8, 8))
table_voxel = torch.randint(512, (8, 8, 8))

new_vox, validity = vox.swap_voxel_vertical(chair_voxel, table_voxel)

print(new_vox)
print(validity)

