

class VoxelSwapper:

    def __init__(self):
        pass

    def swap_voxel_vertical(self, chair_voxel, table_voxel):
        tmp_voxel = chair_voxel
        tmp_voxel[0:4,:,:] = table_voxel[0:4,:,:]
        return tmp_voxel


    # def swapVoxels_horizontal(self):