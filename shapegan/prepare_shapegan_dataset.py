import os
# Enable this when running on a computer without a screen:
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import torch
from tqdm import tqdm
import numpy as np
from util import ensure_directory
from multiprocessing import Pool
import traceback
from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_cube, scale_to_unit_sphere, BadMeshException
import VoxelSwapper
from render_functions import render_voxel

# Voxel resolutions to create.
# Set to [] if no voxels are needed.
# Set to [32] for for all models except for the progressively growing DeepSDF/Voxel GAN
# VOXEL_RESOLUTIONS = [8, 16, 32, 64]
# Options for virtual scans used to generate SDFs
USE_DEPTH_BUFFER = True
SCAN_COUNT = 50
SCAN_RESOLUTION = 1024
class PrepareShapeGanDataset():
    # Voxel resolutions to create.
    # Set to [] if no voxels are needed.
    # Set to [32] for for all models except for the progressively growing DeepSDF/Voxel GAN
    # VOXEL_RESOLUTIONS = [8, 16, 32, 64]
    VOXEL_RESOLUTIONS = [16, 32, 64]
    # Options for virtual scans used to generate SDFs
    USE_DEPTH_BUFFER = True
    SCAN_COUNT = 50
    SCAN_RESOLUTION = 1024
    DATASET_NAME = 'chair_table_combinations'
    DIRECTORY_MODELS = '../data/shapenet/03001627'
    MODEL_EXTENSION = '.obj'
    DIRECTORY_VOXELS = '../data/{:s}/voxels_{{:d}}/'.format(DATASET_NAME)
    DIRECTORY_BAD_MESHES = '../data/{:s}/bad_meshes/'.format(DATASET_NAME)
    base_path = '../data/part_objs/'

    def __init__(self):
        pass

    def get_model_files(self):
        for directory, _, files in os.walk(self.DIRECTORY_MODELS):
            for filename in files:
                if filename.endswith(self.MODEL_EXTENSION):
                    yield os.path.join(directory, filename)


    def get_hash(self,filename):
        return filename.split('/')[-3]

    def get_voxel_filename(self,model_filename, resolution):
        return os.path.join(self.DIRECTORY_VOXELS.format(resolution), self.get_hash(model_filename) + '.npy')

    def get_bad_mesh_filename(self,model_filename):
        return os.path.join(self.DIRECTORY_BAD_MESHES, self.get_hash(model_filename))


    def mark_bad_mesh(self,model_filename):
        filename = self.get_bad_mesh_filename(model_filename)
        ensure_directory(os.path.dirname(filename))
        open(filename, 'w').close()

    def is_bad_mesh_corinne(self, model_filename):
        return os.path.exists(os.path.join(self.DIRECTORY_BAD_MESHES, model_filename))

    def is_bad_mesh(self,model_filename):
        return os.path.exists(self.get_bad_mesh_filename(model_filename))

    def process_model_file(self,filename):
        try:
            if self.is_bad_mesh(filename):
                return

            mesh = trimesh.load(filename)

            voxel_filenames = [self.get_voxel_filename(filename, resolution) for resolution in self.VOXEL_RESOLUTIONS]
            if not all(os.path.exists(f) for f in voxel_filenames):
                mesh_unit_cube = scale_to_unit_cube(mesh)
                surface_point_cloud = get_surface_point_cloud(mesh_unit_cube, bounding_radius=3 ** 0.5,
                                                              scan_count=SCAN_COUNT, scan_resolution=SCAN_RESOLUTION)
                try:
                    for resolution in self.VOXEL_RESOLUTIONS:
                        voxels = surface_point_cloud.get_voxels(resolution, use_depth_buffer=USE_DEPTH_BUFFER,
                                                                check_result=True)
                        render_voxel(voxels, image_size=256, voxel_size=resolution, device=None,
                                     output_filename=f"images/ORIGINAL_{self.DATASET_NAME}_{resolution}_{int(torch.randint(1, 100, (1,)))}.gif")
                        np.save(self.get_voxel_filename(filename, resolution), voxels)
                        del voxels

                except BadMeshException:
                    tqdm.write("Skipping bad mesh. ({:s})".format(self.get_hash(filename)))
                    self.mark_bad_mesh(filename)
                    return
                del mesh_unit_cube, surface_point_cloud

        except:
            traceback.print_exc()


    def process_model_files(self):
        for res in self.VOXEL_RESOLUTIONS:
            ensure_directory(self.DIRECTORY_VOXELS.format(res))
        ensure_directory(self.DIRECTORY_BAD_MESHES)

        files = list(self.get_model_files())
        for filename in files:
            self.process_model_file(filename)


    def get_ids(self, obj_name):
        full_path = self.base_path + obj_name + "/"+obj_name+".txt"
        f = open(full_path)
        ids = [line[:-1] for line in f.readlines()]

        return ids
    def process_one_type(self, obj_type, type_ids):
        self.DATASET_NAME = obj_type
        self.DIRECTORY_BAD_MESHES = '../data/{:s}/bad_meshes/'.format(self.DATASET_NAME)
        self.DIRECTORY_VOXELS = '../data/{:s}/voxels_{{:d}}/'.format(self.DATASET_NAME, self.VOXEL_RESOLUTIONS)

        for i in range(len(type_ids)):
            print(type_ids[i])
            self.DIRECTORY_MODELS = f'../data/part_objs/{self.DATASET_NAME}/{type_ids[i]}/models'
            self.process_model_files()
    def process_all_types(self):

        chair_ids = self.get_ids("chair")
        table_ids = self.get_ids("table")
        print("chair_ids", chair_ids)
        self.process_one_type("chair", chair_ids)
        self.process_one_type("table", table_ids)
        voxel_swapper = VoxelSwapper.VoxelSwapper()
        for res in self.VOXEL_RESOLUTIONS:
            DIRECTORY_VOXELS_chairs = f'../data/chair/voxels_{res}/'
            DIRECTORY_VOXELS_tables = f'../data/table/voxels_{res}/'
            DIRECTORY_VOXELS_combined = f'../data/chair_table_combinations/voxels_{res}/'
            ensure_directory(DIRECTORY_VOXELS_combined)

            for t in table_ids:
                self.DIRECTORY_BAD_MESHES = '../data/table/bad_meshes/'

                if self.is_bad_mesh_corinne(t):
                    continue
                print(t)
                table_voxel = np.load(DIRECTORY_VOXELS_tables+t+".npy")
                for c in chair_ids:
                    self.DIRECTORY_BAD_MESHES = '../data/chair/bad_meshes/'

                    if self.is_bad_mesh_corinne(c):
                        continue
                    print(c)
                    chair_voxel = np.load(DIRECTORY_VOXELS_chairs + c + ".npy")
                    new_voxel, valid = voxel_swapper.swap_voxel_vertical(chair_voxel, table_voxel, res)
                    if valid:
                        render_voxel(new_voxel, image_size=256, voxel_size=res, device=None,
                                     output_filename=f"images/ORIGINAL_chair_table_{res}_{int(torch.randint(1,100,(1,)))}.gif")

                        np.save(DIRECTORY_VOXELS_combined + "TABLE_"+str(t)+"_CHAIR_"+str(c)+"npy", new_voxel)


if __name__ == '__main__':
    prepareShapeGanDataset = PrepareShapeGanDataset()
    prepareShapeGanDataset.process_all_types()