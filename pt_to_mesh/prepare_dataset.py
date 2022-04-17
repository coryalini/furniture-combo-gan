import os
from tqdm import tqdm
from mesh_to_sdf import get_surface_point_cloud, BadMeshException
from multiprocessing import Pool
import trimesh
import numpy as np
USE_DEPTH_BUFFER = True

from render_functions import render_voxel

DATASET_NAME = 'chairs'
# DIRECTORY_MODELS = '../data/6969/objs'
# MODEL_EXTENSION = '.obj'
DIRECTORY_VOXELS = '../data/{:s}/voxels_{{:d}}/'.format(DATASET_NAME)
DIRECTORY_BAD_MESHES = '../data/{:s}/bad_meshes/'.format(DATASET_NAME)

VOXEL_RESOLUTIONS = [8, 16, 32, 64]


def get_hash(filename):
    return filename.split('/')[-3]
#
# def get_voxel_filename(model_filename, resolution):
#     print(model_filename, resolution)
#     print(get_hash(model_filename))
#     return os.path.join(DIRECTORY_VOXELS.format(resolution), get_hash(model_filename) + '.npy')

def get_voxel_filename(model_filename, resolution):
    return os.path.join(DIRECTORY_VOXELS.format(resolution), model_filename + '.npy')

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def scale_to_unit_sphere(mesh):
    mesh = trimesh.Trimesh(vertices=mesh.verts_list()[0].detach().cpu(), faces=mesh.faces_list()[0].detach().cpu())
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def process_model_file(mesh_new,filename):
    mymesh = scale_to_unit_sphere(mesh_new)
    surface_point_cloud = get_surface_point_cloud(mymesh)
    voxel_filenames = [get_voxel_filename(filename, resolution) for resolution in VOXEL_RESOLUTIONS]
    print(voxel_filenames)
    # if not all(os.path.exists(f) for f in voxel_filenames):
    try:
        for resolution in VOXEL_RESOLUTIONS:
            voxels = surface_point_cloud.get_voxels(resolution, use_depth_buffer=USE_DEPTH_BUFFER,check_result=True)
            render_voxel(voxels, image_size=256, voxel_size=64, device=None,output_filename="images/pre_process_{:d}.gif".format(resolution))
            np.save(get_voxel_filename(filename, resolution), voxels)
            del voxels
    except BadMeshException:
        print("Skipping bad mesh. ({:s})".format(voxel_filenames))
        tqdm.write("Skipping bad mesh. ({:s})".format(voxel_filenames))
        return
    except Exception as e:
        print("process model file failed",e)