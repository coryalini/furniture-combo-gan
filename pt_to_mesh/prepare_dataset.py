import os
from tqdm import tqdm
from mesh_to_sdf import get_surface_point_cloud, BadMeshException
from multiprocessing import Pool
import trimesh
import numpy as np
DIRECTORY_MODELS = 'data/'
MODEL_EXTENSION = '.npy'
DIRECTORY_SDF = 'data/sdf/'
VOXEL_RESOLUTION = 32


def get_model_files():
    for directory, _, files in os.walk(DIRECTORY_MODELS):
        for filename in files:
            if filename.endswith(MODEL_EXTENSION):
                yield os.path.join(directory, filename)

def get_npy_filename(model_filename, qualifier=''):
    return DIRECTORY_SDF + model_filename[len(DIRECTORY_MODELS):-len(MODEL_EXTENSION)] + qualifier + '.npy'

def get_voxel_filename(model_filename):
    return get_npy_filename(model_filename, '-voxels-{:d}'.format(VOXEL_RESOLUTION))
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def scale_to_unit_sphere(mesh):
    mesh = trimesh.Trimesh(vertices=mesh.verts_list()[0].detach().cpu(), faces=mesh.faces_list()[0].detach().cpu())
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def process_model_file(mesh_new):
    voxels_filename = get_voxel_filename("model_test")
    mymesh = scale_to_unit_sphere(mesh_new)
    surface_point_cloud = get_surface_point_cloud(mymesh)
    try:
        voxels = surface_point_cloud.get_voxels(voxel_resolution=VOXEL_RESOLUTION, use_depth_buffer=True)
        print(voxels)
        ensure_directory(os.path.dirname(voxels_filename))
        np.save(voxels_filename, voxels)
    except BadMeshException:
        print()
        tqdm.write("Skipping bad mesh. ({:s})".format(voxels_filename))
        return
