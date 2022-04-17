from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.autograd as autograd

import random
import time
import sys
from collections import deque
from tqdm import tqdm

from model.sdf_net import SDFNet
from model.progressive_gan import Discriminator, LATENT_CODE_SIZE, RESOLUTIONS
from util import create_text_slice, device, standard_normal_distribution, get_voxel_coordinates

SDF_CLIPPING = 0.1
from util import create_text_slice
from datasets import VoxelDataset
from torch.utils.data import DataLoader


def get_parameter(name, default):
    for arg in sys.argv:
        if arg.startswith(name + '='):
            return arg[len(name) + 1:]
    return default


ITERATION = int(get_parameter('iteration', 3))
# Continue with model parameters that were previously trained at the SAME iteration
# Otherwise, it will use the model parameters of the previous iteration or initialize randomly at iteration 0
CONTINUE = "continue" in sys.argv

FADE_IN_EPOCHS = 10
BATCH_SIZE = 16
GRADIENT_PENALTY_WEIGHT = 10
NUMBER_OF_EPOCHS = int(get_parameter('epochs', 250))

VOXEL_RESOLUTION = RESOLUTIONS[ITERATION]
print(VOXEL_RESOLUTION)
dataset = VoxelDataset.from_split('../data/chairs/voxels_{:d}/{{:s}}.npy'.format(VOXEL_RESOLUTION),
                                  '../data/chairs/test.txt')
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


def get_generator_filename(iteration):
    return 'hybrid_progressive_gan_generator_{:d}.to'.format(iteration)


generator = SDFNet(device='cpu')
generator.filename = get_generator_filename(ITERATION)
generator.load()

if torch.cuda.device_count() > 1:
    print("Using dataparallel with {:d} GPUs.".format(torch.cuda.device_count()))
    generator_parallel = nn.DataParallel(generator)
else:
    generator_parallel = generator

generator_parallel.to(device)

LOG_FILE_NAME = "plots/hybrid_gan_training_{:d}.csv".format(ITERATION)
first_epoch = 0
if 'continue' in sys.argv:
    log_file_contents = open(LOG_FILE_NAME, 'r').readlines()
    first_epoch = len(log_file_contents)

log_file = open(LOG_FILE_NAME, "a" if "continue" in sys.argv else "w")

generator_optimizer = optim.RMSprop(generator_parallel.parameters(), lr=0.0001)

show_viewer = "nogui" not in sys.argv

if show_viewer:
    from rendering import MeshRenderer

    viewer = MeshRenderer()


def sample_latent_codes(current_batch_size):
    latent_codes = standard_normal_distribution.sample(sample_shape=[current_batch_size, LATENT_CODE_SIZE]).to(device)
    latent_codes = latent_codes.repeat((1, 1, grid_points.shape[0])).reshape(-1, LATENT_CODE_SIZE)
    return latent_codes


grid_points = get_voxel_coordinates(VOXEL_RESOLUTION, return_torch_tensor=True)
grid_points_default_batch = grid_points.repeat((BATCH_SIZE, 1))

def test():
    progress = tqdm(total=NUMBER_OF_EPOCHS * (len(dataset) // BATCH_SIZE + 1),
                    initial=first_epoch * (len(dataset) // BATCH_SIZE + 1))

    for valid_sample in data_loader:
        try:

            valid_sample = valid_sample.to(device)
            current_batch_size = valid_sample.shape[0]
            if current_batch_size == BATCH_SIZE:
                batch_grid_points = grid_points_default_batch
            else:
                batch_grid_points = grid_points.repeat((current_batch_size, 1))

            generator_optimizer.zero_grad()

            latent_codes = sample_latent_codes(current_batch_size)
            fake_sample = generator_parallel(batch_grid_points, latent_codes)
            fake_sample = fake_sample.reshape(-1, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)
            if batch_index % 50 == 0 and show_viewer:
                viewer.set_voxels(fake_sample[0, :, :, :].squeeze().detach().cpu().numpy())
            if batch_index % 50 == 0 and "show_slice" in sys.argv:
                tqdm.write(create_text_slice(fake_sample[0, :, :, :] / SDF_CLIPPING))

            generator_optimizer.step()


            progress.update()
        except KeyboardInterrupt:
            if show_viewer:
                viewer.stop()
            return


test()
log_file.close()
