import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from tqdm import tqdm

from model import LATENT_CODE_SIZE, LATENT_CODES_FILENAME
import random
from util import device
import imageio

class ImageGrid():
    def __init__(self, width, height=1, cell_width=3, cell_height=None, margin=0.2, create_viewer=True, crop=True):
        print("Plotting...")
        self.width = width
        self.height = height
        cell_height = cell_height if cell_height is not None else cell_width

        self.figure, self.axes = plt.subplots(height, width,
                                              figsize=(width * cell_width, height * cell_height),
                                              gridspec_kw={'left': 0, 'right': 1, 'top': 1, 'bottom': 0,
                                                           'wspace': margin, 'hspace': margin})
        self.figure.patch.set_visible(False)

        self.crop = crop
        if create_viewer:
            from rendering import MeshRenderer
            self.viewer = MeshRenderer(start_thread=False)
        else:
            self.viewer = None

    def set_image(self, image, x=0, y=0):
        cell = self.axes[y, x] if self.height > 1 and self.width > 1 else self.axes[x + y]
        cell.imshow(image)
        cell.axis('off')
        cell.patch.set_visible(False)

    def set_voxels(self, voxels, x=0, y=0, color=None):
        if color is not None:
            self.viewer.model_color = color
        self.viewer.set_voxels(voxels)
        image = self.viewer.get_image(crop=self.crop)
        self.set_image(image, x, y)

    def save(self, filename):
        plt.axis('off')
        extent = self.figure.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        plt.savefig(filename, bbox_inches=extent, dpi=400)
        if self.viewer is not None:
            self.viewer.delete_buffers()


def load_sdf_net(filename=None, return_latent_codes=False):
    from model.sdf_net import SDFNet, LATENT_CODES_FILENAME
    sdf_net = SDFNet()
    if filename is not None:
        sdf_net.filename = filename
    sdf_net.load()
    sdf_net.eval()

    if return_latent_codes:
        latent_codes = torch.load(LATENT_CODES_FILENAME).to(device)
        latent_codes.requires_grad = False
        return sdf_net, latent_codes
    else:
        return sdf_net


from rendering.raymarching import render_image
from util import standard_normal_distribution

generator = load_sdf_net(filename='hybrid_progressive_gan_generator_3-epoch-00250.to')

COUNT = 5

codes = standard_normal_distribution.sample([COUNT, LATENT_CODE_SIZE]).to(device)

plot = ImageGrid(COUNT, create_viewer=False)
angles = np.linspace(0, 360, 15)
for im in range(COUNT):
    images = []
    for a in angles:
        print("Starting iteration", a)
        image = render_image(generator, codes[im, :], radius=1.6, crop=True, sdf_offset=-0.045,
                             vertical_cutoff=1, angle=a)
        images.append(image)
    print("Saving image", im)
    imageio.mimsave(f"plots/results_{im}.gif",images)
