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

generator = load_sdf_net(filename='pc_progressive_hybrid_progressive_gan_generator_0.to')

COUNT = 30

codes = standard_normal_distribution.sample([COUNT, LATENT_CODE_SIZE]).to(device)

plot = ImageGrid(COUNT, create_viewer=False)
angles = np.linspace(0, 360, 30)
for im in range(COUNT):
    images = []
    for a in angles:
        print("Starting iteration", a)
        image = render_image(generator, codes[im, :], resolution=200, radius=1.6, crop=True, sdf_offset=-0.045,
                             vertical_cutoff=1, angle=a)
        images.append(image)
    print("Saving image", im)
    imageio.mimsave(f"pc_plots_32/results_{im}.gif",images)

# if "hybrid_gan_interpolation" in sys.argv:
#     from rendering.raymarching import render_image_for_index, render_image
#     from util import standard_normal_distribution
#     import cv2
#     sdf_net = load_sdf_net(filename='hybrid_gan_generator.to')

#     OPTIONS = 10
    
#     codes = standard_normal_distribution.sample([OPTIONS, LATENT_CODE_SIZE]).to(device)
    # for i in range(OPTIONS):
    #     image = render_image(sdf_net, codes[i, :], resolution=200, radius=1.6, sdf_offset=-0.045, vertical_cutoff=1, crop=True)
    #     image.save('plots/option-{:d}.png'.format(i))
    
STEPS = 30
        
code_start = codes[int(input('Enter index for starting shape: ')), :]
code_end = codes[int(input('Enter index for ending shape: ')), :]

with torch.no_grad():
    codes = torch.zeros([STEPS, LATENT_CODE_SIZE], device=device)
    for i in range(STEPS):
        codes[i, :] = code_start * (1.0 - i / (STEPS - 1)) + code_end * i / (STEPS - 1)

plot = ImageGrid(STEPS, create_viewer=False)

angles = np.linspace(0, 360, STEPS)
for im in range(STEPS):
    images = []
    for a in angles:
        print("Starting iteration", a)
        image = render_image(generator, codes[im, :], resolution=200, radius=1.6, crop=True, sdf_offset=-0.045,
                            vertical_cutoff=1, angle=a)
        images.append(image)
    print("Saving image", im)
    imageio.mimsave(f"pc_plots_32_interpolation/results_{im}.gif",images)
    
# for i in range(STEPS):
#     plot.set_image(render_image(generator, codes[i, :], crop=True, radius=1.6, sdf_offset=-0.045, vertical_cutoff=1), i)