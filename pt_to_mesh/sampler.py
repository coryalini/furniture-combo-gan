import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth,self.max_depth, self.n_pts_per_ray, device="cuda")
        # TODO (1.4): Sample points from z values
        z_val_large = z_vals.repeat(ray_bundle.directions.shape[0])
        z_val_large = z_val_large.repeat_interleave(3)
        z_val_large = z_val_large.reshape((self.n_pts_per_ray * ray_bundle.directions.shape[0],3))
        z_val_large = z_val_large.reshape((len(ray_bundle.directions), len(z_vals), 3))
        unit_vec_large = torch.repeat_interleave(ray_bundle.directions, self.n_pts_per_ray, dim=0).to("cuda")
        unit_vec_large = unit_vec_large.reshape((len(ray_bundle.directions), len(z_vals), 3))
        sample_points = unit_vec_large * z_val_large + ray_bundle.origins[0].reshape(1,3)
        sample_points = sample_points.to("cuda")
        z_vals = z_vals.unsqueeze(-1)
        # assert(sample_points.shape == (len(ray_bundle.directions), self.n_pts_per_ray, 3))
        # assert((z_vals * torch.ones_like(sample_points[..., :1])).shape == (len(ray_bundle.directions), self.n_pts_per_ray, 1))
        # Return
        #sample points
        # 32k64x3
        # 32x64x1

        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}