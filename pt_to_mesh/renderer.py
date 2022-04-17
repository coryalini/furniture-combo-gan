import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        T = torch.exp(-rays_density * deltas).to("cuda")
        my_ones = torch.ones((T.shape[0],1,1)).to("cuda")
        T = torch.cat((my_ones,T), 1)
        T = T[:,0:rays_density.shape[1]]
        T = torch.cumprod(T,1)
        weights = T*(1 - torch.exp(-rays_density * deltas))
        # TODO (1.5): Compute weight used for rendering from transmittance and density
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        new_feature = torch.sum(weights * rays_feature, dim=1)
        return new_feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]
        # Process the chunks of rays.
        chunk_outputs = []
        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']
            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            # print("feature",feature.shape)
            unit_vec_large = feature.reshape((weights.shape[0], weights.shape[1], 3))
            # assert (unit_vec_large.shape == (32768., 64, 3))

            feature = self._aggregate(weights, unit_vec_large)
            # print("feature",feature.shape)
            # assert (feature.shape == (32768., 3))
            # 32kx3
            # TODO (1.5): Render depth map

            # 32kx64x1
            # print("depth_values",depth_values.shape)
            depth = self._aggregate(weights, depth_values.unsqueeze(-1))
            # 32x1
            # assert (depth.shape == (32768, 1))


            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}
