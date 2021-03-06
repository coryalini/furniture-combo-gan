import numpy as np
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import torch

from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
from pytorch3d.renderer import OpenGLPerspectiveCameras
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from mesh_to_sdf import BadMeshException

import mcubes
import skimage.measure
from create_plot import ImageGrid

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def render_points(
    points,
    cameras,
    image_size,
    save=False,
    file_prefix='',
    color=[0.7, 0.7, 1]
):
    device = points.device
    if device is None:
        device = get_device()

    # Get the renderer.
    points_renderer = get_points_renderer(image_size=image_size[0], radius=0.01)

    textures = torch.ones(points.size()).to(device)   # (1, N_v, 3)
    rgb = textures * torch.tensor(color).to(device)  # (1, N_v, 3)

    point_cloud = pytorch3d.structures.pointclouds.Pointclouds(
        points=points, features=rgb
    )
    
    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = points_renderer(point_cloud, cameras=cameras[cam_idx].to(device))
            image = image[0,:,:,:3].detach().cpu().numpy()
            all_images.append(image)

            # Save
            if save:
                plt.imsave(
                    f'{file_prefix}_{cam_idx}.png',
                    image
                )
    
    return all_images



def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.
    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def implicit_to_mesh(implicit_fn, scale=0.5, grid_size=128, device='cpu', color=[0.7, 0.7, 1], chunk_size=262144, thresh=0):
    Xs = torch.linspace(-1*scale, scale, grid_size+1).to(device)
    Ys = torch.linspace(-1*scale, scale, grid_size+1).to(device)
    Zs = torch.linspace(-1*scale, scale, grid_size+1).to(device)
    grid = torch.stack(torch.meshgrid(Xs, Ys, Zs), dim=-1)

    grid = grid.view(-1, 3)
    num_points = grid.shape[0]
    sdfs = torch.zeros(num_points)
    
    with torch.no_grad():
        for chunk_start in range(0, num_points, chunk_size):
            torch.cuda.empty_cache()
            chunk_end = min(num_points, chunk_start+chunk_size)
            sdfs[chunk_start:chunk_end] = implicit_fn.get_distance(grid[chunk_start:chunk_end,:]).view(-1)

        sdfs = sdfs.view(grid_size+1, grid_size+1, grid_size+1)

    vertices, triangles = mcubes.marching_cubes(sdfs.cpu().numpy(), thresh)
    # normalize to [-scale, scale]
    vertices = (vertices/grid_size - 0.5)*2*scale

    vertices = torch.from_numpy(vertices).unsqueeze(0).float()
    faces = torch.from_numpy(triangles.astype(np.int64)).unsqueeze(0)
    
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    return mesh
    

def render_geometry(
    model,
    cameras,
    image_size,
    save=False,
    thresh=0.,
    file_prefix=''    
):
    device = list(model.parameters())[0].device
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    mesh_renderer = get_mesh_renderer(image_size=image_size[0], lights=lights, device=device)

    mesh = implicit_to_mesh(model.implicit_fn, scale=3, device=device, thresh=thresh)
    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = mesh_renderer(mesh, cameras=cameras[cam_idx].to(device))
            image = image[0,:,:,:3].detach().cpu().numpy()
            all_images.append(image)

            # Save
            if save:
                plt.imsave(
                    f'{file_prefix}_{cam_idx}.png',
                    image
                )
    
    return all_images

# Spiral cameras looking at the origin
def create_surround_cameras(radius, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=1.0, at=None):
    cameras = []

    for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:

        if np.abs(up[1]) > 0:
            eye = [np.cos(theta + np.pi / 2) * radius, 1.0, -np.sin(theta + np.pi / 2) * radius]
        else:
            eye = [np.cos(theta + np.pi / 2) * radius, np.sin(theta + np.pi / 2) * radius, 2.0]
        if at is None:
            at =[0.0, 0.0, 0.0]
        R, T = look_at_view_transform(
            eye=(eye,),
            at=(at,),
            up=(up,),
        )

        cameras.append(
            PerspectiveCameras(
                focal_length=torch.tensor([focal_length])[None],
                principal_point=torch.tensor([0.0, 0.0])[None],
                R=R,
                T=T,
            )
        )

    return cameras
def render_mesh(
        mesh,
        image_size=256,
        save=False,
        file_prefix=''
):
    device = get_device()
    cameras = create_surround_cameras(3.0, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    mesh_renderer = get_mesh_renderer(image_size=image_size[0], lights=lights, device=device)

    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = mesh_renderer(mesh, cameras=cameras[cam_idx].to(device))
            image = image[0, :, :, :3].detach().cpu().numpy()
            all_images.append(image)

            # Save
            if save:
                plt.imsave(
                    f'{file_prefix}_{cam_idx}.png',
                    image
                )
    imageio.mimsave("original_mesh.gif", [np.uint8(im * 255) for im in all_images])

    return all_images


def render_partnet_bullshit(points,
            distances,
            image_size=256,
            save=False,
            file_prefix='',
            filename ="partnet_image.gif",):
    cameras = create_surround_cameras(3.0, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0)
    device = points.device
    if device is None:
        device = get_device()

    # Get the renderer.
    points_renderer = get_points_renderer(image_size=image_size, radius=0.01)
    min_val = 0
    max_val = 1
    norm_dist = (distances - torch.min(distances))/(torch.max(distances)- torch.min(distances))

    textures = torch.ones(points.size()).to(device)  # (1, N_v, 3)
    rgb = textures * torch.tensor([1.0,1.0,1.0]).to(device)  # (1, N_v, 3)

    for i in range(len(textures)):
        # if distances[i] > 0.5:
        #     rgb[i] = textures[i] * torch.tensor([0., 0.,0.]).to(device)
        # else:
        rgb[i] = textures[i] * torch.tensor([0.5, 0.5,norm_dist[i]]).to(device)
    point_cloud = pytorch3d.structures.pointclouds.Pointclouds(
        points=points.unsqueeze(0), features=rgb.unsqueeze(0)
    )

    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = points_renderer(point_cloud, cameras=cameras[cam_idx].to(device))
            image = image[0, :, :, :3].detach().cpu().numpy()
            all_images.append(image)

            # Save
            if save:
                plt.imsave(
                    f'{file_prefix}_{cam_idx}.png',
                    image
                )
    imageio.mimsave(filename, [np.uint8(im * 255) for im in all_images])

    return all_images


def render_voxel(voxels,image_size=256, voxel_size=64, device=None,output_filename="images/test_voxel.gif" ):
    # if voxels is None:
    #     voxels = np.load("../data/chairs/voxels_8/a4.npy")
    if device is None:
        device = get_device()

    size = 2
    voxel_resolution = voxel_size
    voxels_current = voxels
    voxels_current = np.pad(voxels_current, 1, mode='constant', constant_values=1)
                        
    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels_current, spacing=(size / voxel_resolution, size / voxel_resolution, size / voxel_resolution))
    # vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    print("vec",vertices.shape)
    print("faces",faces.shape)

    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    if vertices.shape[0] == 0:
        raise BadMeshException
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    # vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    # textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    # textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    textures = torch.ones_like(vertices.unsqueeze(0))  # (1, N_v, 3)
    textures = textures * torch.tensor([0.0,0.0,0.1])  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=pytorch3d.renderer.TexturesVertex(textures)).to(
        device
    )
    if voxel_size == 64:
        cameras = create_surround_cameras(voxel_size*2, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0, at=[32.0, 32.0, 16.0])
    elif voxel_size == 32:
        cameras = create_surround_cameras(voxel_size*2, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0, at=[16.0, 10.0, 16.0])
    else:
        cameras = create_surround_cameras(voxel_size*2, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=2.0, at=[0.0, 0.0, 0.0])

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    mesh_renderer = get_mesh_renderer(image_size=image_size, lights=lights, device=device)

    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = mesh_renderer(mesh, cameras=cameras[cam_idx].to(device))
            image = image[0,:,:,:3].detach().cpu().numpy()
            all_images.append(image)
    # fig = plot_scene({
    #     "figure": {
    #         "Mesh": mesh,
    #         "Camera": cameras[0],
    #         "Camera1": cameras[1],
    #         "Camera2": cameras[2],
    #         "Camera3": cameras[3],
    #         "Camera4": cameras[4],
    #         "Camera5": cameras[5],
    #         "Camera6": cameras[6],
    #         "Camera8": cameras[7],
    #         "Camera9": cameras[8],
    
    #     }
    # })
    # fig.show()
    imageio.mimsave(output_filename, [np.uint8(im * 255) for im in all_images])

def save_voxel_headless(voxels, filename):
    COUNT = 1
    plot = ImageGrid(COUNT, create_viewer=False)
    plot.set_voxels(voxels)
    
    plot.save(filename)