import pyvista as pv
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


res = 512
step = 14

# data_pts = [
#     [256, 256],
# ]
# sigmas = [0.2,]
# bld_w = [75,]

data_pts = [
    [100, 100],
    [150, 300],
    [res - 100, res - 200]
]
# sigmas = [0.11, 0.11, 0.11]
# bld_w = [20, 15, 22]

sigmas = [0.11, 0.11, 0.11]
bld_w = [24, 12, 18]


data_pts = np.array(data_pts, dtype=np.float64)

def mixed_gaussian_2d(x, y, data_pts, sigmas, bld_w):
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    
    data_pts = torch.tensor(data_pts)
    sigmas = torch.tensor(sigmas)
    bld_w = torch.tensor(bld_w)
    
    diff_x = (x / res - data_pts[:, 0].view(-1, 1, 1) / res)**2
    diff_y = (y / res - data_pts[:, 1].view(-1, 1, 1) / res)**2
    dist_sq = diff_x + diff_y

    # Calculate the Gaussian values
    gaussians = torch.exp(-dist_sq / (2 * sigmas.view(-1, 1, 1)**2)) / (2 * np.pi * sigmas.view(-1, 1, 1)**2)
    
    weighted_gaussians = bld_w.view(-1, 1, 1) * gaussians

    mixed = torch.sum(weighted_gaussians, dim=0)
    return mixed

x = np.linspace(0, res-1, res)
y = np.linspace(0, res-1, res)
x, y = np.meshgrid(x, y)

x_param = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y_param = torch.tensor(y, dtype=torch.float32, requires_grad=True)
pdfs = mixed_gaussian_2d(x_param, y_param, data_pts, sigmas, bld_w)


log_pdfs = torch.log(pdfs)
torch.sum(log_pdfs).backward()
# Generate the surface landscape
pdfs = pdfs.clone().detach().numpy()
points_x, points_y, points_z = x.flatten(), y.flatten(), pdfs.flatten()
points = np.c_[points_x, points_y, points_z]

faces = []
for i in range(res - 1):
    for j in range(res - 1):
        idx = i * res + j
        # Define two triangles for each square in the grid
        faces.append([idx, idx + 1, idx + res])
        faces.append([idx + 1, idx + res + 1, idx + res])

faces = np.array(faces)  # Ensure faces is a 2D array

mesh = trimesh.Trimesh(vertices=points, faces=faces)

gradient_x = x_param.grad.numpy()
gradient_y = y_param.grad.numpy()
gradients = np.stack([gradient_x, gradient_y, np.zeros_like(gradient_x)], axis=-1)
gradients = gradients.reshape(-1, 3)


normals = mesh.vertex_normals
# rotated_vectors = np.zeros_like(gradients)
# for i in tqdm(range(len(gradients))):
#     # Create a rotation that aligns the z-axis with the normal
#     normal = normals[i]
#     if np.allclose(normal, [0, 0, 1]):
#         # If the normal is already pointing in the z direction, no rotation is needed
#         rotated_vectors[i] = gradients[i]
#     else:
#         # Find the rotation axis and angle
#         rotation, _ = R.align_vectors([normal], [[0, 0, 1]])
#         rotated_vectors[i] = rotation.apply(gradients[i])

# # Scale vectors for visualization
# vectors = rotated_vectors * 1200

faces_pv = np.hstack([[3, face[0], face[1], face[2]] for face in faces])
surf = pv.PolyData(mesh.vertices, faces=faces_pv)

# p = pv.Plotter(window_size=(3840, 3840), off_screen=True)
p = pv.Plotter(window_size=(3840, 2160))

# colors = [(1, 0, 0), (1, 1, 1)]  # Red to White
# n_bins = 256  # Number of bins in the colormap
# custom_cmap = mcolors.LinearSegmentedColormap.from_list('red_white', colors[::-1], N=n_bins)
# custom_cmap = plt.get_cmap('Reds')
custom_cmap = plt.get_cmap('BuPu')

##############
# x_center = res // 2
# y_center = res // 2
# distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)
# normalized_distances = distances / distances.max()

# # Calculate opacity based on distance (fade out towards the edges)
# k = 90  # Steepness of the curve
# x0 = 0.7  # Midpoint of the sigmoid curve
# opacities = 1 - 1 / (1 + np.exp(-k * (normalized_distances.flatten() - x0)))
# opacities[pdfs_flat > 0.01 * (np.max(pdfs_flat) - np.min(pdfs_flat)) + np.min(pdfs_flat)] = 1.
###########
# scalar_value = pdfs_norm[res - 1, 0]
# normalized_scalar = (scalar_value - pdfs_norm.min()) / (pdfs_norm.max() - pdfs_norm.min())
# background_color = custom_cmap(normalized_scalar)
# p.background_color = background_color 
##############

p.add_mesh(surf, scalars=pdfs.flatten(), opacity=1.0, cmap=custom_cmap, show_vertices=False, show_scalar_bar=False)
# p.add_mesh(surf, scalars=pdfs_norm.flatten(), opacity=opacities, cmap=custom_cmap, show_scalar_bar=False)

x_subgrid = np.arange(0, res, step)
y_subgrid = np.arange(0, res, step)
x_subgrid, y_subgrid = np.meshgrid(x_subgrid, y_subgrid)
grid_indices = np.ravel_multi_index((x_subgrid.flatten(), y_subgrid.flatten()), (res, res))


x_subgrid = np.arange(0, res, step * 3)
y_subgrid = np.arange(0, res, step * 3)
x_subgrid, y_subgrid = np.meshgrid(x_subgrid, y_subgrid)
grid_indices_sparse = np.ravel_multi_index((x_subgrid.flatten(), y_subgrid.flatten()), (res, res))

prob_threshold = 0.00
pdfs_flat = pdfs.flatten()
pts_idx = grid_indices[pdfs_flat[grid_indices] > prob_threshold * pdfs_flat.max()]
pts_idx_sparse = grid_indices_sparse[pdfs_flat[grid_indices_sparse] < prob_threshold * pdfs_flat.max()]
# vectors_subgrid = vectors[pts_idx]
# vector_subgrid_sparse = vectors[pts_idx_sparse] * 0.6
# vectors_subgrid = np.concatenate([vectors_subgrid, vector_subgrid_sparse])  
pts_idx = np.concatenate([pts_idx, pts_idx_sparse])
# vectors_subgrid = vectors[pts_idx]
points_subgrid = np.c_[points_x[pts_idx], points_y[pts_idx], points_z[pts_idx] + 1]

# arrow_mesh = pv.PolyData()
# arrow_mesh.points = points_subgrid #+ normals[pts_idx] * 10.0
# arrow_mesh["vectors"] = vectors_subgrid
# length = np.linalg.norm(vectors_subgrid, axis=1).mean() * 0.25
# arrows = arrow_mesh.glyph(orient="vectors", scale=True, factor=0.4)
# arrows = arrow_mesh.glyph(orient="vectors", scale=False, factor=length)
# p.add_mesh(arrows, opacity=1.0, color='black')# scalars="colors" , cmap='plasma')

# sphere_radius = 3 
# spheres = [pv.Sphere(radius=sphere_radius, center=[pt[0], pt[1], pdfs_norm[int(pt[0]), int(pt[1])]]) for pt, s in zip(data_pts, bld_w)]

def print_camera_position():
    print("Camera Position:", p.camera_position)
p.add_key_event('p', print_camera_position)

# p.camera.SetPosition((915.0402480352863, -847.0574060853203, 499.704460166908))
# p.set_focus((320.0705590005367, 278.29074125860194, 159.10115200705894),) 
# p.set_viewup((-0.11450332210404066, 0.23183429547891837, 0.965992675265673))
# p.screenshot("teaser.png", window_size=(3840, 3840))

# position = (1178.539021773769, -849.4717850960942, 626.3088375426138)
# focus = (415.45324016943596, 353.087393955779, 136.06445871394635)
# viewup = (-0.18450929518179715, 0.2683620997262891, 0.9454830000703416)
position = (1091.219566642046, 1568.0730874326089, 575.36601570934)
focus = (207.11267848424657, 403.5471649479393, 198.16112876486795)
viewup = (-0.13432903306272603, -0.21159201226385202, 0.9680829154687973)
p.camera.SetPosition(position)
p.set_focus(focus) 
p.set_viewup(viewup) 


p.show_axes()
p.show()

# After closing the window, print the camera settings
camera = p.camera
print(f"position = {p.camera_position[0]}")
print(f"focus = {p.camera_position[1]}")
print(f"viewup = {p.camera_position[2]}")

p.screenshot("teaser.png")
