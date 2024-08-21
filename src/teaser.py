import pyvista as pv
from pyvista import examples
import numpy as np
import torch
import matplotlib.pyplot as plt


res = 128
step = 15

data_pts = [
    [0, 0, 0],
    [res - 1, res-1, res-1]    
]
data_pts = np.array(data_pts, dtype=np.float64)

sigmas = [0.45, 0.5]
bld_w = [0.5, 0.5]

def mixed_gaussian(x, y, z, data_pts, sigmas, bld_w):
    x = x.unsqueeze(0)#.unsqueeze(-1).unsqueeze(-1)
    y = y.unsqueeze(0)#.unsqueeze(-1).unsqueeze(0)
    z = z.unsqueeze(0)#.unsqueeze(0).unsqueeze(-1)
    
    data_pts = torch.tensor(data_pts)#.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    sigmas = torch.tensor(sigmas)#.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    bld_w = torch.tensor(bld_w)#.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    diff_x = (x / res - data_pts[:, 0].view(-1, 1, 1, 1) / res)**2
    diff_y = (y / res - data_pts[:, 1].view(-1, 1, 1, 1) / res)**2
    diff_z = (z / res - data_pts[:, 2].view(-1, 1, 1, 1) / res)**2
    dist_sq = diff_x + diff_y + diff_z

    # Calculate the Gaussian values
    gaussians = torch.exp(-dist_sq / (2 * sigmas.view(-1, 1, 1, 1)**2)) / (2 * np.pi * sigmas.view(-1, 1, 1, 1)**2)**(3/2)
    
    weighted_gaussians = bld_w.view(-1, 1, 1, 1) * gaussians

    mixed = torch.sum(weighted_gaussians, dim=0)
    return mixed

# create a grid with gaussian value
x = np.linspace(0, res-1, res)
y = np.linspace(0, res-1, res)
z = np.linspace(0, res-1, res)
x, y, z = np.meshgrid(x, y, z)

grid = pv.StructuredGrid(x, y, z)

x_param = torch.tensor(x, dtype=torch.float32, requires_grad=True)
y_param = torch.tensor(y, dtype=torch.float32, requires_grad=True)
z_param = torch.tensor(z, dtype=torch.float32, requires_grad=True)
pdfs = mixed_gaussian(x_param, y_param, z_param, data_pts, sigmas, bld_w)

pdfs_norm = pdfs.flatten().clone().detach().numpy()

grid['pdfs_norm'] = pdfs_norm

log_pdfs = torch.log(pdfs)
torch.sum(log_pdfs).backward()
score = torch.stack([x_param.grad, y_param.grad, z_param.grad]).permute(1, 2, 3, 0)
score = score.reshape(-1, 3)
score = score.clone().detach().numpy()

x_subgrid = np.arange(step, res + 1, step)
y_subgrid = np.arange(step, res + 1, step)
z_subgrid = np.arange(step, res + 1, step)
x_subgrid, y_subgrid, z_subgrid = np.meshgrid(x_subgrid, y_subgrid, z_subgrid)
pts_idx = np.ravel_multi_index((x_subgrid.flatten(), y_subgrid.flatten(), z_subgrid.flatten()), (res, res, res))
# pts_idx = np.random.choice(np.arange(res**3), 1000)

points = grid.points[pts_idx]
vectors_subgrid = score[pts_idx]

# visualize the voxel grid
p = pv.Plotter()
p.add_mesh(grid, scalars='pdfs_norm', opacity=0.4, cmap='plasma')
# p.add_mesh(grid, scalars='pdfs_norm', opacity=0.5, cmap='inferno')
# p.add_mesh(grid, scalars='pdfs_norm', opacity=0.5, cmap='viridis')

# p.add_arrows(points, vectors_subgrid, mag=300.0, color='inferno')

colormap = plt.get_cmap('inferno')  
colors = colormap(pdfs_norm)[:, :3][pts_idx]
colors = np.ones_like(colors) * 255

arrow_mesh = pv.PolyData()
arrow_mesh.points = points
arrow_mesh["vectors"] = vectors_subgrid
arrow_mesh["colors"] = colors
arrows = arrow_mesh.glyph(orient="vectors", scale=False, factor=5.0)
p.add_mesh(arrows, opacity=0.5, color='black')# scalars="colors" , cmap='plasma')

sphere_radius = 3 
spheres = [pv.Sphere(radius=sphere_radius, center=pt) for pt in data_pts]
for sphere in spheres:
    p.add_mesh(sphere, color='red')

p.show_axes()
p.show()
