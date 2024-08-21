import pyvista as pv
from pyvista import examples
import numpy as np

res = 128
step = 10

data_pts = [
    [10, 10, 10],
    [118, 118, 118]    
]
data_pts = np.array(data_pts)

# create a grid with gaussian value
x = np.linspace(0, 128, res)
y = np.linspace(0, 128, res)
z = np.linspace(0, 128, res)
x, y, z = np.meshgrid(x, y, z)

# blending weights based on gaussian
bld_w = []
sigmas = [100, 100]
for i, (px, py, pz) in enumerate(data_pts):
    bld_w_i = np.exp(-0.5 * ((x - px)**2 + (y - py)**2 + (z - pz)**2) / sigmas[i] ** 2)
    bld_w.append(bld_w_i)

bld_w = np.stack(bld_w, axis=-1)
bld_w /= np.sum(bld_w, axis=-1, keepdims=True)

grid = pv.StructuredGrid(x, y, z)
# Calculate the distance to the nearest data point
paired_dist = [np.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2) for px, py, pz in data_pts]
# distances = np.min(paired_dist, axis=0)
distances = paired_dist[0] * bld_w[..., 0] + paired_dist[1] * bld_w[..., 1]

# Normalize the distances to [0, 1] and invert for the coloring (darker when closer)
distances_normalized = 1 - (distances / np.max(distances))
distances_normalized = distances_normalized.flatten()

# Assign the normalized distances as a scalar field to the grid
grid['distance'] = distances_normalized

# Determine the closest data point index for each voxel
closest_idx = np.argmin(paired_dist, axis=0)

# Calculate the vectors pointing towards the closest data point
vectors = np.zeros((res, res, res, 3))
for i, (px, py, pz) in enumerate(data_pts):
    mask = (closest_idx == i)
    vectors[mask, 0] = px - x[mask]
    vectors[mask, 2] = py - y[mask]
    vectors[mask, 1] = pz - z[mask]
    
# vectors = np.stack([[px - x, pz - z, py - y] for px, py, pz in data_pts], axis=-1)
# # vectors = np.stack([[px - x, py - y, pz - z] for px, py, pz in data_pts], axis=-1)
# vectors = vectors * bld_w[None, ...]
# vectors = np.sum(vectors, axis=-1).transpose(1, 2, 3, 0)

# Flatten the vectors for visualization
vectors = vectors.reshape(-1, 3)
vectors /= np.sum(bld_w, axis=-1).flatten()[:, None]
vectors /= 25


# Randomly pick up points for visualization
# pts_idx = np.random.choice(res**3, 500, replace=False)
# pts_idx = list(range(0, res**3, 3))
x_subgrid = np.arange(0, res, step)
y_subgrid = np.arange(0, res, step)
z_subgrid = np.arange(0, res, step)
x_subgrid, y_subgrid, z_subgrid = np.meshgrid(x_subgrid, y_subgrid, z_subgrid)
pts_idx = np.ravel_multi_index((x_subgrid.flatten(), y_subgrid.flatten(), z_subgrid.flatten()), (res, res, res))

points = grid.points[pts_idx]
vectors_subgrid = vectors[pts_idx]

# Create spheres at the data points
sphere_radius = 3  # Adjust the size of the spheres as needed
spheres = [pv.Sphere(radius=sphere_radius, center=pt) for pt in data_pts]

# visualize the voxel grid
p = pv.Plotter()
p.add_mesh(grid, scalars='distance', opacity=0.5, cmap='plasma')
p.add_arrows(points, vectors_subgrid, mag=1.5, color='black')


vectors = np.zeros((res, res, res, 3))
for i, (px, py, pz) in enumerate(data_pts):
    mask = (closest_idx != i)
    vectors[mask, 0] = px - x[mask]
    vectors[mask, 2] = py - y[mask]
    vectors[mask, 1] = pz - z[mask]
    
# Flatten the vectors for visualization
vectors = vectors.reshape(-1, 3)
vectors /= np.sum(bld_w, axis=-1).flatten()[:, None]
vectors /= 25

vectors_subgrid = vectors[pts_idx]
p.add_arrows(points, vectors_subgrid, mag=1.5, color='red')

# p.add_points(data_pts, color='red')
for sphere in spheres:
    p.add_mesh(sphere, color='red')
p.show_axes()
p.show()
