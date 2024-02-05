# https://zhuanlan.zhihu.com/p/628375238

import numpy as np
from scipy.spatial import Delaunay, cKDTree


# define the function to be interpolated
def f(x, y):
    return np.sin(x * y)


# generate random scattered points and evaluate the function at those points
np.random.seed(0)
n_points = 500
x = np.random.uniform(-50, 50, n_points)
y = np.random.uniform(-50, 50, n_points)
z = f(x, y)

# create a grid of interpolation points
n_interp_pts_seed = 1000
n_interp_pts = n_interp_pts_seed * n_interp_pts_seed
xi = np.linspace(-50, 50, n_interp_pts_seed)
yi = np.linspace(-50, 50, n_interp_pts_seed)
interp_pts = np.meshgrid(xi, yi)
interp_pts = np.array(interp_pts).reshape(2, -1).T

# perform Delaunay triangulation on the scattered points
points = np.column_stack((x, y))
tri = Delaunay(points)

# create a KDTree for fast nearest neighbor search
tree = cKDTree(points)


# define the barycentric interpolation function
def barycentric_interpolation(vals, bary_coords):
    return np.sum(vals * bary_coords, axis=0)


# create an array to store the interpolated values
interp_vals = np.zeros((n_interp_pts,), dtype=np.float64)

# loop over the interpolation points
for i in range(n_interp_pts):
    # find the index of the triangle that contains the interpolation point
    j = tri.find_simplex(interp_pts[i])
    # if the interpolation point is outside the convex hull, use the nearest neighbor value
    if j == -1:
        dist, ind = tree.query(interp_pts[i].reshape(1, -1))
        interp_vals[i] = z[ind]
    else:
        # transform the interpolation point to barycentric coordinates
        barycentric_coords_pre = tri.transform[j, :2].dot(
            np.transpose(interp_pts[i] - tri.transform[j, 2])
        )
        barycentric_coords = np.append(
            np.transpose(barycentric_coords_pre), 1 - barycentric_coords_pre.sum(axis=0)
        )
        # interpolate the value using the barycentric coordinates and the values at the triangle vertices
        interp_vals[i] = barycentric_interpolation(
            z[tri.simplices[j]], barycentric_coords
        )

# reshape the interpolated values into a grid
interp_vals = interp_vals.reshape(n_interp_pts_seed, n_interp_pts_seed)

# plot the results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(100, 4))
ax[0].scatter(x, y, c=z, cmap="coolwarm", edgecolors="k")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Scattered Points")
im = ax[1].imshow(interp_vals, extent=(-50, 50, -50, 50), cmap="coolwarm")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Interpolated Surface")
fig.colorbar(im, ax=ax.ravel().tolist())
plt.show()