import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

# rng = np.random.default_rng()
# points = rng.random((100, 2))

# vor = Voronoi(points)

# print(f"total vertices: {len(vor.vertices)}")
# # for r in vor.regions:
# #     print(r)

# fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False)
# plt.savefig("output.png")


def generate_voronoi(size: int, n_point: int) -> Voronoi:
    rng = np.random.default_rng()
    points = rng.random((n_point, 2))*size
    vor = Voronoi(points)

    return vor
