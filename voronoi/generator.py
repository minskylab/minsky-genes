from numpy import ndarray
from numpy.random import default_rng
from scipy.spatial import Voronoi


def uniform_random_points(n_points: int, amplitude: float) -> ndarray:
    rng = default_rng()
    points = rng.random((n_points, 2))*amplitude

    return points


def seeded_random_points(seed: int, n_points: int, amplitude: float) -> ndarray:
    rng = default_rng(seed)
    points = rng.random((n_points, 2))*amplitude

    return points


def generate_voronoi(points: ndarray) -> Voronoi:
    vor = Voronoi(points)

    return vor
