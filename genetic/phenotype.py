from typing import List

from core.calculator import polygons_overlapped
from core.diff import compare_two_images_from_memory
from core.drawer import polygons_as_shape
from numpy import ndarray
from scipy.spatial.qhull import Voronoi
from shapely.geometry.polygon import Polygon
from voronoi.generator import generate_voronoi_from_points

from genetic.genotype import Genotype


def calculate_phenotype(input_shape: ndarray, selected_polygons: List[Polygon]) -> float:
    result_shape = polygons_as_shape(input_shape.shape, selected_polygons)
    diff = compare_two_images_from_memory(input_shape, result_shape, operator="sqr_mean")

    return diff


def genotype_to_phenotype(genotype: Genotype, input_shape: ndarray) -> float:
    voronoi_diagram = generate_voronoi_from_points(genotype.points)
    selected_polygons = polygons_overlapped(voronoi_diagram, input_shape, int(genotype.selected_points_threshold))

    return calculate_phenotype(input_shape, selected_polygons)
