from genetic.genotype import Genotype
from core.diff import compare_two_images_from_memory
from core.drawer import polygons_as_shape
from voronoi.generator import generate_voronoi

from numpy import ndarray

from core.calculator import polygons_overlapped


def genotype_to_phenotype(genotype: Genotype, input_shape: ndarray) -> float:
    voronoi_diagram = generate_voronoi(genotype.points)
    selected_polygons = polygons_overlapped(voronoi_diagram, input_shape, points_threshold=int(genotype.threshold))

    result_shape = polygons_as_shape(input_shape.shape, selected_polygons)

    diff = compare_two_images_from_memory(input_shape, result_shape, operator="sqr_mean")

    return diff
