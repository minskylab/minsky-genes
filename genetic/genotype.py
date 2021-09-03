from dataclasses import dataclass
from typing import Tuple

from numpy import max, ndarray
from numpy.random import default_rng


@dataclass
class Genotype:
    points: ndarray
    selected_points_threshold: int


def mutate_genotype(genotype: Genotype, translate_points_percent: float = 0.1, threshold_mutate_ratio: float = 0.3) -> Genotype:
    rng = default_rng()

    max_point = max(genotype.points)

    offset_points = rng.random(genotype.points.shape) * translate_points_percent*max_point

    genotype.points += offset_points

    genotype.points[genotype.points > max_point] = max_point
    genotype.points[genotype.points < 0] = 0

    t_offset = genotype.selected_points_threshold * ((rng.random()*2)-1) * threshold_mutate_ratio

    genotype.selected_points_threshold += t_offset

    return genotype


def crossover_genotypes(genotype_a: Genotype, genotype_b: Genotype) -> Tuple[Genotype, Genotype]:
    rng = default_rng()

    points_1 = rng.random(genotype_a.points.shape)
    aux_points_1 = points_1

    points_2 = rng.random(genotype_b.points.shape)
    aux_points_2 = points_2

    points_1[aux_points_1 < 0.5] = genotype_a.points[aux_points_1 < 0.5]
    points_1[aux_points_1 >= 0.5] = genotype_b.points[aux_points_1 >= 0.5]

    points_2[aux_points_2 < 0.5] = genotype_a.points[aux_points_2 < 0.5]
    points_2[aux_points_2 >= 0.5] = genotype_b.points[aux_points_2 >= 0.5]

    threshold_1 = rng.random() * (genotype_a.selected_points_threshold - genotype_b.selected_points_threshold) + \
        genotype_b.selected_points_threshold
    threshold_2 = rng.random() * (genotype_a.selected_points_threshold - genotype_b.selected_points_threshold) + \
        genotype_a.selected_points_threshold

    print(len(points_1), len(points_2))

    child_1 = Genotype(points_1, threshold_1)
    child_2 = Genotype(points_2, threshold_2)

    return child_1, child_2
