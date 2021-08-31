from dataclasses import dataclass
from typing import Tuple

from numpy import max, ndarray
from numpy.random import default_rng


@dataclass
class Genotype:
    points: ndarray
    threshold: float


def mutate_genotype(genotype: Genotype, translate_points_percent: float = 0.1, threshold_mutate_ratio: float = 0.3) -> Genotype:
    rng = default_rng()

    max_point = max(genotype.points)

    offset_points = rng.random(genotype.points.shape) * translate_points_percent*max_point

    genotype.points += offset_points

    genotype.points[genotype.points > max_point] = max_point
    genotype.points[genotype.points < 0] = 0

    genotype.threshold *= ((rng.random()*2)-1) * threshold_mutate_ratio

    return genotype


def crossover_genotypes(genotype_a: Genotype, genotype_b: Genotype) -> Tuple[Genotype, Genotype]:
    rng = default_rng()

    points_1 = rng.random(genotype_a.points.shape)
    points_2 = rng.random(genotype_b.points.shape)

    points_1[points_1 < 0.5] = genotype_a.points[points_1 < 0.5]
    points_1[points_1 >= 0.5] = genotype_b.points[points_1 >= 0.5]

    points_2[points_2 < 0.5] = genotype_a.points[points_2 >= 0.5]
    points_2[points_2 >= 0.5] = genotype_b.points[points_2 < 0.5]

    threshold_1 = rng.random() * (genotype_a.threshold - genotype_b.threshold) + genotype_b.threshold
    threshold_2 = rng.random() * (genotype_a.threshold - genotype_b.threshold) + genotype_a.threshold

    child_1 = Genotype(points_1, threshold_1)
    child_2 = Genotype(points_2, threshold_2)

    return child_1, child_2
