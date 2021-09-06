from dataclasses import dataclass
from typing import List
from numpy import concatenate, ndarray

from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from genetic.genotype import Genotype
from genetic.population import Population, calculate_population_fitness


# class Individual:
#     genotype: Genotype
#     fitness


@dataclass
class Metaparameters:
    input_shape: ndarray

    n_individuals: int = 20
    m_points: int = 30
    total_iterations: int = 100

    total_crossvers: int = 4
    mutation_probability: float = 0.01

    alpha: float = 0.4
    beta: float = 0.9

    translate_points_percent: float = 0.1
    threshold_mutate_ratio: float = 0.3


@dataclass
class ContextualizedPopulation:
    population: List[Genotype]
    fitnesses: List[float]
    voronoi: List[Voronoi]
    polygons: List[List[Polygon]]

    hash: int


@dataclass
class GeneticAlgorithmContext:
    populations: List[ContextualizedPopulation]


def inflate_population(population: Population, params: Metaparameters) -> ContextualizedPopulation:
    points: ndarray = concatenate([pop.points for pop in population], axis=0)

    fitnesses = calculate_population_fitness(population, params.input_shape, params.alpha, params.beta)

    return ContextualizedPopulation(
        population=population,
        hash=hash(points.data),
        fitnesses=fitnesses,
        polygons=)
