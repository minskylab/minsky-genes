from core.diff import open_image
from dataclasses import dataclass
from typing import List
from numpy import concatenate, ndarray

from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from genetic.genotype import Genotype
from genetic.population import Fitness, Phenotype, Population, calculate_population_fitness, calculate_population_metrics


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


def inflate_population(meta: Metaparameters, population: Population) -> ContextualizedPopulation:
    points: ndarray = concatenate([pop.points for pop in population], axis=0)

    metrics = calculate_population_metrics(meta, population)

    phenotypes: List[Phenotype] = [0.0 for _ in range(meta.n_individuals)]
    fitnesses: List[Fitness] = [0.0 for _ in range(meta.n_individuals)]
    voronois: List[Voronoi] = [Voronoi() for _ in range(meta.n_individuals)]
    polygons: List[List[Polygon]] = [[] for _ in range(meta.n_individuals)]

    for i, (voronoi, polygons, phenotype, fitness) in enumerate(metrics):
        phenotypes[i] = phenotype
        fitnesses[i] = fitness
        voronois[i] = voronoi
        polygons[i] = polygons

    # fitnesses = calculate_population_fitness(population, meta.input_shape, meta.alpha, meta.beta)

    return ContextualizedPopulation(
        population=population,
        fitnesses=fitnesses,
        voronoi=voronois,
        polygons=polygons,
        hash=hash(points.data),
        # polygons=
    )
