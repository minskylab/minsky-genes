from dataclasses import dataclass
from typing import List

from numpy import concatenate, ndarray
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from genetic.genotype import Genotype
from genetic.meta import Metaparameters
from genetic.population import Phenotype, Population, calculate_population_metrics, new_random_population


@dataclass
class ContextualizedPopulation:
    population: List[Genotype]
    phenotypes: List[Phenotype]
    fitnesses: List[float]
    voronoi: List[Voronoi]
    polygons: List[List[Polygon]]

    hash: int


@dataclass
class GeneticAlgorithmContext:
    meta: Metaparameters
    populations: List[ContextualizedPopulation]


def inflate_population(meta: Metaparameters, population: Population) -> ContextualizedPopulation:
    points: ndarray = concatenate([pop.points for pop in population], axis=0)

    voronois, polygons, phenotypes, fitnesses = calculate_population_metrics(meta, population)

    return ContextualizedPopulation(
        population=population,
        phenotypes=phenotypes,
        fitnesses=fitnesses,
        voronoi=voronois,
        polygons=polygons,
        hash=hash(points.tobytes()),
        # polygons=
    )


def new_default_genetic_algorithm(input_shape: ndarray) -> GeneticAlgorithmContext:
    meta = Metaparameters(input_shape=input_shape)
    pop = new_random_population(meta.n_individuals, meta.m_points, input_shape)
    ctx_pop = inflate_population(meta, pop)

    return GeneticAlgorithmContext(
        meta=meta,
        populations=[ctx_pop],
    )


def new_genetic_algorithm(meta: Metaparameters) -> GeneticAlgorithmContext:
    pop = new_random_population(meta.n_individuals, meta.m_points, meta.input_shape)
    ctx_pop = inflate_population(meta, pop)

    return GeneticAlgorithmContext(
        meta=meta,
        populations=[ctx_pop],
    )
