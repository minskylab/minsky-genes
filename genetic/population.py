from typing import List, Optional, Tuple

from core.calculator import polygons_overlapped
from numpy import max, min, ndarray
from numpy.random import Generator, default_rng
from scipy.spatial.qhull import Voronoi
from shapely.geometry.polygon import Polygon
from tqdm import trange
from voronoi.generator import generate_voronoi_from_points, uniform_random_points

from genetic.genotype import Genotype
from genetic.meta import Metaparameters
from genetic.phenotype import calculate_phenotype, genotype_to_phenotype
from genetic.utils import map_float

Population = List[Genotype]
Phenotype = float
Fitness = float


def new_genotype(n_points: int, input_shape: ndarray, rng: Generator) -> Genotype:
    points = uniform_random_points(n_points=n_points, amplitude=input_shape.shape[0])
    threshold = rng.random()*max(points) - 1  # random number between 0 and the total of points - 1

    return Genotype(points, threshold)


def new_random_population(n_individuals: int, m_points: int, input_shape: ndarray) -> Population:
    rng = default_rng()

    population = [new_genotype(m_points, input_shape, rng) for _ in range(n_individuals)]

    return population


def calculate_population_phenotypes(population: Population, input_shape: ndarray) -> List[Phenotype]:
    n_individuals = len(population)

    phenotypes: List[float] = [0.0 for _ in range(n_individuals)]

    for i in trange(n_individuals):
        phenotype = genotype_to_phenotype(population[i], input_shape)
        phenotypes[i] = phenotype

    return phenotypes


# def calculate_population_metrics(meta: Metaparameters, population: Population) -> List[Tuple[Voronoi, List[Polygon], Phenotype, Fitness]]:
def calculate_population_metrics(meta: Metaparameters, population: Population) -> Tuple[List[Optional[Voronoi]], List[List[Polygon]], List[Phenotype], List[Fitness]]:
    n_individuals = len(population)

    phenotypes: List[Phenotype] = [0.0 for _ in range(n_individuals)]
    fitnesses: List[Fitness] = [0.0 for _ in range(n_individuals)]
    voronois: List[Optional[Voronoi]] = [None for _ in range(n_individuals)]
    polygons: List[Polygon] = [[] for _ in range(n_individuals)]

    for i in trange(n_individuals):
        threshold = int(population[i].selected_points_threshold)
        voronoi_diagram = generate_voronoi_from_points(population[i].points)
        selected_polygons = polygons_overlapped(voronoi_diagram, meta.input_shape, threshold)

        voronois[i] = voronoi_diagram
        polygons[i] = selected_polygons
        phenotypes[i] = calculate_phenotype(meta.input_shape, selected_polygons)

    weak_individual = 1 - max(phenotypes)
    better_individual = 1 - min(phenotypes)

    for i in range(n_individuals):
        prob_to_survive = 1 - phenotypes[i]
        prob_to_survive = map_float(prob_to_survive, weak_individual, better_individual, meta.alpha, meta.beta)

        fitnesses[i] = prob_to_survive

    return voronois, polygons, phenotypes, fitnesses


def calculate_population_fitness(population: Population, input_shape: ndarray, alpha: float = 0.05, beta: float = 0.9) -> List[Fitness]:
    n_individuals = len(population)
    phenotypes = calculate_population_phenotypes(population, input_shape)

    # rng = default_rng()

    weak_individual = 1 - max(phenotypes)
    better_individual = 1 - min(phenotypes)

    fitness: List[float] = [0.0 for _ in range(n_individuals)]

    for i in trange(n_individuals):
        prob_to_survive = 1 - phenotypes[i]
        prob_to_survive = map_float(prob_to_survive, weak_individual, better_individual, alpha, beta)

        fitness[i] = prob_to_survive

    return fitness
