from typing import List, Optional

from numpy import max, min, ndarray
from numpy.random import Generator, default_rng
from tqdm import trange
from voronoi.generator import uniform_random_points

from genetic.genotype import Genotype
from genetic.phenotype import genotype_to_phenotype
from genetic.utils import map_float

Population = List[Genotype]


def new_individual(n_points: int, input_shape: ndarray, rng: Generator) -> Genotype:
    points = uniform_random_points(n_points=n_points, amplitude=input_shape.shape[0])
    threshold = rng.random()*max(points) - 1  # random number between 0 and the total of points - 1

    return Genotype(points, threshold)


def new_random_population(n_individuals: int, m_points: int, input_shape: ndarray) -> Population:
    rng = default_rng()

    population = [new_individual(m_points, input_shape, rng) for _ in range(n_individuals)]

    return population


def calculate_population_phenotypes(population: Population, input_shape: ndarray) -> List[float]:
    n_individuals = len(population)

    phenotypes: List[float] = [0.0 for _ in range(n_individuals)]

    for i in trange(n_individuals):
        if population[i] is not None:
            phenotypes[i] = genotype_to_phenotype(population[i], input_shape)

    return phenotypes


def calculate_population_fitness(population: Population, input_shape: ndarray, alpha: float = 0.05, beta: float = 0.9) -> List[float]:
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
