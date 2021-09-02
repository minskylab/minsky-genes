from genetic.population import Population

from genetic.population import calculate_population_fitness, new_random_population
from genetic.selection import best_individuals_of_population, survive_population
from numpy import ndarray
from tqdm.std import trange


def evolve_population(pop: Population, input_shape: ndarray, alpha: float = 0.05, beta: float = 0.9, total_crossvers: int = 10) -> Population:
    bests = best_individuals_of_population(pop, input_shape, 10, alpha, beta)
    new_pop = survive_population(pop, input_shape, alpha, beta)

    for i in trange(total_crossvers):
        # TODO: Implement it
        pass

    return bests
