from genetic.genotype import crossover_genotypes
from genetic.population import Population

from genetic.population import calculate_population_fitness, new_random_population
from genetic.selection import best_individuals_of_population, survive_population
from numpy import ndarray
from tqdm.std import trange
from numpy.random import default_rng


def evolve_population(pop: Population, input_shape: ndarray, alpha: float = 0.05, beta: float = 0.9, total_crossvers: int = 10) -> Population:
    rng = default_rng()
    # bests = best_individuals_of_population(pop, input_shape, 10, alpha, beta)
    new_pop = survive_population(pop, input_shape, alpha, beta)

    for i in trange(total_crossvers):
        # TODO: Implement it
        idx_1 = int(rng.random() * len(new_pop))
        idx_2 = int(rng.random() * len(new_pop))

        individual_1 = new_pop[idx_1]
        individual_2 = new_pop[idx_2]

        child_1, child_2 = crossover_genotypes(individual_1, individual_2)

        new_pop[idx_1] = child_1
        new_pop[idx_2] = child_2

    return new_pop
