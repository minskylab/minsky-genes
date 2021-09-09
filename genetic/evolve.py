from genetic.algorithm import GeneticAlgorithmContext, inflate_population
from genetic.genotype import crossover_genotypes, mutate_genotype
from genetic.population import Population

from genetic.population import calculate_population_fitness, new_random_population
from genetic.selection import best_individuals_of_fitnesses, best_individuals_of_population, survive_contextualized_population, survive_population
from numpy import ndarray
from tqdm.std import trange
from numpy.random import default_rng


def evolve_population(pop: Population, input_shape: ndarray, alpha: float = 0.05, beta: float = 0.9, total_crossvers: int = 10, mutation_probability: float = 0.05, translate_points_percent: float = 0.1, threshold_mutate_ratio: float = 0.3) -> Population:
    rng = default_rng()
    new_pop = survive_population(pop, input_shape, alpha, beta)

    for _ in trange(total_crossvers):
        idx_1 = int(rng.random() * len(new_pop))
        idx_2 = int(rng.random() * len(new_pop))

        individual_1 = new_pop[idx_1]
        individual_2 = new_pop[idx_2]

        child_1, child_2 = crossover_genotypes(individual_1, individual_2)

        new_pop[idx_1] = child_1
        new_pop[idx_2] = child_2

    for i, individual in enumerate(new_pop):
        if rng.random() < mutation_probability:
            mutant = mutate_genotype(individual, translate_points_percent, threshold_mutate_ratio)
            new_pop[i] = mutant

    return new_pop


def evolve_contextualized_population(ga: GeneticAlgorithmContext) -> GeneticAlgorithmContext:
    if len(ga.populations) < 1:
        raise Exception("No population to survive")

    rng = default_rng()

    last_pop = ga.populations[-1]

    population_fragment: Population = []

    for i in range(len(last_pop.population)):
        fitness = last_pop.fitnesses[i]

        dart = rng.random()
        survive = dart < fitness

        if survive:
            population_fragment.append(last_pop.population[i])

        # print(f"{phenotypes[i]:.4f} | {prob_to_survive} | {dart} | {survive}")
    cream_of_cream = best_individuals_of_fitnesses(last_pop.population, last_pop.fitnesses, ga.meta.elite_ranking)

    j: int = 0

    while len(population_fragment) < len(last_pop.population):
        individual = cream_of_cream[j % len(cream_of_cream)]
        population_fragment.append(individual)
        j += 1

    for _ in trange(ga.meta.total_crossvers):
        idx_1 = int(rng.random() * len(population_fragment))
        idx_2 = int(rng.random() * len(population_fragment))

        individual_1 = population_fragment[idx_1]
        individual_2 = population_fragment[idx_2]

        child_1, child_2 = crossover_genotypes(individual_1, individual_2)

        population_fragment[idx_1] = child_1
        population_fragment[idx_2] = child_2

    for i, individual in enumerate(population_fragment):
        if rng.random() < ga.meta.mutation_probability:
            mutant = mutate_genotype(individual, ga.meta.translate_points_percent, ga.meta.threshold_mutate_ratio)
            population_fragment[i] = mutant

    new_pop = inflate_population(ga.meta, population_fragment)

    ga.populations.append(new_pop)

    return ga
