

from genetic.population import Population, calculate_population_fitness
from numpy import ndarray, asarray
from numpy.random import default_rng


def best_individuals_of_population(population: Population, input_shape: ndarray, how: int, alpha: float = 0.05, beta: float = 0.9):
    fintenesses = asarray(calculate_population_fitness(population, input_shape, alpha, beta))

    return asarray(population)[fintenesses.argsort()[how:0:-1]]


def select_survivals(population: Population, input_shape: ndarray, alpha: float = 0.05, beta: float = 0.9) -> Population:
    rng = default_rng()

    fintenesses = calculate_population_fitness(population, input_shape, alpha, beta)

    new_population: Population = []

    for i in range(len(population)):
        fitness = fintenesses[i]

        dart = rng.random()
        survive = dart < fitness

        if survive:
            new_population.append(population[i])

        # print(f"{phenotypes[i]:.4f} | {prob_to_survive} | {dart} | {survive}")

    return new_population


def evolve_population(population: Population, input_shape: ndarray, alpha: float = 0.05, beta: float = 0.9):
    population_fragment = select_survivals(population, input_shape, alpha, beta)

    while len(population_fragment) < len(population):
        population_fragment.append(population[len(population_fragment)])
