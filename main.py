from genetic.evolve import evolve_population
from numpy import ndarray, mean
from numpy.random import default_rng

from genetic.population import calculate_population_fitness, new_random_population
from genetic.selection import best_individuals_of_population, survive_population
from shape.image import get_processed_input_shape

input_shape: ndarray = get_processed_input_shape("assets/minsky_low_small.png")
input_shape = input_shape[0:-1, 0:-1]


rng = default_rng()

m_points = 30
n_individuals = 20
total_iterations = 100

alpha = 0.4
beta = 0.9

total_crossvers = 4
mutation_probability = 0.01

translate_points_percent = 0.1
threshold_mutate_ratio = 0.3

pop = new_random_population(n_individuals, m_points, input_shape)
# print(calculate_population_fitness(pop, input_shape, alpha, beta))
# print(pop)


# bests = best_individuals_of_population(pop, input_shape, 10, alpha, beta)
# print(calculate_population_fitness(bests, input_shape, alpha, beta))
# print(bests)


for i in range(total_iterations):
    new_pop = survive_population(pop, input_shape, alpha, beta)

    # fitness = calculate_population_fitness(new_pop, input_shape, alpha, beta)
    # print(mean(fitness))

    new_pop = evolve_population(new_pop, input_shape, alpha, beta,
                                total_crossvers, mutation_probability,
                                translate_points_percent, threshold_mutate_ratio)

    # print(new_pop)
    fitness = calculate_population_fitness(new_pop, input_shape, alpha, beta)
    print(mean(fitness))

# print()

# new_pop = select_survivals(pop, input_shape, alpha, beta)
# print(len(new_pop))
# print(calculate_population_fitness(new_pop, input_shape, alpha, beta))
