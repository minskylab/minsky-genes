from numpy import ndarray
from numpy.random import default_rng

from genetic.population import calculate_population_fitness, new_random_population
from genetic.selection import best_individuals_of_population, survive_population
from shape.image import get_processed_input_shape

input_shape: ndarray = get_processed_input_shape("assets/minsky_low_small.png")
input_shape = input_shape[0:-1, 0:-1]


rng = default_rng()

m_points = 30
n_individuals = 20

alpha = 0.6
beta = 0.9

total_crossvers = 10
total_iterations = 100

pop = new_random_population(n_individuals, m_points, input_shape)
print(calculate_population_fitness(pop, input_shape, alpha, beta))
# print(pop)

bests = best_individuals_of_population(pop, input_shape, 10, alpha, beta)
print(calculate_population_fitness(bests, input_shape, alpha, beta))
# print(bests)

new_pop = survive_population(pop, input_shape, alpha, beta)

# new_pop = select_survivals(pop, input_shape, alpha, beta)
# print(len(new_pop))
# print(calculate_population_fitness(new_pop, input_shape, alpha, beta))
