from core.drawer import save_voronoi_with_selected_polygons_as_image
from genetic.algorithm import Metaparameters, new_genetic_algorithm
from genetic.evolve import evolve_contextualized_population, evolve_population
from numpy import ndarray, mean, max, min
from numpy.core.fromnumeric import argmax
from numpy.random import default_rng

from genetic.population import calculate_population_fitness, new_random_population
from genetic.selection import best_individuals_of_population, survive_population
from shape.image import get_processed_input_shape

input_shape: ndarray = get_processed_input_shape("assets/minsky_low_small.png")
input_shape = input_shape[0:-1, 0:-1]


# rng = default_rng()

# m_points = 30
# n_individuals = 20
# total_iterations = 100

# alpha = 0.4
# beta = 0.9

# total_crossvers = 4
# mutation_probability = 0.01

# translate_points_percent = 0.1
# threshold_mutate_ratio = 0.3


meta = Metaparameters(
    input_shape=input_shape,

    n_individuals=50,
    m_points=30,
    total_iterations=100,

    total_crossvers=20,
    mutation_probability=0.1,

    alpha=0.4,
    beta=0.9,

    translate_points_percent=0.4,
    threshold_mutate_ratio=0.3,

    elite_ranking=20
)


ga = new_genetic_algorithm(meta)
# pop = new_random_population(n_individuals, m_points, input_shape)
# print(calculate_population_fitness(pop, input_shape, alpha, beta))
# print(pop)


# bests = best_individuals_of_population(pop, input_shape, 10, alpha, beta)
# print(calculate_population_fitness(bests, input_shape, alpha, beta))
# print(bests)


for i in range(meta.total_iterations):
    ga = evolve_contextualized_population(ga)
    # new_pop = survive_population(pop, input_shape, alpha, beta)

    # # fitness = calculate_population_fitness(new_pop, input_shape, alpha, beta)
    # # print(mean(fitness))

    # new_pop = evolve_population(new_pop, input_shape, alpha, beta,
    #                             total_crossvers, mutation_probability,
    #                             translate_points_percent, threshold_mutate_ratio)

    # # print(new_pop)
    # fitness = calculate_population_fitness(new_pop, input_shape, alpha, beta)
    best_ind_index = argmax(ga.populations[-1].fitnesses)

    best_threshold = [gen.selected_points_threshold for gen in ga.populations[-1].population][best_ind_index]
    best_phenotype = ga.populations[-1].phenotypes[best_ind_index]
    best_voronoi = ga.populations[-1].voronoi[best_ind_index]
    best_polygons = ga.populations[-1].polygons[best_ind_index]

    image_name = f"vor-{i}-{best_phenotype:.3f}-{best_threshold}.png"

    save_voronoi_with_selected_polygons_as_image(best_voronoi, best_polygons, image_name)

    print(f"[generation {len(ga.populations)-1}] | mean fitness: {mean(ga.populations[-1].fitnesses)} | min phenotype: {min(ga.populations[-1].phenotypes)}")

# print()

# new_pop = select_survivals(pop, input_shape, alpha, beta)
# print(len(new_pop))
# print(calculate_population_fitness(new_pop, input_shape, alpha, beta))
