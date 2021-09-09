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


meta = Metaparameters(
    input_shape=input_shape,

    n_individuals=500,
    m_points=40,
    total_iterations=100,

    total_crossvers=50,
    mutation_probability=0.05,

    alpha=0.2,
    beta=0.9,

    translate_points_percent=0.4,
    threshold_mutate_ratio=0.3,

    elite_ranking=50
)


ga = new_genetic_algorithm(meta)


for i in range(meta.total_iterations):
    ga = evolve_contextualized_population(ga)

    best_individual_index = argmax(ga.populations[-1].fitnesses)

    best_genotype = ga.populations[-1].population[best_individual_index]
    best_threshold = ga.populations[-1].population[best_individual_index].selected_points_threshold
    best_phenotype = ga.populations[-1].phenotypes[best_individual_index]
    best_voronoi = ga.populations[-1].voronoi[best_individual_index]
    best_polygons = ga.populations[-1].polygons[best_individual_index]

    image_name = f"vor-{i}-{best_phenotype:.3f}-{best_threshold}.png"

    save_voronoi_with_selected_polygons_as_image(best_voronoi, best_polygons, image_name)

    print(f"[generation {len(ga.populations)-1}] | best total points: {len(best_genotype.points)} |mean fitness: {mean(ga.populations[-1].fitnesses)} | min phenotype: {min(ga.populations[-1].phenotypes)}")
