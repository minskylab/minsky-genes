from genetic.individual import best_individual_of_population, contextualized_population_to_dataframe
from core.drawer import polygons_as_image, save_polygons_as_image, save_voronoi_with_selected_polygons_as_image
from genetic.algorithm import Metaparameters, new_genetic_algorithm
from genetic.evolve import evolve_contextualized_population
from numpy import ndarray, mean, min

from shape.image import get_processed_input_shape

input_shape: ndarray = get_processed_input_shape("assets/minsky_low.png")
input_shape = input_shape[0:-1, 0:-1]


meta = Metaparameters(
    input_shape=input_shape,

    n_individuals=100,
    m_points=36,
    total_iterations=100,

    total_crossvers=80,
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

    ind = best_individual_of_population(ga.populations[-1])

    df = contextualized_population_to_dataframe(ga.populations[-1])
    print(df.head()["fitness"])

    image_name = f"vor-{i}-{ind.phenotype:.3f}-{ind.genotype.selected_points_threshold}.png"

    save_voronoi_with_selected_polygons_as_image(ind.voronoi_diagram, ind.selected_polygons, image_name)

    # save_polygons_as_image((400, 400), ind.selected_polygons, image_name, resize_polygons=False)

    print(f"[generation {len(ga.populations)-1}] | mean fitness: {mean(ga.populations[-1].fitnesses)} | min phenotype: {min(ga.populations[-1].phenotypes)}")
