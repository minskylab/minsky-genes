from dataclasses import dataclass
from typing import List, Optional

from numpy import argmax
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from genetic.algorithm import ContextualizedPopulation
from genetic.genotype import Genotype
from pandas import DataFrame


@dataclass
class Individual:
    genotype: Genotype
    phenotype: float

    fitness: float

    voronoi_diagram: Voronoi
    selected_polygons: List[Polygon]


def best_individual_of_population(pop: ContextualizedPopulation) -> Individual:
    best_individual_index = argmax(pop.fitnesses)
    pop.population[best_individual_index]

    best_genotype = pop.population[best_individual_index]
    best_fitness = pop.fitnesses[best_individual_index]
    best_phenotype = pop.phenotypes[best_individual_index]
    best_voronoi = pop.voronoi[best_individual_index]
    best_polygons = pop.polygons[best_individual_index]

    return Individual(
        genotype=best_genotype,
        phenotype=best_phenotype,

        fitness=best_fitness,

        voronoi_diagram=best_voronoi,
        selected_polygons=best_polygons,
    )


def contextualized_population_to_individuals(pop: ContextualizedPopulation) -> List[Individual]:
    individuals = []

    for i in range(len(pop.population)):
        genotype = pop.population[i]
        fitness = pop.fitnesses[i]
        phenotype = pop.phenotypes[i]
        voronoi = pop.voronoi[i]
        polygons = pop.polygons[i]

        individuals.append(Individual(
            genotype=genotype,
            phenotype=phenotype,

            fitness=fitness,

            voronoi_diagram=voronoi,
            selected_polygons=polygons,
        ))

    return individuals


def contextualized_population_to_dataframe(pop: ContextualizedPopulation) -> DataFrame:
    data = contextualized_population_to_individuals(pop)

    return DataFrame(data)
