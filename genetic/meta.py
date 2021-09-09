from dataclasses import dataclass

from numpy import ndarray


@dataclass
class Metaparameters:
    input_shape: ndarray

    n_individuals: int = 20
    m_points: int = 30
    total_iterations: int = 100

    total_crossvers: int = 4
    mutation_probability: float = 0.01

    alpha: float = 0.4
    beta: float = 0.9

    translate_points_percent: float = 0.1
    threshold_mutate_ratio: float = 0.3

    elite_ranking: int = 3
