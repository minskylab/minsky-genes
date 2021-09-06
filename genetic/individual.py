from dataclasses import dataclass
from typing import List, Optional

from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from genetic.genotype import Genotype


@dataclass
class Individual:
    genotype: Genotype
    phenotype: Optional[float] = None

    fitness: Optional[float] = None

    voronoi_diagram: Optional[Voronoi] = None
    selected_polygons: Optional[List[Polygon]] = None
