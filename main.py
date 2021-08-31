from core.diff import compare_two_images_from_memory
from core.drawer import polygons_as_shape, save_polygons_as_image, save_voronoi_as_image, save_voronoi_with_selected_polygons_as_image
from typing import List, Optional

from matplotlib.pyplot import hist, savefig
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from core.calculator import polygons_overlapped
from shape.image import get_processed_input_shape
from voronoi.generator import generate_voronoi, uniform_random_points


input_shape = get_processed_input_shape("assets/minsky_low_small.png")
input_shape = input_shape[0:-1, 0:-1]


diffs: List[float] = []

best_polygons: List[Polygon] = []
best_voronoi: Optional[Voronoi] = None

min_diff = 1.0

for i in range(1000):
    points = uniform_random_points(input_shape.shape[0], 40)
    voronoi_diagram = generate_voronoi(points)

    selected_polygons = polygons_overlapped(voronoi_diagram, input_shape)
    result_shape = polygons_as_shape(input_shape.shape, selected_polygons)

    diff = compare_two_images_from_memory(input_shape, result_shape, operator="sqr_mean")

    diffs.append(diff)

    if diff < min_diff:
        best_polygons = selected_polygons
        best_voronoi = voronoi_diagram
        min_diff = diff


print(diffs)
print(min_diff)

hist(diffs)
savefig("errors_hist.png")

save_polygons_as_image(input_shape.shape, best_polygons)
save_voronoi_with_selected_polygons_as_image(best_voronoi, best_polygons)
