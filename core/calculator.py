

from typing import List

from numpy import arange, ndarray
from scipy.spatial import Voronoi
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon


def polygons_overlapped(voronoi_diagram: Voronoi, input_shape: ndarray, points_threshold: int = 1) -> List[Polygon]:
    valid_regions: List[Polygon] = []

    for region in voronoi_diagram.regions:
        if -1 not in region:
            reg = map(tuple, voronoi_diagram.vertices[region])
            valid_regions.append(Polygon(reg))

    points_inside: List[int] = [0] * len(valid_regions)

    for i in arange(input_shape.shape[0]):
        for j in arange(input_shape.shape[1]):
            if input_shape[j, i] < 255:
                for n, reg in enumerate(valid_regions):
                    if reg.contains(Point(i, j)):
                        points_inside[n] += 1

    selected_polygons: List[Polygon] = []
    for i, total_points in enumerate(points_inside):
        if total_points > points_threshold:
            selected_polygons.append(valid_regions[i])

    return selected_polygons

    # print(f"Time taken: {time() - start_time}")

    # print(valid_regions)
    # print(points_inside)
    # for r_index in points_inside_regions:
    # voronoi_diagram.regions[r_index]

    # print(voronoi_diagram.regions[r_index])

    # ax = hist(points_inside)
    # savefig("histogram.png")

    # print(f"Total of selected polygons: {len(selected_polygons)}")
    # # print(f"{i}: {points_inside[i]}")

    # fig = imshow(input_shape, cmap=get_cmap("gray"))

    # voronoi_plot_2d(voronoi_diagram, show_points=False, show_vertices=False)
    # savefig("output.png")

    # # print(list(map(tuple, selected_polygons[0].exterior.coords)))

    # image = Image.new('RGB', (input_shape.shape[0], input_shape.shape[1]))

    # draw = ImageDraw.Draw(image)
    # draw.rectangle(
    #     ((0, 0), (input_shape.shape[0], input_shape.shape[1])), fill=(255, 255, 255))
    # for polygon in selected_polygons:
    #     draw.polygon(list(map(tuple, polygon.exterior.coords)), fill=(0, 0, 0))

    # image.save("view.png")
