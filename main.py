from core.diff import compare_two_images_from_memory
from core.drawer import polygons_as_shape, save_polygons_as_image, save_voronoi_as_image, save_voronoi_with_selected_polygons_as_image
from typing import List

from matplotlib.pyplot import hist, savefig
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

from core.calculator import polygons_overlapped
from shape.image import get_processed_input_shape
from voronoi.generator import generate_voronoi

# rng = np.random.default_rng()
# points = rng.random((100, 2))

# vor = Voronoi(points)

# print(f"total vertices: {len(vor.vertices)}")
# # for r in vor.regions:
# #     print(r)

# fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False)
# plt.savefig("output.png")


input_shape = get_processed_input_shape("minsky_low_small.png")
input_shape = input_shape[0:-1, 0:-1]


diffs: List[float] = []

best_polygons: List[Polygon] = []
best_voronoi: Voronoi = None

min_diff = 1.0

for i in range(5000):
    voronoi_diagram = generate_voronoi(input_shape.shape[0], 40)

    # save_voronoi_as_image(voronoi_diagram, "voronoi_diagram.png")

    selected_polygons = polygons_overlapped(voronoi_diagram, input_shape)
    result_shape = polygons_as_shape(input_shape.shape, selected_polygons)

    # save_polygons_as_image(input_shape.shape, selected_polygons)

    diff = compare_two_images_from_memory(input_shape, result_shape, operator="sqr_mean")

    diffs.append(diff)

    if diff < min_diff:
        best_polygons = selected_polygons
        best_voronoi = voronoi_diagram
        min_diff = diff

# diff = compare_two_images("minsky_low_small.png", "selected_polygons.png")

print(diffs)
print(min_diff)

hist(diffs)
savefig("errors_hist.png")

save_polygons_as_image(input_shape.shape, best_polygons)
save_voronoi_with_selected_polygons_as_image(best_voronoi, best_polygons)

# imsave("shape.png", input_shape, cmap=get_cmap("gray"))

# print(input_shape.shape)

# start_time = time()

# valid_regions: List[Polygon] = []

# for region in voronoi_diagram.regions:
#     if -1 not in region:
#         reg = map(tuple, voronoi_diagram.vertices[region])
#         valid_regions.append(Polygon(reg))


# points_inside: List[int] = [0] * len(valid_regions)

# for i in arange(input_shape.shape[0]):
#     for j in arange(input_shape.shape[1]):
#         if input_shape[j, i] < 255:
#             for n, reg in enumerate(valid_regions):
#                 if reg.contains(Point(i, j)):
#                     points_inside[n] += 1

# print(f"Time taken: {time() - start_time}")


# # print(valid_regions)
# print(points_inside)
# # for r_index in points_inside_regions:
# # voronoi_diagram.regions[r_index]

# # print(voronoi_diagram.regions[r_index])

# ax = hist(points_inside)
# savefig("histogram.png")

# selected_polygons: List[Polygon] = []
# for i, total_points in enumerate(points_inside):
#     if total_points > 1:
#         selected_polygons.append(valid_regions[i])

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
# #     polygon_x: List[float] = []
# #     polygon_y: List[float] = []

# #     for x, y in polygon.exterior.coords:
# #         polygon_x.append(x)
# #         polygon_y.append(y)

# #     fill(polygon_x, polygon_y, color=(0, 0, 0, 1))

# # savefig("insiders.png")

# # print(selected_polygons[0].exterior.coords.xy)
