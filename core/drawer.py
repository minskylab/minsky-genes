# import matplotlib.pyplot as plt
# import numpy as np
from time import time
from typing import Dict, List, Tuple

from matplotlib.cm import get_cmap
from matplotlib.pyplot import figure, fill, hist, imsave, imshow, plot, savefig
from numpy import arange, dot, ndarray, asarray
from PIL import Image, ImageDraw
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from core.calculator import polygons_overlapped
from shape.image import get_image_array, get_processed_input_shape
from voronoi.generator import generate_voronoi_from_points


def save_voronoi_as_image(voronoi_diagram: Voronoi, image_path: str = "voronoi.png"):
    voronoi_plot_2d(voronoi_diagram, show_points=False, show_vertices=False)
    savefig(image_path)


def save_voronoi_with_selected_polygons_as_image(voronoi_diagram: Voronoi, polygons: List[Polygon], image_path: str = "filled_voronoi.png"):
    voronoi_plot_2d(voronoi_diagram, show_points=False, show_vertices=False)

    for polygon in polygons:
        polygon_x: List[float] = []
        polygon_y: List[float] = []

        for x, y in polygon.exterior.coords:
            polygon_x.append(x)
            polygon_y.append(y)

        fill(polygon_x, polygon_y, color=(0, 0, 0, 1))

    savefig(image_path)


def polygons_as_image(size: Tuple[int, int], polygons: List[Polygon]) -> Image:
    image = Image.new('RGB', size)

    draw = ImageDraw.Draw(image)

    rect = ((0, 0), size)
    draw.rectangle(rect, fill=(255, 255, 255))

    for polygon in polygons:
        coords = list(map(tuple, polygon.exterior.coords))
        # print(f"coords: {len(coords)}")

        if len(coords) < 1:
            # print(polygon)
            continue

        draw.polygon(coords, fill=(0, 0, 0))

    return image


def polygons_as_shape(size: Tuple[int, int], polygons: List[Polygon]) -> ndarray:
    image = asarray(polygons_as_image(size, polygons))

    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = dot(image[..., :3], rgb_weights)

    return grayscale_image


def save_polygons_as_image(size: Tuple[int, int], polygons: List[Polygon], image_path: str = "polygons.png"):
    image = polygons_as_image(size, polygons)

    image.save(image_path)
