

from typing import Literal

from numpy import asarray, ndarray
from PIL import Image

from numpy import mean, max

ImageCompareOperator = Literal["sub", "sqr_mean", "xor"]


def open_image(path: str) -> Image:
    return Image.open(path)


def compare_images(img_a: ndarray, img_b: ndarray, operator: ImageCompareOperator = "sub") -> float:
    img_a = img_a/max(img_a)
    img_b = img_b/max(img_b)

    if operator == "sub":
        return mean(img_a - img_b)
    elif operator == "xor":
        return mean(img_a != img_b)
    elif operator == "sqr_mean":
        return mean((img_a - img_b) ** 2)

    raise ValueError("Unknown operator: {}".format(operator))


def compare_two_images_from_file(img_a_path: str, img_b_path: str, operator: ImageCompareOperator = "sub") -> float:
    img_a = asarray(open_image(img_a_path))
    img_b = asarray(open_image(img_b_path))

    return compare_images(img_a, img_b, operator)


def compare_two_images_from_memory(img_a: ndarray, img_b: ndarray, operator: ImageCompareOperator = "sub") -> float:
    return compare_images(img_a, img_b, operator)
