from PIL import Image
from numpy import asarray, dot, ndarray


def open_image(path: str) -> Image:
    return Image.open(path)


def get_image_array(path: str) -> ndarray:
    image = asarray(open_image(path))

    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = dot(image[..., :3], rgb_weights)

    return grayscale_image


def get_processed_input_shape(path: str, threshold: int = 127) -> ndarray:
    img = get_image_array(path)

    img[img > threshold] = 255
    img[img <= threshold] = 0

    return img
