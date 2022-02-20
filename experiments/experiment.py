from numpy import asarray, concatenate, ndarray, zeros, zeros_like
import taichi as ti
from scipy.signal import convolve

N = 256

ti.init(arch=ti.gpu)

input_path = "public/minsky_low.png"
img = ti.imread(input_path)


ca = ti.field(ti.f32, shape=(N, N, 16))

# model related
x = ti.field(ti.f32, shape=(1, 3, 3, 16))
w = ti.field(ti.f32, shape=(3, 16, 128))


@ti.kernel
def compute_y():
    ti.Matrix()


@ti.kernel
def init():
    for i, j, k in ca:
        ca[i, j, k] = ca[i, j, k]/255


def perceive(neighborhood: ndarray):
    sobel_x = asarray([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] for i in range(16)])
    sobel_y = sobel_x.T

    # print(sobel_x.shape)
    grad_x = convolve(neighborhood, sobel_x,  mode="same")
    grad_y = convolve(neighborhood, sobel_y,  mode="same")

    perception_vector = concatenate((neighborhood, grad_x, grad_y), axis=2)

    # print(perception_vector.shape)

    return perception_vector


def update(perception_vector):  # (256, 256, 48)
    # print()
    W = zeros(shape=(128, 48))

    cell = perception_vector[0, 0, :]
    res = W * cell

    print(res.shape)
    pass


gui = ti.GUI("Experiments", res=(N, N))


init_state = zeros_like(ca.to_numpy())
init_state[:, :, :4] = img

ca.from_numpy(init_state)


init()

# p_vector = perceive(ca)

# update(p_vector)

for i in range(1000000):
    # update(i)
    gui.set_image(ca.to_numpy()[:, :, :4])
    gui.show()
