from numpy import asarray, concatenate, ndarray, zeros, zeros_like
import taichi as ti
from scipy.signal import convolve

N = 256

ti.init(arch=ti.gpu)

input_path = "public/minsky_low.png"
img = ti.imread(input_path)


@ti.data_oriented
class GeneralizedCellularAutomaton:
    def __init__(self, space: ndarray, neighborhood: ndarray, rule: ti.func):
        self.space = ti.field(ti.f32, shape=space.shape)
        self.kernel = ti.field(ti.f32, shape=neighborhood.shape)
        self.rule = rule

        self.space.from_numpy(space)
        self.kernel.from_numpy(neighborhood)

    @ti.kernel
    def step(self):
        # ti.field(ti.f32, shape=self.space.shape)
        for i, j, k in self.space:
            k_w, k_h, k_d = self.kernel.shape

            offset_i = (i-k_w//2)  # % self.space.shape[0]
            offset_j = (j-k_h//2)  # % self.space.shape[1]
            offset_k = (k-k_d//2)  # % self.space.shape[2]

            # perceived_slice = self.space[offset_i:offset_i+k_w, offset_j:offset_j+k_h, 1] * self.kernel

            # self.space[i, j, k] = self.rule(self.space[i, j, k], sum(perceived_slice))

            neighbors_sum = 0.0
            for x, y, z in ti.ndrange(k_w, k_h, k_d):
                neighbors_sum += self.space[offset_i+x, offset_j+y, offset_k+z] * self.kernel[x, y, z]

            # print(neighbors_sum)
            self.space[i, j, k] = self.rule(self.space[i, j, k], int(neighbors_sum))

        # new_ca = zeros_like(self.ca)
        # for i in range(1, N - 1):
        #     for j in range(1, N - 1):
        #         for k in range(16):
        #             neighbors = self.ca[i - 1:i + 2, j - 1:j + 2, k]
        #             new_ca[i, j, k] = self.kernel(neighbors)
        # self.ca = new_ca
        # model related
        # x = ti.field(ti.f32, shape=(1, 3, 3, 16))
        # w = ti.field(ti.f32, shape=(3, 16, 128))

        # @ti.kernel
        # def compute_y():
        #     ti.Matrix()

        # @ti.kernel
        # def init():
        #     for i, j, k in ca:
        #         ca[i, j, k] = ca[i, j, k]/255

        # def perceive(neighborhood: ndarray):
        #     sobel_x = asarray([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] for i in range(16)])
        #     sobel_y = sobel_x.T

        #     # print(sobel_x.shape)
        #     grad_x = convolve(neighborhood, sobel_x,  mode="same")
        #     grad_y = convolve(neighborhood, sobel_y,  mode="same")

        #     perception_vector = concatenate((neighborhood, grad_x, grad_y), axis=2)

        #     # print(perception_vector.shape)

        #     return perception_vector

        # def update(perception_vector):  # (256, 256, 48)
        #     # print()
        #     W = zeros(shape=(128, 48))

        #     cell = perception_vector[0, 0, :]
        #     res = W * cell

        #     print(res.shape)
#         #     pass

# gui = ti.GUI("Experiments", res=(N, N))

# init_state = zeros_like(ca.to_numpy())
# init_state[:, :, :4] = img

#         # ca.from_numpy(init_state)

#     # init()

#         # # p_vector = perceive(ca)

#         # # update(p_vector)

# for i in range(1000000):
#     # update(i)
#     gui.set_image(ca.to_numpy()[:, :, :4])
#     gui.show()
