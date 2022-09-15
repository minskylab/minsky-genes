import numpy as np
import taichi as ti

from experiments.experiment import GeneralizedCellularAutomaton


@ti.func
def basic_rule(s, n):
    return 1.0 if n == 3.0 else (s if n == 2.0 else 0.0)


def entrypoint():
    W, H, D = 1024, 1024, 1

    gui = ti.GUI('Window Title', (W, H))

    kernel = np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
    space = np.around(np.random.uniform(size=(W, H, D)))

    ca = GeneralizedCellularAutomaton(space=space, neighborhood=kernel, rule=basic_rule)

    while gui.running:
        if gui.get_event(ti.GUI.SPACE):
            ca.step()
        # ca.step()
        gui.set_image(ca.space)
        gui.show()


if __name__ == "__main__":
    entrypoint()
