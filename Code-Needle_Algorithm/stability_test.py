from needle_utils import *
from Linear_equation import *
import numpy as np
import random

def stability_test():
    iterations = 20


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.xlabel("X")
    plt.ylabel("Z")
    # plt.zlabel("Y")

    # ax.set_xlim(-2, -1)
    # ax.set_zlim(10, 12)
    # ax.set_ylim(52, 56)

    color = ['red', 'green', 'blue', 'purple', 'orange']

    diff = 3
    num_steps = 3
    anchor = 1

    mtx, dist = camera_para_retrieve()
    coordinates = np.array([[506.49, 853.06, 0],
                            [479.72, 642.93, 0],
                            [439.34, 410.83, 0]], dtype='float64')
    board_coordinate = np.load("../Coordinate/board_coordinate.npy")
    tip_b = board_coordinate[anchor]
    print(tip_b)
    ax.scatter(tip_b[0], tip_b[2], tip_b[1], c='red', marker='*', s=30)

    coordinates_c = coordinates.copy()
    tip_s, end = scale_estimation_multi(coordinates_c[0], coordinates_c[1], coordinates_c[2], 50, 60, mtx, 2.2)
    error = error_calc_board(tip_s, anchor=anchor)
    print(f"s, pixediff = 0, error = {error:.2f}")
    ax.scatter(tip_s[0], tip_s[2], tip_s[1], c='black', marker='*', s=30)


    for i in range(iterations):
        coordinates_c = coordinates.copy()
        x_noise = random.uniform(-diff, diff)
        y_noise = random.uniform(-diff, diff)
        noise = math.sqrt(x_noise ** 2 + y_noise ** 2)

        pt = random.choice([0, 1, 2])
        coordinates_c[pt, 0] += x_noise
        coordinates_c[pt, 1] += y_noise
        tip, end = scale_estimation_multi(coordinates_c[0], coordinates_c[1], coordinates_c[2], 50, 60, mtx, 2.2)

        tip_diff = np.linalg.norm(tip_s - tip)
        tip_diff_vec = tip_s - tip
        tip_diff_vec = [round(x, 2) for x in tip_diff_vec]

        error = error_calc_board(tip, anchor=anchor)
        error_vec = error_vec_calc_board(tip, anchor)
        print(f"{i}, noise = {noise:.2f}, tip_diff = {tip_diff:.2f}, tip_diff_vec = {tip_diff_vec}, tip_error = {error:.2f}, error_vec = {error_vec}")

        # trajs = np.linspace(tip_s, tip, num=num_steps)
        # plt.plot(trajs[:, 0], trajs[:, 2], trajs[:, 1], color[i])


        # trajs = np.linspace(tip, tip_b, num=num_steps)
        # plt.plot(trajs[:, 0], trajs[:, 2], trajs[:, 1], color[i])

    # plt.show()
    # plt.close()


if __name__ == "__main__":
    stability_test()