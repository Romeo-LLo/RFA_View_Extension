import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([], [], [])[0] for _ in range(2)]

l = 2

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-l, l)
ax.set_ylim(-l, l)
ax.set_zlim(-l, l)

for k in range(500):
    start_pos = np.random.random(3)
    end_pos = np.random.random(3)
    start_pos_t = np.random.random(3)
    end_pos_t = np.random.random(3)

    num_steps = 15


    trajs = np.zeros((2, 15, 3))
    trajs[0] = np.linspace(start_pos_t, end_pos_t, num=num_steps)
    trajs[1] = np.linspace(start_pos, end_pos, num=num_steps)
    for line, traj in zip(lines, trajs):
        line.set_data(traj[:, :2].T)
        line.set_3d_properties(traj[:, 2])
    fig.canvas.draw()
    fig.canvas.flush_events()
