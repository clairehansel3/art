import art
import numpy as np

def foo(x, y):
    return np.exp(-10 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

x_min = y_min = -1
x_max = y_max = 1
x_num = 1000
y_num = 1000
t_max = 0.1
t_num = 10

x_step = (x_max - x_min) / x_num
y_step = (y_max - y_min) / y_num
i = np.arange(x_num)
j = np.arange(y_num)
x = x_min + x_step * (0.5 + i)
y = y_min + y_step * (0.5 + j)
x_mesh, y_mesh = np.meshgrid(x, y)
foo_mesh = foo(x_mesh, y_mesh)

sol = art.solve_homogeneous_heat_equation_t_values(x_step, y_step, foo_mesh, t_max, t_num)

image = art.save_video(sol, 'vid.mp4')
