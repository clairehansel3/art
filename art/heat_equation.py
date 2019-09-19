import numpy as np
import scipy.signal as sig

def solve_homogeneous_heat_equation_single_t(x_step, y_step, initial_mesh, t):
    if t == 0:
        return initial_mesh
    x_num, y_num = initial_mesh.shape
    value_1 = x_step * y_step / (4 * np.pi * t)
    value_2 = -(x_step ** 2) / (4 * t)
    value_3 = -(y_step ** 2) / (4 * t)
    i = np.arange(-x_num + 1, x_num).reshape(1, 2 * x_num - 1)
    j = np.arange(-y_num + 1, y_num).reshape(2 * y_num - 1, 1)
    kernel = value_1 * np.exp(value_2 * (i ** 2) + value_3 * (j ** 2))
    return sig.convolve(initial_mesh, kernel, mode='same')

def solve_homogeneous_heat_equation_t_values(x_step, y_step, initial_mesh, t_max, t_num):
    solution = np.empty((t_num, *initial_mesh.shape))
    for i, t in enumerate(np.linspace(0, t_max, t_num, endpoint=True)):
        solution[i] = solve_homogeneous_heat_equation_single_t(x_step, y_step, initial_mesh, t)
    return solution

def solve_inhomogeneous_heat_equation_no_initial(x_step, y_step, t_step, source_mesh):
    x_num, y_num, t_num = source_mesh.shape
    value_1 = x_step * y_step / (4 * np.pi * t_step)
    value_2 = -(x_step ** 2) / (4 * t_step)
    value_3 = -(y_step ** 2) / (4 * t_step)
    i = np.arange(-x_num + 1, x_num).reshape(1, 1, 2 * x_num - 1)
    j = np.arange(-y_num + 1, y_num).reshape(1, 2 * y_num - 1, 1)
    k = np.arange(-t_num + 1, t_num).reshape(2 * t_num - 1, 1, 1)
    kernel = (value_1 / k) * np.exp((value_2 * (i ** 2) + value_3 * (j ** 2)) / k)
    return sig.convolve(source_mesh, kernel, mode='same')

def solve_inhomogeneous_heat_equation(x_step, y_step, t_step, initial_mesh, source_mesh):
    x_num, y_num, t_num = source_mesh.shape
    assert initial_mesh.shape == (x_num, y_num)
    solution = solve_inhomogeneous_heat_equation_no_initial(x_step, y_step, t_step, source_mesh)
    solution[0] += initial_mesh
    for i in range(1, t_num):
        solution[i] += solve_homogeneous_heat_equation(x_step, y_step, initial_mesh, i * t_step)
    return solution




'''




import numpy as np
import scipy.signal as sig

def solve_homogeneous_heat_equation_mesh(x_step, y_step, initial_mesh, tau):
    x_num, y_num = initial_heat_distribution.shape
    value_1 = x_step * y_step / (4 * np.pi * tau)
    value_2 = -(x_step ** 2) / (4 * tau)
    value_3 = -(y_step ** 2) / (4 * tau)
    i = np.arange(-x_num + 1, x_num).reshape(1, 2 * x_num + 1)
    j = np.arange(-y_num + 1, y_num).reshape(2 * y_num + 1, 1)
    kernel = value_1 * np.exp(value_2 * (i ** 2) + value_3 * (j ** 2))
    return sig.convolve(initial_heat_distribution, kernel, mode='same')

def solve_homogeneous_heat_equation_function(x_min, x_max, x_num, y_min, y_max, y_num, initial_function, tau):
    x_step = (x_max - x_min) / x_num
    y_step = (y_max - y_min) / y_num
    x = x_min + x_step * (0.5 + np.arange(x_num))
    y = y_min + y_step * (0.5 + np.arange(y_num))
    initial_mesh = initial_function(*np.meshgrid(x, y))
    return solve_homogeneous_heat_equation_mesh(x_step, y_step, initial_mesh, tau)

def solve_inhomogeneous_heat_equation_function_no_initial(x_min, x_max, x_num, y_min, y_max, y_num, t_min, t_max, t_num, source_function, k):
    x_step = (x_max - x_min) / x_num
    y_step = (y_max - y_min) / y_num
    t_step = (t_max - t_min) / t_num
    x = x_min + x_step * (0.5 + np.arange(x_num))
    y = y_min + y_step * (0.5 + np.arange(y_num))
    t = y_min + y_step * (0.5 + np.arange(y_num))
    source_mesh = source_function(*np.meshgrid(x, y, t))
    value_1 = x_step * y_step / (4 * np.pi * k)
    value_2 = -(x_step ** 2) / (4 * k * t_step)
    value_3 = -(y_step ** 2) / (4 * k * t_step)
    i = np.arange(-x_num + 1, x_num).reshape(1, 2 * x_num + 1, 1)
    j = np.arange(-y_num + 1, y_num).reshape(2 * y_num + 1, 1, 1)
    l = np.arange(-t_num + 1, t_num).reshape(1, 1, 2 * t_num + 1)
    kernel = (value_1 / l) * np.exp((value_2 * (i ** 2) + value_3 * (j ** 2)) / l)
    return sig.convolve(source_mesh, kernel, mode='same')

    def solve_heat_equation(x_min, x_max, x_num, y_min, y_max, y_num, initial, tau):
        x_step = (x_max - x_min) / x_num
        y_step = (y_max - y_min) / y_num
        i = np.arange(x_num)
        j = np.arange(y_num)
        x = x_min + x_step * (0.5 + i)
        y = y_min + y_step * (0.5 + j)
        x_mesh, y_mesh = np.meshgrid(x, y)
        initial_values = initial(*np.meshgrid(x, y)).reshape(y_num, x_num, 1, 1)
        ik = i.reshape((1, x_num, 1, 1)) - i.reshape((1, 1, 1, x_num))
        jl = j.reshape((y_num, 1, 1, 1)) - j.reshape((1, 1, y_num, 1))
        value_1 = x_step * y_step / (4 * np.pi * tau)
        value_2 = -(x_step ** 2) / (4 * tau)
        value_3 = -(y_step ** 2) / (4 * tau)
        kernel = value_1 * np.exp(value_2 * (ik ** 2) + value_3 * (jl ** 2))
        return np.sum(kernel * initial_values, axis=(0, 1))

'''
