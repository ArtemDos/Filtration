from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from typing import List
from functools import partial
import os
import sys


const_file_path = os.path.expanduser("./const.txt")

#читаем константы из файла 
with open(const_file_path, "r") as f:
    constants = {}
    for line in f:
        key, value = line.strip().split("=")
        constants[key] = float(value)

# Присваиваем считанные значения переменным
L = constants["L"]
C = constants["C"]
d_0 = constants["d_0"]
c_0 = constants["c_0"]
b_0 = constants["b_0"]
m_0 = constants["m_0"]
k=constants["k"]
nu=constants["nu"]
lambda_s_2nu_s = constants["lambda_s+2nu_s"]
v_inc = constants["v_inc"]
ro_f_ist_0 = constants["ro_f_ist_0"]
p_0 = constants["p_0"]
p_inc = constants["p_inc"]
v_inc_start=constants["v_inc_start"]
v_inc_finish=constants["v_inc_finish"]


# Определение функции для системы дифференциальных уравнений
def fun(x: List[float], y: List[List[float]], Q: float) -> List[List[float]]: # y[0] = u, y[1] = v, y[2] = s
    v = np.where(y[1] == 0, 1e-9, y[1])  # Замена нулей на очень маленькое значение

    dudx = y[2]
    dvdx = (d_0 * v + c_0 * v * v) / (C * Q / (v * v) - Q - b_0 * v)
    dsdx = -1 / lambda_s_2nu_s * (d_0 * v + c_0 * v * v + ((1 - m_0) * C * Q / (m_0 * v * v)) * dvdx)
    return [
        dudx, 
        dvdx,
        dsdx
        ]


# Определение граничных условий y[0] = u, y[1] = v, y[2] = s
def bc_v_inc(ya: List[List[float]], yb: List[List[float]], v_inc: float) -> List[List[float]]:                               
    return np.array([
        ya[0] - 0, # для u(0) = 0
        ya[1] - v_inc / m_0, # для v(0) = v_inc / m
        yb[2] - 0 # для s(L) = 0
        ])


# Решение дифференциальной системы уравнений
def solving_equations(v_inc: float, x_plot: List[float]) -> List[List[float]]:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Вычисляем расход жидкости, используя граничное условие p(0)=p_inc и v(0)=v_inc/m
    x = np.linspace(0, L, N) # Заполняем массив координат х нулями
    y_guess = np.zeros((3, x.size)) # Пример np.zeros((2, 1)) ---> array([[ 0.],[ 0.]]) - начальные значения функций
    fun_v_inc = partial(fun, Q=Q) # Подставляем в функцию fun значение Q 
    result = solve_bvp(fun_v_inc, lambda ya, yb: bc_v_inc(ya, yb, v_inc), x, y_guess, tol=1e-6) # Решаем систему уравнений
    y_plot = result.sol(x_plot) # (u(x), v(x), s(x))
    return y_plot # y[0] = u, y[1] = v, y[2] = s


# Вычисление числа Рейнольдса и В в зависимости от параметра v_inc
def calculation_Re_and_B(x_plot: List[float], v_inc_start: float, v_inc_finish: float, M: int)-> List[List[float]]:
    v_inc_values = np.linspace(v_inc_start, v_inc_finish, M) # Массив значений входящих скоростей v_inc
    Re_values = [] # Массив чисел Рейнольдса для каждой скорости v_inc
    B_values = [] # Массив чисел B для каждой скорости v_inc

    for v_inc in v_inc_values:
        solve = solving_equations(v_inc, x_plot) # y[0] = u, y[1] = v, y[2] = s
        # Вычисление Re и B
        Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Вычисляем расход жидкости, используя граничное условие p(0)=p_inc и v(0)=v_inc/m
        Re = Q * np.sqrt(k / m_0) / nu
        v = solve[1]  # Скорость жидкости
        dvdx = -(d_0 * v + c_0 * v * v) / (C * Q / (v * v) - Q * v + b_0 * v)  # Производная скорости
        B = v / (dvdx * np.sqrt(k / m_0))
        # Добавление максимальных значений Re и B
        Re_values.append(np.max(Re))
        B_values.append(np.max(B))
    
    return [v_inc_values, Re_values, B_values]


# Построение графика скорости v/v_inc(x/L)
def plot_velocity_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], v_inc: float, x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, solve_without_Fb[1] / v_inc, label='b_0 = 0', linestyle='dashed')
    plt.plot(x_plot / L, solve_with_Fb[1] / v_inc, label='b_0 != 0', linestyle='dotted')
    plt.title("Отношение v/v_inc от х/L")
    plt.xlabel('x/L')
    plt.ylabel('v/v_inc')
    plt.legend()
    plt.grid()
    plt.show()


# Построение графика перемещения u/L(x/L)
def plot_u_l_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, solve_without_Fb[0] / L, label='b_0 = 0', linestyle='dashed')
    plt.plot(x_plot / L, solve_with_Fb[0] / L, label='b_0 != 0', linestyle='dotted')
    plt.title("u/L от x/L")
    plt.xlabel('x/L')
    plt.ylabel('u/L')
    plt.legend()
    plt.grid()
    plt.show()


# Построение графика плотности жидкости ro_ist/ro_ist_0
def plot_density_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (m_0 * solve_with_Fb[1])
    ro_values_without_Fb = Q / (m_0 * solve_without_Fb[1])
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, ro_values_without_Fb / ro_f_ist_0, label='b_0 = 0', linestyle='dashed')
    plt.plot(x_plot / L, ro_values_with_Fb / ro_f_ist_0, label='b_0 != 0', linestyle='dotted')
    plt.title("ro_f_ist/ro_f_ist_0 от x/L")
    plt.xlabel('x/L')
    plt.ylabel('ro_f_ist/ro_f_ist_0')
    plt.legend()
    plt.grid()
    plt.show()


# Построение графика давления жидкости p/p_0(x/L)
def plot_pressure_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (m_0 * solve_with_Fb[1])
    ro_values_without_Fb = Q / (m_0 * solve_without_Fb[1])
    p_values_with_Fb = (p_0 + C * (ro_values_with_Fb - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_values_without_Fb = (p_0 + C * (ro_values_without_Fb - ro_f_ist_0))
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, p_values_without_Fb / p_0, label='b_0 = 0', linestyle='dashed')
    plt.plot(x_plot / L, p_values_with_Fb / p_0, label='b_0 != 0', linestyle='dotted')
    plt.title("p/p_0 от x/L")
    plt.xlabel('x/L')
    plt.ylabel('p/p_0')
    plt.grid()
    plt.show()


# Построение графика тензора напряжения жидкости sigma_f/p_0(x/L)
def plot_sigma_f_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (m_0 * solve_with_Fb[1])
    ro_values_without_Fb = Q / (m_0 * solve_without_Fb[1])
    p_values_with_Fb = (p_0 + C * (ro_values_with_Fb - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_values_without_Fb = (p_0 + C * (ro_values_without_Fb - ro_f_ist_0))
    sigma_f_with_Fb = - m_0 * p_values_with_Fb # sigma_f =  - p * m
    sigma_f_without_Fb = - m_0 * p_values_without_Fb
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, sigma_f_without_Fb / p_0, label='b_0 = 0', linestyle='dashed')
    plt.plot(x_plot / L, sigma_f_with_Fb / p_0, label='b_0 != 0', linestyle='dotted')
    plt.title("sigma_f/p_0 от x/L")
    plt.xlabel('x/L')
    plt.ylabel('sigma_f/p_0')
    plt.grid()
    plt.show()


# Построение графика тензора напряжения каркаса sigma_s/p_0(x/L)
def plot_sigma_s_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (m_0 * solve_with_Fb[1])
    ro_values_without_Fb = Q / (m_0 * solve_without_Fb[1])
    p_values_with_Fb = (p_0 + C * (ro_values_with_Fb - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_values_without_Fb = (p_0 + C * (ro_values_without_Fb - ro_f_ist_0))
    fueling_part_with_Fb = -(1 - m_0) * p_values_with_Fb
    fueling_part_without_Fb = -(1 - m_0) * p_values_without_Fb
    sigma_s_with_Fb = fueling_part_with_Fb + lambda_s_2nu_s * solve_with_Fb[2]
    sigma_s_without_Fb = fueling_part_without_Fb + lambda_s_2nu_s * solve_without_Fb[2]
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, sigma_s_without_Fb / p_0, label='b_0 = 0', linestyle='dashed')
    plt.plot(x_plot / L, sigma_s_with_Fb / p_0, label='b_0 != 0', linestyle='dotted')
    plt.title("sigma_s/p_0 от x/L")
    plt.xlabel('x/L')
    plt.ylabel('sigma_s/p_0')
    plt.grid()
    plt.show()


# Получение значения,
# где N - разбиение по х от 0 до L, M - разбие
# ние входящих скоростей v_inc от v_inc_start до v_inc_finish
# из аргументов командной строки
if len(sys.argv) > 2:  # Если аргумент был передан
    N = int(sys.argv[1])
    M = int(sys.argv[2])
else:
    N = 500
    M = 200


x_plot = np.linspace(0, L, N) # Массив значений координат х 

answer = solving_equations(v_inc, x_plot)

b_0 = 0
answer_withot_Fb = solving_equations(v_inc, x_plot)

# plot_velocity_ratio(answer, answer_withot_Fb, v_inc, x_plot)
# plot_u_l_ratio(answer, answer_withot_Fb, x_plot)
# plot_density_ratio(answer, answer_withot_Fb, x_plot)
# plot_pressure_ratio(answer, answer_withot_Fb, x_plot)
# plot_sigma_f_ratio(answer, answer_withot_Fb, x_plot)
# plot_sigma_s_ratio(answer, answer_withot_Fb, x_plot)
