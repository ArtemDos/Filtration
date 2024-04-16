from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
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
def fun(x: List[float], y: List[List[float]], Q: float, m_0: float) -> List[List[float]]: # y[0] = u, y[1] = v, y[2] = s
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
def bc_v_inc(ya: List[List[float]], yb: List[List[float]], v_inc: float, m_0: float) -> List[List[float]]:                               
    return np.array([
        ya[0] - 0, # для u(0) = 0
        ya[1] - v_inc / m_0, # для v(0) = v_inc / m
        yb[2] - 0 # для s(L) = 0
        ])


# Решение дифференциальной системы уравнений
def solving_equations(v_inc: float, m_0: float, x_plot: List[float]) -> List[List[float]]:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Вычисляем расход жидкости, используя граничное условие p(0)=p_inc и v(0)=v_inc/m
    x = np.linspace(0, L, N) # Заполняем массив координат х нулями
    y_guess = np.zeros((3, x.size)) # Пример np.zeros((2, 1)) ---> array([[ 0.],[ 0.]]) - начальные значения функций
    fun_v_inc = partial(fun, Q=Q, m_0=m_0) # Подставляем в функцию fun значение Q и m_0
    result = solve_bvp(fun_v_inc, lambda ya, yb: bc_v_inc(ya, yb, v_inc, m_0), x, y_guess, tol=1e-6) # Решаем систему уравнений
    y_plot = result.sol(x_plot) # (u(x), v(x), s(x))
    return y_plot # y[0] = u, y[1] = v, y[2] = s


# Вычисление числа Рейнольдса и В в зависимости от параметра v_inc
def calculation_Re_and_B(x_plot: List[float], v_inc_start: float, v_inc_finish: float, M: int) -> List[List[float]]:
    v_inc_values = np.linspace(v_inc_start, v_inc_finish, M) # Массив значений входящих скоростей v_inc
    Re_values = [] # Массив чисел Рейнольдса для каждой скорости v_inc
    B_values = [] # Массив чисел B для каждой скорости v_inc

    for v_inc in v_inc_values:
        solve = solving_equations(v_inc, m_0, x_plot) # y[0] = u, y[1] = v, y[2] = s
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


# Функция для вычисления максимальных значений сил
def calculate_max_forces(v_inc: float, x_plot: List[float]) -> None:
    F_d = []
    F_c = []
    F_b = []
    m_values = np.linspace(0.005, 1, 1000) # Задание параметров и значений m
    for m in m_values:
        y_plot = solving_equations(v_inc, m, x_plot)
        v = np.where(y_plot[1] == 0, 1e-9, y_plot[1])  # Замена нулей на очень маленькое значение
        Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Вычисляем расход жидкости, используя граничное условие p(0)=p_inc и v(0)=v_inc/m
        dvdx = -(d_0 * v + c_0 * v * v) / (C * Q / (v * v) - Q * v + b_0 * v) # Производная скорости
        F_d.append((abs(d_0 * v)).max())
        F_c.append(abs((c_0 * v**2)).max()) 
        F_b.append(abs((b_0 * v * dvdx)).max())

    # Построение графика максимальных сил F_d, F_c, F_b от m
    plt.figure()
    plt.plot(m_values, F_d, label=r'$F_d$')
    plt.plot(m_values, F_c, label=r'$F_c$')
    plt.plot(m_values, F_b, label=r'$F_b$')
    plt.title(r"$F_d$,$F_c$,$F_b$  от $m$", fontsize=16, loc='center')
    plt.xlabel('m')
    plt.ylabel('Forces')
    plt.legend()
    plt.yscale('symlog')
    plt.show()


# Построение графика скорости mv/v_inc(x/L)
def plot_velocity_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], v_inc: float, m_0: float, x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, m_0 * solve_without_Fb[1] / v_inc, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, m_0 * solve_with_Fb[1] / v_inc, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$mv/v_{inc}$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.4f}'.format(x))) 
    plt.grid()

    correlation_matrix = np.corrcoef(solve_with_Fb[1], solve_without_Fb[1])
    correlation = correlation_matrix[0, 1]
    norm_diff = np.linalg.norm(solve_with_Fb[1] - solve_without_Fb[1])
    norm = np.linalg.norm(solve_without_Fb[1])
    plt.text(0.95, 0.1, r"Корреляция $v_{{b_0 \neq 0}}$ и $v_{{b_0 = 0}}$ = {:.2f}".format(correlation), transform=plt.gca().transAxes, ha='right')
    plt.text(0.95, 0.15, r"Норма $\frac{{\|v_{{b_0 \neq 0}} - v_{{b_0 = 0}}\|}}{{\|v_{{b_0 = 0}}\|}}$ = {:.2f}".format(norm_diff/norm), transform=plt.gca().transAxes, ha='right')

    plt.show()


# Построение графика перемещения u/L(x/L)
def plot_u_l_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, solve_without_Fb[0] / L, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, solve_with_Fb[0] / L, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$u/L$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.5f}'.format(x))) 
    plt.grid()

    correlation_matrix = np.corrcoef(solve_with_Fb[0], solve_without_Fb[0])
    correlation = correlation_matrix[0, 1]
    norm_diff = np.linalg.norm(solve_with_Fb[0] - solve_without_Fb[0])
    norm = np.linalg.norm(solve_without_Fb[0])
    plt.text(0.95, 0.1, r"Корреляция $u_{{b_0 \neq 0}}$ и $u_{{b_0 = 0}}$ = {:.2f}".format(correlation), transform=plt.gca().transAxes, ha='right')
    plt.text(0.95, 0.15, r"Норма $\frac{{\|u_{{b_0 \neq 0}} - u_{{b_0 = 0}}\|}}{{\|u_{{b_0 = 0}}\|}}$ = {:.2f}".format(norm_diff/norm), transform=plt.gca().transAxes, ha='right')

    plt.show()


# Построение графика плотности жидкости ro_ist/ro_ist_0
def plot_density_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (m_0 * solve_with_Fb[1])
    ro_values_without_Fb = Q / (m_0 * solve_without_Fb[1])
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, ro_values_without_Fb / ro_f_ist_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, ro_values_with_Fb / ro_f_ist_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$\rho^{ист}_f/\rho^{ист}_{f_0}$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.4f}'.format(x))) 
    plt.grid()

    correlation_matrix = np.corrcoef(ro_values_with_Fb, ro_values_without_Fb)
    correlation = correlation_matrix[0, 1]
    norm_diff = np.linalg.norm(ro_values_with_Fb - ro_values_without_Fb)
    norm = np.linalg.norm(ro_values_without_Fb)
    plt.text(0.05, 0.1, r"Корреляция $\rho_{{b_0 \neq 0}}$ и $\rho_{{b_0 = 0}}$ = {:.2f}".format(correlation), transform=plt.gca().transAxes, ha='left')
    plt.text(0.05, 0.15, r"Норма $\frac{{\|\rho_{{b_0 \neq 0}} - \rho_{{b_0 = 0}}\|}}{{\|\rho_{{b_0 = 0}}\|}}$ = {:.2f}".format(norm_diff/norm), transform=plt.gca().transAxes, ha='left')

    plt.show()


# Построение графика давления жидкости p/p_0(x/L)
def plot_pressure_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (m_0 * solve_with_Fb[1])
    ro_values_without_Fb = Q / (m_0 * solve_without_Fb[1])
    p_values_with_Fb = (p_0 + C * (ro_values_with_Fb - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_values_without_Fb = (p_0 + C * (ro_values_without_Fb - ro_f_ist_0))
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, p_values_without_Fb / p_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, p_values_with_Fb / p_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$p/p_0$ от $x/L$")
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.1f}'.format(x))) 
    plt.grid()

    correlation_matrix = np.corrcoef(p_values_with_Fb, p_values_without_Fb)
    correlation = correlation_matrix[0, 1]
    norm_diff = np.linalg.norm(p_values_with_Fb - p_values_without_Fb)
    norm = np.linalg.norm(p_values_without_Fb)
    plt.text(0.05, 0.1, r"Корреляция $p_{{b_0 \neq 0}}$ и $p_{{b_0 = 0}}$ = {:.2f}".format(correlation), transform=plt.gca().transAxes, ha='left')
    plt.text(0.05, 0.15, r"Норма $\frac{{\|p_{{b_0 \neq 0}} - p_{{b_0 = 0}}\|}}{{\|p_{{b_0 = 0}}\|}}$ = {:.2f}".format(norm_diff/norm), transform=plt.gca().transAxes, ha='left')

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
    plt.plot(x_plot / L, sigma_f_without_Fb / p_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, sigma_f_with_Fb / p_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$\sigma_f/p_0$ от $x/L$", fontsize=16, loc='center')
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.2f}'.format(x))) 
    plt.grid()

    correlation_matrix = np.corrcoef(sigma_f_with_Fb, sigma_f_without_Fb)
    correlation = correlation_matrix[0, 1]
    norm_diff = np.linalg.norm(sigma_f_with_Fb - sigma_f_without_Fb)
    norm = np.linalg.norm(sigma_f_without_Fb)
    plt.text(0.95, 0.1, r"Корреляция $\sigma_{{f_{{b_0 \neq 0}}}}$ и $\sigma_{{f_{{b_0 = 0}}}}$ = {:.2f}".format(correlation), transform=plt.gca().transAxes, ha='right')
    plt.text(0.95, 0.15, r"Норма $\frac{{\|\sigma_{{f_{{b_0 \neq 0}}}} - \sigma_{{f_{{b_0 = 0}}}}\|}}{{\|\sigma_{{f_{{b_0 = 0}}}}\|}}$ = {:.2f}".format(norm_diff/norm), transform=plt.gca().transAxes, ha='right')

    plt.legend()
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
    plt.plot(x_plot / L, sigma_s_without_Fb / p_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, sigma_s_with_Fb / p_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$\sigma_s/p_0$ от $x/L$", fontsize=16, loc='center')
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.2f}'.format(x))) 
    plt.grid()

    correlation_matrix = np.corrcoef(sigma_s_with_Fb, sigma_s_without_Fb)
    correlation = correlation_matrix[0, 1]
    norm_diff = np.linalg.norm(sigma_s_with_Fb - sigma_s_without_Fb)
    norm = np.linalg.norm(sigma_s_without_Fb)
    plt.text(0.05, 0.1, r"Корреляция $\sigma_{{s_{{b_0 \neq 0}}}}$ и $\sigma_{{s_{{b_0 = 0}}}}$ = {:.2f}".format(correlation), transform=plt.gca().transAxes, ha='left')
    plt.text(0.05, 0.15, r"Норма $\frac{{\|\sigma_{{s_{{b_0 \neq 0}}}} - \sigma_{{s_{{b_0 = 0}}}}\|}}{{\|\sigma_{{s_{{b_0 = 0}}}}\|}}$ = {:.2f}".format(norm_diff/norm), transform=plt.gca().transAxes, ha='left')

    plt.legend()
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


answer = solving_equations(v_inc, m_0, x_plot)

b_0 = 0
answer_withot_Fb = solving_equations(v_inc, m_0, x_plot)

# plot_velocity_ratio(answer, answer_withot_Fb, v_inc, m_0, x_plot)
# plot_u_l_ratio(answer, answer_withot_Fb, x_plot)
# plot_density_ratio(answer, answer_withot_Fb, x_plot)
# plot_pressure_ratio(answer, answer_withot_Fb, x_plot)
# plot_sigma_f_ratio(answer, answer_withot_Fb, x_plot)
# plot_sigma_s_ratio(answer, answer_withot_Fb, x_plot)
# calculate_max_forces(v_inc, x_plot)

result_file_path = os.path.expanduser("./small.txt")
# Запись результатов в файл
with open(result_file_path, "w") as result_file:
    result_file.write("Результаты с F_b\n")
    for i in range(len(x_plot)):
        result_file.write("x = {:.3f}, u = {:.6f}, v = {:.6f}, s = {:.6f}\n".format(x_plot[i], answer[0][i], answer[1][i], answer[2][i]))

    result_file.write("\nРезультаты без F_b\n")
    for i in range(len(x_plot)):
        result_file.write("x = {:.3f}, u = {:.6f}, v = {:.6f}, s = {:.6f}\n".format(x_plot[i], answer_withot_Fb[0][i], answer_withot_Fb[1][i], answer_withot_Fb[2][i]))
