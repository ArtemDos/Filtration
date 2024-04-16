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
a = constants["a"]
c = constants["c"]
h = constants["h"]
k=constants["k"]
lambda_s_2nu_s = constants["lambda_s+2nu_s"]
v_inc = constants["v_inc"]
ro_f_ist_0 = constants["ro_f_ist_0"]
p_0 = constants["p_0"]
p_inc = constants["p_inc"]


# Определяем A(u, v, s):
def A_u_v_s(C: float, Q: float, m_0:  float, u: List[float], v: List[float], s: List[float]) -> List[float]:
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return  ((1 - m_0) * C * Q) / (m_0 * m_0 * v_val) + lambda_s_2nu_s # A(u, v, s) = 1-m/m^2 * CQ/v +lambda_s + 2nu_s


# Определяем B(u, v, s):
def B_u_v_s(C: float, Q: float, m_0:  float, u: List[float], v: List[float], s: List[float]) -> List[float]:
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return C * Q / (m_0 * v_val * v_val) - Q # B(u, v, s) = CQ/(mv^2) + Q


# Определяем C(u, v, s):
def C_u_v_s(C: float, Q: float, m_0:  float, p_0: float, ro_f_ist_0: float, u: List[float], v: List[float], s: List[float]) -> List[float]:
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return -(1 - m_0) * (p_0 - C * ro_f_ist_0) # C(u, v, s) = -(1-m)(p_0 - C*ro_f_ist_0)


# Определяем D(u, v, s):
def D_u_v_s(C: float, Q: float, b_0:  float, u: List[float], v: List[float], s: List[float]) -> float:
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return C * Q / (v_val * v_val) - b_0 * v_val - Q  # D(u, v, s) =  CQ/v^2 - b_0*v - Q


# Определяем E(u, v, s):
def E_u_v_s(d_0: float, c_0: float, v: List[float]) -> List[float]:
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return d_0 * v_val + c_0 * v_val * v_val  # E(u, v, s) = d_0*v+c_0*v^2

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


# Определяем систему уравнений:
def ode_system(C: float, Q: float, m_0: float, b_0: float, c_0: float, d_0: float, p_0: float, ro_f_ist_0: float, y: List[List[float]]) -> List[List[float]]: # y[0] = u, y[1] = v, y[2] = s
    u, v, s = y

    dudx = s
    dvdx = A_u_v_s(C, Q, m_0, u, v, s) * E_u_v_s(d_0, c_0, v) / (A_u_v_s(C, Q, m_0, u, v, s) * D_u_v_s(C, Q, b_0, u, v, s) - B_u_v_s(C, Q, m_0, u, v, s) * C_u_v_s(C, Q, m_0, p_0, ro_f_ist_0, u, v, s))
    dsdx = - B_u_v_s(C, Q, m_0, u, v, s) * E_u_v_s(d_0, c_0, v) / (A_u_v_s(C, Q, m_0, u, v, s) * D_u_v_s(C, Q, b_0, u, v, s) - B_u_v_s(C, Q, m_0, u, v, s) * C_u_v_s(C, Q, m_0, p_0, ro_f_ist_0, u, v, s))

    return [dudx, dvdx, dsdx]


# Определение граничных условий y[0] = u, y[1] = v, y[2] = s
def bc_v_inc(ya: List[List[float]], yb: List[List[float]], v_inc: float, m_0: float) -> List[List[float]]:                               
    return np.array([
        ya[0] - 0, # для u(0) = 0
        ya[1] - v_inc / m_0, # для v(0) = v_inc / m
        yb[2] - 0 # для s(L) = 0
        ])


# Решение дифференциальной системы уравнений
def solving_equations(v_inc: float, C: float, m_0: float, b_0: float, c_0: float, d_0: float, p_0: float, ro_f_ist_0: float, x_plot: List[float]) -> List[List[float]]:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Вычисляем расход жидкости, используя граничное условие p(0)=p_inc и v(0)=v_inc/m
    x = np.linspace(0, L, N) # Заполняем массив координат х нулями
    y_guess = np.zeros((3, x.size))
    result = solve_bvp(lambda x, y: ode_system(C, Q, m_0, b_0, c_0, d_0, p_0, ro_f_ist_0, y), lambda ya, yb: bc_v_inc(ya, yb, v_inc, m_0), x, y_guess, tol=1e-6)
    y_plot = result.sol(x_plot) # (u(x), v(x), s(x))
    return y_plot # y[0] = u, y[1] = v, y[2] = s


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
    plt.text(0.95, 0.1, r"Корреляция $\sigma_{{s_{{b_0 \neq 0}}}}$ и $\sigma_{{s_{{b_0 = 0}}}}$ = {:.2f}".format(correlation), transform=plt.gca().transAxes, ha='right')
    plt.text(0.95, 0.15, r"Норма $\frac{{\|\sigma_{{s_{{b_0 \neq 0}}}} - \sigma_{{s_{{b_0 = 0}}}}\|}}{{\|\sigma_{{s_{{b_0 = 0}}}}\|}}$ = {:.2f}".format(norm_diff/norm), transform=plt.gca().transAxes, ha='right')

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
answer = solving_equations(v_inc, C, m_0, b_0, c_0, d_0, p_0, ro_f_ist_0, x_plot)

b_0 = 0
answer_withot_Fb = solving_equations(v_inc, C, m_0, b_0, c_0, d_0, p_0, ro_f_ist_0, x_plot)

# plot_velocity_ratio(answer, answer_withot_Fb, v_inc, m_0, x_plot)
# plot_u_l_ratio(answer, answer_withot_Fb, x_plot)
# plot_density_ratio(answer, answer_withot_Fb, x_plot)
# plot_pressure_ratio(answer, answer_withot_Fb, x_plot)
# plot_sigma_f_ratio(answer, answer_withot_Fb, x_plot)
# plot_sigma_s_ratio(answer, answer_withot_Fb, x_plot)


result_file_path = os.path.expanduser("./real_small.txt")
# Запись результатов в файл
with open(result_file_path, "w") as result_file:
    result_file.write("Результаты с F_b\n")
    for i in range(len(x_plot)):
        result_file.write("x = {:.3f}, u = {:.6f}, v = {:.6f}, s = {:.6f}\n".format(x_plot[i], answer[0][i], answer[1][i], answer[2][i]))

    result_file.write("\nРезультаты без F_b\n")
    for i in range(len(x_plot)):
        result_file.write("x = {:.3f}, u = {:.6f}, v = {:.6f}, s = {:.6f}\n".format(x_plot[i], answer_withot_Fb[0][i], answer_withot_Fb[1][i], answer_withot_Fb[2][i]))


