import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from small_solve import solving_equations as small
from real_small_solve import solving_equations as real_small
from typing import List
import numpy as np
import os
import sys


# Построение графика скорости mv/v_inc(x/L)
def plot_velocity_ratio(small: List[List[float]], real_small: List[List[float]], v_inc: float, m_0: float, x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, m_0 * small[1] / v_inc, label=r'small', linestyle='dashed')
    plt.plot(x_plot / L, m_0 * real_small[1] / v_inc, label=r'real small', linestyle='dotted')
    plt.title(r"$mv/v_{inc}$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.4f}'.format(x))) 
    plt.grid()
    plt.show()


# Построение графика перемещения u/L(x/L)
def plot_u_l_ratio(small: List[List[float]], real_small: List[List[float]], x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, small[0] / L, label=r'small', linestyle='dashed')
    plt.plot(x_plot / L, real_small[0] / L, label=r'real small', linestyle='dotted')
    plt.title(r"$u/L$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.5f}'.format(x))) 
    plt.grid()
    plt.show()


# Построение графика плотности жидкости ro_ist/ro_ist_0
def plot_density_ratio(small: List[List[float]], real_small: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_small = Q / (m_0 * small[1])
    ro_real_small = Q / (m_0 * real_small[1])
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, ro_small / ro_f_ist_0, label=r'$small$', linestyle='dashed')
    plt.plot(x_plot / L, ro_real_small / ro_f_ist_0, label=r'real small', linestyle='dotted')
    plt.title(r"$\rho^{ист}_f/\rho^{ист}_{f_0}$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.4f}'.format(x))) 
    plt.grid()
    plt.show()


# Построение графика давления жидкости p/p_0(x/L)
def plot_pressure_ratio(small: List[List[float]], real_small: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_small = Q / (m_0 * small[1])
    ro_real_small = Q / (m_0 * real_small[1])
    p_ro_small = (p_0 + C * (ro_small - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_real_small = (p_0 + C * (ro_real_small - ro_f_ist_0))
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, p_ro_small / p_0, label=r'small', linestyle='dashed')
    plt.plot(x_plot / L, p_real_small / p_0, label=r'real small', linestyle='dotted')
    plt.title(r"$p/p_0$ от $x/L$")
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.1f}'.format(x))) 
    plt.grid()
    plt.show()


# Построение графика тензора напряжения жидкости sigma_f/p_0(x/L)
def plot_sigma_f_ratio(small: List[List[float]], real_small: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_small = Q / (m_0 * small[1])
    ro_real_small = Q / (m_0 * real_small[1])
    p_small = (p_0 + C * (ro_small - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_real_small = (p_0 + C * (ro_real_small - ro_f_ist_0))
    sigma_f_small = - m_0 * p_small # sigma_f =  - p * m
    sigma_f_real_small = - m_0 * p_real_small
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, sigma_f_small / p_0, label=r'small', linestyle='dashed')
    plt.plot(x_plot / L, sigma_f_real_small / p_0, label=r'real small', linestyle='dotted')
    plt.title(r"$\sigma_f/p_0$ от $x/L$", fontsize=16, loc='center')
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.2f}'.format(x))) 
    plt.grid()
    plt.legend()
    plt.show()


# Построение графика тензора напряжения каркаса sigma_s/p_0(x/L)
def plot_sigma_s_ratio(small: List[List[float]], real_small: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_small = Q / (m_0 * small[1])
    ro_real_small = Q / (m_0 * real_small[1])
    p_small = (p_0 + C * (ro_small - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_real_small = (p_0 + C * (ro_real_small - ro_f_ist_0))
    fueling_part_small = -(1 - m_0) * p_small
    fueling_part_real_small = -(1 - m_0) * p_real_small
    sigma_s_with_small = fueling_part_small + lambda_s_2nu_s * small[2]
    sigma_s_real_small = fueling_part_real_small + lambda_s_2nu_s * real_small[2]
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, sigma_s_with_small / p_0, label=r'small', linestyle='dashed')
    plt.plot(x_plot / L, sigma_s_real_small / p_0, label=r'real small', linestyle='dotted')
    plt.title(r"$\sigma_s/p_0$ от $x/L$", fontsize=16, loc='center')
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.2f}'.format(x))) 
    plt.grid()
    plt.legend()
    plt.show()


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


small_answer = small(v_inc, m_0, x_plot)

real_small_answer = real_small(v_inc, C, m_0, b_0, c_0, d_0, p_0, ro_f_ist_0, x_plot)

# plot_velocity_ratio(small_answer, real_small_answer, v_inc, m_0, x_plot)
# plot_u_l_ratio(small_answer, real_small_answer, x_plot)
# plot_density_ratio(small_answer, real_small_answer, x_plot)
# plot_pressure_ratio(small_answer, real_small_answer, x_plot)
# plot_sigma_f_ratio(small_answer, real_small_answer, x_plot)
plot_sigma_s_ratio(small_answer, real_small_answer, x_plot)
