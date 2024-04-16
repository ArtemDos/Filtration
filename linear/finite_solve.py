from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import numpy as np
from scipy import integrate
from scipy import interpolate
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


# Определяем коэффициент при интерактивных силах:
def f_m(f_0: float, m_0: float, m: List[float]) -> List[float]:
    m_val = np.where(m == 0, 1e-6, m)  # Замена нулей на очень маленькое значение
    return f_0 * (m_0 / m_val) * ((1 - m_val) / (1 - m_0)) # f = f_0 * (m0 / m) * ((1 - m) / (1 - m0))
    # return f_0 * (m_0 / m_val)


# Определяем A(m, v):
def A_m_v(C: float, Q: float, m_0: float, a: float, c: float, h: float, m: List[float], v: List[float]) -> List[float]:
    m_val = np.where(m == 0, 1e-9, m)  # Замена нулей на очень маленькое значение
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return (1 - m_0) * C * Q / (m_val * m_val * v_val) - h / (1 - m_0) + 2 * (a + c) * (1 - m_0) / ((1 - m_val) * (1 - m_val)) # A(m, v) = (1 - m_0) * C * Q / (m^2 * v) - h / (1 - m_0) + 2 * (a + c) * (1 - m_0) / (1 - m)^2
    # return p_0 + C * (Q/ (m_val * v_val) - ro_f_ist_0) + h / (1 - m_0) + 2 * (a + c) * (1 - m_0) / ((1 - m_val) * (1 - m_val))


# Определяем B(m, v):
def B_m_v(C: float, Q: float, m_0: float, b_0: float, m: List[float], v: List[float]) -> List[float]:
    m_val = np.where(m == 0, 1e-6, m)  # Замена нулей на очень маленькое значение
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return (1 - m_0) * C * Q / (m_val * v_val * v_val) + f_m(b_0, m_0, m) * v_val # B(m, v) = (1 - m_0) * C * Q / (m * v^2) + b(m) * v


# Определяем C(m, v):
def C_m_v(c_0: float, m_0: float, m: List[float], v: List[float]) -> List[float]:
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return f_m(d_0, m_0, m) * v_val + f_m(c_0, m_0, m) * v_val * v_val # C(m, v) =  d(m) * v + c(m) * v^2


# Определяем D(m, v):
def D_m_v(p_0: float, C: float, ro_f_ist_0: float) -> float:
    return -p_0 + C * ro_f_ist_0 # D(m, v) =  -p_0 + C * ro_f_ist_0


# Определяем E(m, v):
def E_m_v(C: float, Q: float, m_0: float, b_0: float, m: List[float], v: List[float]) -> List[float]:
    v_val = np.where(v== 0, 1e-9, v)  # Замена нулей на очень маленькое значение
    return (1 - m_0) * C * Q / (v_val * v_val) - f_m(b_0, m_0, m) * v_val - Q # E(m, v) = (1 - m_0) * C * Q / v^2 - b(m) * v - Q


# Определяем систему уравнений:
def ode_system(C: float, Q: float, m_0: float, a: float, c: float, h: float, x: List[float], y: List[List[float]]) -> List[List[float]]:
    m, v = y
    dmdx = (B_m_v(C, Q, m_0, b_0, m, v) * C_m_v(c_0, m_0, m, v) + C_m_v(c_0, m_0, m, v) * E_m_v(C, Q, m_0, b_0, m, v)) / (B_m_v(C, Q, m_0, b_0, m, v) * D_m_v(p_0, C, ro_f_ist_0) - A_m_v(C, Q, m_0, a, c, h, m, v) * E_m_v(C, Q, m_0, b_0,m, v))
    dvdx = (A_m_v(C, Q, m_0, a, c, h, m, v) * C_m_v(c_0, m_0, m, v) + C_m_v(c_0, m_0, m, v) * D_m_v(p_0, C, ro_f_ist_0)) / (A_m_v(C, Q, m_0, a, c, h, m, v) * E_m_v(C, Q, m_0, b_0, m, v) - B_m_v(C, Q, m_0, b_0, m, v) * D_m_v(p_0, C, ro_f_ist_0))
    return [dmdx, dvdx]


# Определение граничных условий y[0] = m, y[1] = v
def bc_v_inc(ya: List[List[float]], yb: List[List[float]], v_inc: float, m_0: float) -> List[List[float]]:                               
    return np.array([
        ya[0] - m_0, # для m(a) = m_0
        ya[1] - v_inc / m_0, # для v(a) = v_inc / m
        ])


# Решение дифференциальной системы уравнений
def solving_equations(v_inc: float, m_0: float, a: float, c: float, h: float, x_plot: List[float]) -> List[List[float]]:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Вычисляем расход жидкости, используя граничное условие p(0)=p_inc и v(0)=v_inc/m
    x = np.linspace(0, L, N) # Заполняем массив координат х нулями
    m_guess = np.linspace(m_0, m_0, num=len(x))
    v_guess = np.linspace(v_inc / m_0, v_inc / m_0, num=len(x))
    y_guess = np.vstack([m_guess, v_guess])

    # y_guess = np.zeros((2, x.size)) # Начальные значения функций
    result = solve_bvp(lambda x, y_guess: ode_system(C, Q, m_0, a, c, h, x, y_guess), lambda ya, yb: bc_v_inc(ya, yb, v_inc, m_0), x, y_guess, tol=1e-6) # Решаем систему уравнений
    y_plot = result.sol(x_plot) # (m(x), v(x))
    return y_plot # y[0] = m, y[1] = v


# Построение графика пористости m(x/L)
def plot_m_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, solve_without_Fb[0], label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, solve_with_Fb[0], label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"m от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.grid()
    plt.show()


# Построение графика скорости v/v_inc(x/L)
def plot_velocity_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], v_inc: float, x_plot: List[float]) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, solve_without_Fb[1] / v_inc * m_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, solve_with_Fb[1] / v_inc * m_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$vm_0/v_{inc}$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.grid()
    plt.show()


# Построение графика перемещения u(x/L)
def plot_u_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], L: float, x_plot: List[float]) -> None:
    f_with_Fb = integrate.cumtrapz(solve_with_Fb[0], x_plot, initial=0)
    F_with_Fb = interpolate.interp1d(x_plot, f_with_Fb, fill_value="extrapolate")
    f_without_Fb = integrate.cumtrapz(solve_without_Fb[0], x_plot, initial=0)
    F_without_Fb = interpolate.interp1d(x_plot, f_without_Fb, fill_value="extrapolate")

    u_with_Fb = F_with_Fb(x_plot)
    u_without_Fb = F_without_Fb(x_plot)

    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, u_with_Fb / L, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, u_without_Fb / L, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$u/L$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.5f}'.format(x))) 
    plt.grid()
    plt.show()


# Построение графика плотности жидкости ro_ist/ro_ist_0
def plot_density_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (solve_with_Fb[0] * solve_with_Fb[1])
    ro_values_without_Fb = Q / (solve_without_Fb[0] * solve_without_Fb[1])
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, ro_values_without_Fb / ro_f_ist_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, ro_values_with_Fb / ro_f_ist_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$\rho^{ист}_f/\rho^{ист}_{f_0}$ от $x/L$", fontsize=16, loc='center')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.4f}'.format(x))) 
    plt.grid()
    plt.show()


# Построение графика давления жидкости p/p_0(x/L)
def plot_pressure_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (solve_with_Fb[0] * solve_with_Fb[1])
    ro_values_without_Fb = Q / (solve_without_Fb[0] * solve_without_Fb[1])
    p_values_with_Fb = (p_0 + C * (ro_values_with_Fb - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_values_without_Fb = (p_0 + C * (ro_values_without_Fb - ro_f_ist_0))
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, p_values_without_Fb / p_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, p_values_with_Fb / p_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$p/p_0$ от $x/L$")
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.1f}'.format(x))) 
    plt.grid()
    plt.show()


# Построение графика тензора напряжения жидкости sigma_f/p_0(x/L)
def plot_sigma_f_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (solve_with_Fb[0] * solve_with_Fb[1])
    ro_values_without_Fb = Q / (solve_without_Fb[0] * solve_without_Fb[1])
    p_values_with_Fb = (p_0 + C * (ro_values_with_Fb - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_values_without_Fb = (p_0 + C * (ro_values_without_Fb - ro_f_ist_0))
    sigma_f_with_Fb = - solve_with_Fb[0] * p_values_with_Fb # sigma_f =  - p * m
    sigma_f_without_Fb = - solve_without_Fb[0] * p_values_without_Fb
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, sigma_f_without_Fb / p_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, sigma_f_with_Fb / p_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$S^{xx}_f/p_0$ от $x/L$", fontsize=16, loc='center')
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.2f}'.format(x))) 
    plt.grid()
    plt.legend()
    plt.show()


# Построение графика тензора напряжения каркаса sigma_s/p_0(x/L)
def plot_sigma_s_ratio(solve_with_Fb: List[List[float]], solve_without_Fb: List[List[float]], x_plot: List[float]) -> None:
    Q = ((p_inc - p_0) / C + ro_f_ist_0) * v_inc # Q = ro_ist * m * v
    ro_values_with_Fb = Q / (solve_with_Fb[0] * solve_with_Fb[1])
    ro_values_without_Fb = Q / (solve_without_Fb[0] * solve_without_Fb[1])
    p_values_with_Fb = (p_0 + C * (ro_values_with_Fb - ro_f_ist_0)) # p = p_0 + C * (ro_ist - ro_ist_0)
    p_values_without_Fb = (p_0 + C * (ro_values_without_Fb - ro_f_ist_0))
    fueling_part_with_Fb = -(1 - solve_with_Fb[0]) * p_values_with_Fb
    fueling_part_without_Fb = -(1 - solve_without_Fb[0]) * p_values_without_Fb
    sigma_s_with_Fb = fueling_part_with_Fb + 2 * (a + c) - h * (1 - solve_with_Fb[0]) * (1 - solve_with_Fb[0]) / ((1 - m_0) * (1 - m_0))
    sigma_s_without_Fb = fueling_part_without_Fb + 2 * (a + c) - h * (1 - solve_without_Fb[0]) * (1 - solve_without_Fb[0]) / ((1 - m_0) * (1 - m_0))
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot / L, sigma_s_without_Fb / p_0, label=r'$b_0 = 0$', linestyle='dashed')
    plt.plot(x_plot / L, sigma_s_with_Fb / p_0, label=r'$b_0 \neq 0$', linestyle='dotted')
    plt.title(r"$P^{xx}/p_0$ от $x/L$", fontsize=16, loc='center')
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.2f}'.format(x))) 
    plt.grid()
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
    N = 200
    M = 200

x_plot = np.linspace(0, L, N) # Массив значений координат х 
answer = solving_equations(v_inc, m_0, a, c, h, x_plot)
b_0 = 0
answer_withot_Fb = solving_equations(v_inc, m_0, a, c, h, x_plot)

plot_m_ratio(answer,answer_withot_Fb, x_plot)
plot_velocity_ratio(answer, answer_withot_Fb, v_inc, x_plot)
plot_u_ratio(answer, answer_withot_Fb, L, x_plot)
plot_density_ratio(answer, answer_withot_Fb, x_plot)
plot_pressure_ratio(answer, answer_withot_Fb, x_plot)
plot_sigma_f_ratio(answer, answer_withot_Fb, x_plot)
plot_sigma_s_ratio(answer, answer_withot_Fb, x_plot)
