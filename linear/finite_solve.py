from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import List
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
lambda_s_2nu_s = constants["lambda_s+2nu_s"]
v_inc = constants["v_inc"]
ro_f_ist_0 = constants["ro_f_ist_0"]
p_0 = constants["p_0"]
p_inc = constants["p_inc"]


# Определяем коэффициент при интерактивных силах:
def f_m(f_0: float, m_0: float, m: List[float]) -> List[float]:
    return f_0 * (m_0 / m) * ((1 - m) / (1 - m_0)) # f = f_0 * (m0 / m) * ((1 - m) / (1 - m0))


# Определяем A(m, v):
def A_m_v(C: float, Q: float, m_0: float, a: float, c: float, h: float, m: List[float], v: List[float]) -> List[float]:
    return (1 - m_0) * C * Q / (m * m * v) - h / (1 - m_0) + 2 * (a + c) * (1 - m_0) / ((1 - m) * (1 - m)) # A(m, v) = (1 - m_0) * C * Q / (m^2 * v) - h / (1 - m_0) + 2 * (a + c) * (1 - m_0) / (1 - m)^2


# Определяем B(m, v):
def B_m_v(C: float, Q: float, m_0: float, b_0: float, m: List[float], v: List[float]) -> List[float]:
    return (1 - m_0) * C * Q / (m * v * v) + f_m(b_0, m_0, m) * v # B(m, v) = (1 - m_0) * C * Q / (m * v^2) + b(m) * v


# Определяем C(m, v):
def C_m_v(c_0: float, m_0: float, m: List[float], v: List[float]) -> List[float]:
    return f_m(d_0, m_0, m) * v + f_m(c_0, m_0, m) * v * v # C(m, v) =  d(m) * v + c(m) * v^2


# Определяем D(m, v):
def D_m_v(p_0: float, C: float, ro_f_ist_0: float) -> float:
    return -p_0 + C * ro_f_ist_0 # D(m, v) =  -p_0 + C * ro_f_ist_0


# Определяем E(m, v):
def E_m_v(C: float, Q: float, m_0: float, b_0: float, m: List[float], v: List[float]) -> List[float]:
    return (1 - m_0) * C * Q / (v * v) - f_m(b_0, m_0, m) * v - Q # E(m, v) = (1 - m_0) * C * Q / v^2 - b(m) * v - Q


# Определяем систему уравнений:
def ode_system(C: float, Q: float, m_0: float, a: float, c: float, h: float, x: List[float], y: List[List[float]]) -> List[List[float]]:
    m, v = y
    dmdx = (B_m_v(C, Q, m_0, b_0, m, v) * C_m_v(c_0, m_0, m, v) + C_m_v(c_0, m_0, m, v) * E_m_v(C, Q, m_0, b_0, m, v)) / (A_m_v(C, Q, m_0, a, c, h, m, v) * D_m_v(p_0, C, ro_f_ist_0) - A_m_v(C, Q, m_0, a, c, h, m, v) * E_m_v(C, Q, m_0, b_0,m, v))
    dvdx = (A_m_v(C, Q, m_0, a, c, h, m, v) * C_m_v(c_0, m_0, m, v) + C_m_v(c_0, m_0, m, v) * D_m_v(p_0, C, ro_f_ist_0)) / (A_m_v(C, Q, m_0, a, c, h, m, v) * E_m_v(C, Q, m_0, b_0, m, v) - B_m_v(C, Q, m_0, b_0, m, v) * D_m_v(p_0, C, ro_f_ist_0))
    return [dmdx, dvdx]

def solve_system(a, b, c, e, p, l, initial_conditions, x_span):
    sol = solve_ivp(ode_system, x_span, initial_conditions, args=(a, b, c, e, p, l), t_eval=np.linspace(x_span[0], x_span[1], 100))
    return sol

# Define your functions a, b, c, e, p, l as needed

# Define initial conditions and x_span
initial_conditions = [m_initial, v_initial]  # Initial conditions for m and v
x_span = [x_initial, x_final]  # Initial and final value of x

# Solve the system of equations
solution = solve_system(a, b, c, e, p, l, initial_conditions, x_span)

# Extract the solution
m = solution.y[0]
v = solution.y[1]
x_values = solution.t

