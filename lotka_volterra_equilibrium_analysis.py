import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры модели (любые положительные для демонстрации)
alpha = 1.0   # темп роста жертвы
beta = 0.5    # естественная смертность хищника
C1 = 0.4      # влияние хищников на жертв
C2 = 0.3      # влияние жертвы на хищников

# Положение равновесия
M0_eq = alpha / C1
N0_eq = beta / C2

print("Положение равновесия:")
print(f"N0 (жертвы) = {N0_eq:.3f}")
print(f"M0 (хищники) = {M0_eq:.3f}")

# ---- ВОПРОС 1 ----
# Если N(0)=N0_eq и M(0)=M0_eq — остаётся ли система неизменной?

def lotka_volterra(t, z):
    N, M = z
    dNdt = (alpha - C1*M) * N
    dMdt = (-beta + C2*N) * M
    return [dNdt, dMdt]

t_span = (0, 50)
t_eval = np.linspace(0, 50, 2000)

# 1. Начальноe условие ровно в равновесии
sol_eq = solve_ivp(lotka_volterra, t_span, [N0_eq, M0_eq], t_eval=t_eval)

# ---- ВОПРОС 2 ----
# Небольшое отклонение → наблюдаем циклы (нейтральная устойчивость)

delta = 0.1
sol_shifted = solve_ivp(
    lotka_volterra,
    t_span,
    [N0_eq + delta, M0_eq - delta],
    t_eval=t_eval
)

# --- ГРАФИКИ ---

plt.figure(figsize=(12, 5))

# Временные графики
plt.subplot(1, 2, 1)
plt.plot(t_eval, sol_eq.y[0], label='N(t) в равновесии')
plt.plot(t_eval, sol_eq.y[1], label='M(t) в равновесии')
plt.plot(t_eval, sol_shifted.y[0], '--', label='N(t) с отклонением')
plt.plot(t_eval, sol_shifted.y[1], '--', label='M(t) с отклонением')
plt.xlabel("t")
plt.ylabel("Численность")
plt.legend()
plt.title("Изменение численности во времени")

# Фазовая плоскость
plt.subplot(1, 2, 2)
plt.plot(sol_eq.y[0], sol_eq.y[1], label='Траектория при равновесии')
plt.plot(sol_shifted.y[0], sol_shifted.y[1], '--', label='Траектория при отклонении')
# Точка равновесия как "траектория при равновесии"
plt.scatter([N0_eq], [M0_eq], 
            color='blue', s=250, edgecolor='black', 
            label='Траектория при равновесии')

# Показываем положение равновесия отдельной точкой
plt.scatter([N0_eq], [M0_eq], 
            color='red', s=80, 
            label='Точка равновесия')

plt.xlabel("N — жертвы")
plt.ylabel("M — хищники")
plt.legend()
plt.title("Фазовая диаграмма системы")

plt.tight_layout()
plt.show()
