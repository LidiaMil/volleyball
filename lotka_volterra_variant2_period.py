import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# Параметры модели
alpha = 1.0
beta = 0.5
C1 = 0.4
C2 = 0.3

# РАвновесие
N0_eq = beta / C2
M0_eq = alpha / C1

print("Равновесные значения:")
print(f"N0 = {N0_eq:.3f},  M0 = {M0_eq:.3f}")

# --- ВАРИАНТ 2: значительное отклонение ---
N0 = N0_eq * 1.6     # 60% отклонение
M0 = M0_eq * 0.5     # 50% отклонение

print("\nНачальные условия варианта 2:")
print(f"N(0) = {N0:.3f},  M(0) = {M0:.3f}")

# Модель Лотки–Вольтерры
def lotka_volterra(t, z):
    N, M = z
    dNdt = (alpha - C1*M) * N
    dMdt = (-beta + C2*N) * M
    return [dNdt, dMdt]

# Диапазон интегрирования
t_span = (0, 80)
t_eval = np.linspace(0, 80, 5000)

# Решение
sol = solve_ivp(lotka_volterra, t_span, [N0, M0], t_eval=t_eval)
N = sol.y[0]
M = sol.y[1]
t = sol.t

# --- ВЫЧИСЛЕНИЕ ПЕРИОДА КОЛЕБАНИЙ ---
# Поиск максимумов N(t)
peaks, _ = find_peaks(N, height=np.mean(N))
if len(peaks) > 1:
    # расстояния между соседними максимумами — это период
    periods = np.diff(t[peaks])
    T = np.mean(periods)
else:
    T = None

print("\n--- Период колебаний ---")
if T:
    print(f"Период T ≈ {T:.3f}")
else:
    print("Недостаточно пиков для вычисления периода (попробуйте увеличить интервал).")

# --- ГРАФИКИ ---
plt.figure(figsize=(12, 5))

# N(t), M(t)
plt.subplot(1, 2, 1)
plt.plot(t, N, label="N(t) — жертвы")
plt.plot(t, M, label="M(t) — хищники")
plt.xlabel("t")
plt.ylabel("Численность")
plt.title("Изменение численности во времени (вариант 2)")
plt.legend()

# Фазовая диаграмма
plt.subplot(1, 2, 2)
plt.plot(N, M, label="Траектория в фазовом пространстве")
plt.scatter(N0_eq, M0_eq, color="red", s=80, label="Равновесие")
plt.xlabel("N — жертвы")
plt.ylabel("M — хищники")
plt.title("Фазовая диаграмма (вариант 2)")
plt.legend()

plt.tight_layout()
plt.show()
