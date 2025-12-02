import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# ПАРАМЕТРЫ МОДЕЛИ
# -----------------------------
M0 = 80          # начальная мощность
R0 = 100         # число работников
a = 2            # параметр f(x)
T = 50           # горизонт времени
t = np.linspace(0, T, 400)

# Производственная функция
def f(x):
    return x / (x + a)

# -----------------------------
# ЗАДАНИЕ 1 — режимы роста
# -----------------------------
gamma_fast = 0.08     # ускоренный рост
gamma_slow = 0.02     # замедленный рост
gamma_decline = 0.005 # спад

gammas = [gamma_fast, gamma_slow, gamma_decline]
labels = ["Ускоренный рост γ=0.08", "Замедленный рост γ=0.02", "Спад γ=0.005"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

plt.figure(figsize=(12, 7))

for gamma, label, col in zip(gammas, labels, colors):
    M = M0 * np.exp(gamma * t)
    plt.plot(t, M, label=label, color=col)

plt.title("Траектории производственной мощности M(t)")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# ВЫПУСК Y(t)
# -----------------------------
plt.figure(figsize=(12, 7))

for gamma, label, col in zip(gammas, ["Ускоренный рост", "Замедленный рост", "Спад"], colors):
    M = M0 * np.exp(gamma * t)
    x = R0 / M
    Y = M * f(x)
    plt.plot(t, Y, label=label, color=col)

plt.title("Выпуск Y(t) при разных режимах роста")
plt.xlabel("t")
plt.ylabel("Y(t)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Душевое потребление c(t)
# -----------------------------
s_star = 0.25

plt.figure(figsize=(12, 7))
for gamma, label, col in zip(gammas,
                             ["Ускоренный рост", "Сбалансированный рост", "Экономический спад"],
                             colors):
    M = M0 * np.exp(gamma * t)
    x = R0 / M
    Y = M * f(x)
    c = (1 - s_star) * Y / R0
    plt.plot(t, c, label=label, color=col)

plt.title("Душевое потребление c(t)")
plt.xlabel("t")
plt.ylabel("c(t)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# ЗАДАНИЕ 2 — влияние нормы накопления s
# -----------------------------
s_values = np.linspace(0, 0.9, 10)
max_c = []

gamma = gamma_slow   # рассматриваем устойчивый рост

for s in s_values:
    M = M0 * np.exp(gamma * t)
    x = R0 / M
    Y = M * f(x)
    c = (1 - s) * Y / R0
    max_c.append(max(c))

plt.figure(figsize=(12, 7))
plt.plot(s_values, max_c, marker='o')
plt.title("Максимальное душевое потребление c(t) в зависимости от нормы накопления s")
plt.xlabel("Норма накопления s")
plt.ylabel("max c(t)")
plt.grid(True)
plt.show()

# 3D-поверхность c(t, s)
s_grid = np.linspace(0, 0.9, 50)
t_grid = np.linspace(0, T, 200)
S, Tm = np.meshgrid(s_grid, t_grid)

M_grid = M0 * np.exp(gamma * Tm)
X_grid = R0 / M_grid
Y_grid = M_grid * f(X_grid)
C_grid = (1 - S) * Y_grid / R0

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, Tm, C_grid, cmap='viridis')

ax.set_title("Поверхность c(t, s)")
ax.set_xlabel("s")
ax.set_ylabel("t")
ax.set_zlabel("c(t,s)")
plt.show()
