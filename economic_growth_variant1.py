import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# ПАРАМЕТРЫ МОДЕЛИ
# ============================================================

alpha = 0.05          # темп роста работников (λ)
gamma = 0.08          # ввод мощностей
beta  = 0.02          # выбытие мощностей
mu = gamma - beta      # темп роста мощности μ

R0 = 100
M0 = 50
a = 1.0               # параметр производственной функции
s = 0.3               # норма накопления

t = np.linspace(0, 50, 800)

# ============================================================
# ФУНКЦИИ МОДЕЛИ
# ============================================================

def R(t):
    return R0 * np.exp(alpha * t)

def M(t):
    return M0 * np.exp(mu * t)

def x(t):
    return R(t) / M(t)

def f(x):
    return x / (x + a)

def Y(t):
    return M(t) * f(x(t))

def omega(t):
    return (1 - s) * Y(t)

def A(t):
    return s * Y(t)

def c(t):
    return omega(t) / R(t)

# ============================================================
# 1. ГРАФИКИ R(t) и M(t)
# ============================================================

plt.figure(figsize=(10,6))
plt.plot(t, R(t), label="R(t) — работники")
plt.plot(t, M(t), label="M(t) — мощность")
plt.title("Рост числа работников и производственной мощности")
plt.grid()
plt.legend()
plt.show()

# ============================================================
# 2. Коэффициент загрузки мощности x(t)
# ============================================================

plt.figure(figsize=(10,6))
plt.plot(t, x(t))
plt.title("Динамика коэффициента загрузки мощности x(t) = R/M")
plt.grid()
plt.show()

# ============================================================
# 3. Производственная функция f(x)
# ============================================================

xx = np.linspace(0, 10, 500)

plt.figure(figsize=(10,6))
plt.plot(xx, f(xx))
plt.title("Производственная функция f(x) = x/(x+a)")
plt.grid()
plt.show()

# ============================================================
# 4. Выпуск Y(t)
# ============================================================

plt.figure(figsize=(10,6))
plt.plot(t, Y(t))
plt.title("Выпуск Y(t)")
plt.grid()
plt.show()

# ============================================================
# 5. Потребление ω(t) и инвестиции A(t)
# ============================================================

plt.figure(figsize=(10,6))
plt.plot(t, omega(t), label="Потребление ω(t)")
plt.plot(t, A(t), label="Инвестиции A(t)")
plt.title("Потребление и инвестиции")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# 6. Душевое потребление c(t) для разных режимов роста
# ============================================================

def c_regime(lam, mu, t):
    xt = x0 * np.exp((lam - mu) * t)
    return (1 - s) * f(xt) / xt

x0 = R0 / M0

lam_fast = 0.06
mu_fast = 0.03

lam_eq = 0.05
mu_eq = 0.05

lam_slow = 0.03
mu_slow = 0.06

plt.figure(figsize=(10,6))
plt.plot(t, c_regime(lam_fast, mu_fast, t), label="Ускоренный рост λ > μ")
plt.plot(t, c_regime(lam_eq,   mu_eq,   t), label="Сбалансированный рост λ = μ")
plt.plot(t, c_regime(lam_slow, mu_slow, t), label="Замедленный рост λ < μ")
plt.title("Душевое потребление в трёх режимах роста")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# 7. Зависимость максимального c от нормы накопления s
# ============================================================

s_values = np.linspace(0.01, 0.99, 200)
max_c = []

for s_temp in s_values:
    cc = (1 - s_temp) * Y(t) / R(t)
    max_c.append(np.max(cc))

plt.figure(figsize=(10,6))
plt.plot(s_values, max_c)
plt.title("Максимальное душевое потребление в зависимости от нормы накопления s")
plt.xlabel("Норма накопления s")
plt.ylabel("max c(t)")
plt.grid()
plt.show()

# ============================================================
# 8. 3D график c(t, s)
# ============================================================

S, T = np.meshgrid(s_values, t)
C = (1 - S) * (M(T) * f(R(T)/M(T))) / R(T)

fig = plt.figure(figsize=(11,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, T, C, cmap="viridis")
ax.set_xlabel("s")
ax.set_ylabel("t")
ax.set_zlabel("c(t,s)")
ax.set_title("Поверхность душевого потребления c(t, s)")
plt.show()
