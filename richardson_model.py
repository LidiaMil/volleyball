# richardson_model.py
# Модель гонки вооружений Льюиса Ричардсона
# Решает оба пункта задачи:
# 1. Изменчивость M1(t) при разных отношениях α/β
# 2. Изменчивость M2(t) при разных отношениях γ1/γ2

import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# НАСТРОЙКИ МОДЕЛИ
# ------------------------

# коэффициенты "старения"
beta1 = 0.4
beta2 = 0.3

# коэффициенты роста вооружений (пункт 1 будет варьировать alpha1 и alpha2)
alpha_pairs = [
    (0.3, 0.2),
    (0.4, 0.2),
    (0.3, 0.3),
]

# уровни настороженности (пункт 2 будет варьировать gamma1 и gamma2)
gamma_pairs = [
    (1.0, 1.0),
    (1.5, 1.0),
    (1.0, 1.5),
]

# начальные условия
M1_0 = 10
M2_0 = 10

# параметры численного метода
t0, t_end, dt = 0, 50, 0.01
t = np.arange(t0, t_end, dt)

# ------------------------
# ФУНКЦИИ ДЛЯ ОДУ
# ------------------------

def rhs(M, alpha1, alpha2, gamma1, gamma2):
    """Правая часть системы Ричардсона."""
    M1, M2 = M
    dM1 = alpha1 * M2 - beta1 * M1 + gamma1
    dM2 = alpha2 * M1 - beta2 * M2 + gamma2
    return np.array([dM1, dM2])


def euler_solve(alpha1, alpha2, gamma1, gamma2):
    """Численное решение методом Эйлера."""
    M = np.zeros((len(t), 2))
    M[0] = [M1_0, M2_0]

    for i in range(1, len(t)):
        M[i] = M[i-1] + dt * rhs(M[i-1], alpha1, alpha2, gamma1, gamma2)

    return M


# ------------------------
# ПУНКТ 1 — исследование M1(t)
# ------------------------

plt.figure(figsize=(10, 6))
for alpha1, alpha2 in alpha_pairs:
    M = euler_solve(alpha1, alpha2, gamma1=1.0, gamma2=1.0)
    label = f"α1={alpha1}, α2={alpha2}"
    plt.plot(t, M[:, 0], label=label)

plt.title("Пункт 1. Изменчивость M1(t) при разных α/β")
plt.xlabel("t")
plt.ylabel("M1(t)")
plt.grid(True)
plt.legend()
plt.show()


# ------------------------
# ПУНКТ 2 — исследование M2(t)
# ------------------------

alpha1_fixed = 0.3
alpha2_fixed = 0.25

plt.figure(figsize=(10, 6))
for gamma1, gamma2 in gamma_pairs:
    M = euler_solve(alpha1_fixed, alpha2_fixed, gamma1, gamma2)
    ratio = gamma1 / gamma2
    label = f"γ1={gamma1}, γ2={gamma2}, γ1/γ2={ratio:.2f}"
    plt.plot(t, M[:, 1], label=label)

plt.title("Пункт 2. Изменчивость M2(t) при разных γ1/γ2")
plt.xlabel("t")
plt.ylabel("M2(t)")
plt.grid(True)
plt.legend()
plt.show()
