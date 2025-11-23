# import numpy as np
# import matplotlib.pyplot as plt

# # ----------------------------------------------------------
# # Реалистичные параметры двухступенчатой ракеты
# # ----------------------------------------------------------
# m_c1 = 1800        # конструкционная масса 1-й ступени
# m_c2 = 900         # конструкционная масса 2-й ступени
# m_payload = 1200   # полезная нагрузка

# fuel_total = 5000  # общий запас топлива (который распределяем)

# ve1 = 3000         # скорость истечения газов первой ступени
# ve2 = 4100         # скорость истечения газов второй ступени


# # ----------------------------------------------------------
# # Функция итоговой скорости ракеты
# # ----------------------------------------------------------
# def rocket_velocity(alpha):
#     """
#     alpha — доля топлива, приходящаяся на 1-ю ступень
#     возвращает итоговую скорость ракеты
#     """

#     m_t1 = fuel_total * alpha
#     m_t2 = fuel_total * (1 - alpha)

#     # Нельзя, чтобы топлива не было
#     if m_t1 <= 0 or m_t2 <= 0:
#         return np.nan

#     # Массы на этапах
#     m0 = m_c1 + m_t1 + m_c2 + m_t2 + m_payload
#     m1 = m_c2 + m_t2 + m_payload
#     m2 = m_payload

#     # ΔV по ступеням
#     dV1 = ve1 * np.log(m0 / m1)
#     dV2 = ve2 * np.log(m1 / m2)

#     return dV1 + dV2


# # ----------------------------------------------------------
# # Диапазон значений α
# # ----------------------------------------------------------
# alphas = np.linspace(0.05, 0.95, 100)
# velocities = np.array([rocket_velocity(a) for a in alphas])

# # Оптимум
# idx = np.nanargmax(velocities)
# alpha_opt = alphas[idx]
# v_opt = velocities[idx]

# print(f"Оптимальная доля топлива: {alpha_opt:.3f}")
# print(f"Максимальная скорость: {v_opt:.1f} м/с")


# # ----------------------------------------------------------
# # График 1 — влияние топлива первой ступени
# # ----------------------------------------------------------
# plt.figure(figsize=(9, 5))
# plt.plot(alphas, velocities, linewidth=2)
# plt.scatter(alpha_opt, v_opt, color='red')

# plt.text(alpha_opt, v_opt,
#          f"  α = {alpha_opt:.2f}\n  V = {v_opt:.1f} м/с",
#          fontsize=10, va='bottom')

# plt.title("Зависимость конечной скорости от доли топлива в первой ступени")
# plt.xlabel("Доля топлива в 1-й ступени, α")
# plt.ylabel("Конечная скорость, м/с")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # ----------------------------------------------------------
# # График 2 — зависимость от топлива второй ступени (β = 1−α)
# # ----------------------------------------------------------
# betas = 1 - alphas   # доля топлива во второй ступени

# plt.figure(figsize=(9, 5))
# plt.plot(betas, velocities, linewidth=2)

# plt.title("Зависимость конечной скорости от доли топлива во второй ступени")
# plt.xlabel("Доля топлива во 2-й ступени, β")
# plt.ylabel("Конечная скорость, м/с")
# plt.grid(True)
# plt.tight_layout()
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------------------------
# # Параметры ступеней
# # -----------------------------------------------
# m_s1 = 15000
# m_s2 = 6000
# m_s3 = 3000
# m_payload = 2000

# Ve1 = 2800
# Ve2 = 3200
# Ve3 = 3600

# mdot1 = mdot2 = mdot3 = 200.0

# total_fuel = 24000

# fuel_scenarios = [
#     (12000, 8000, 4000),
#     (8000, 8000, 8000),
#     (6000, 10000, 8000),
#     (4000, 8000, 12000),
#     (14000, 6000, 4000)
# ]

# def simulate_three_stage(mf1, mf2, mf3, dt=0.1):
#     fuel = [mf1, mf2, mf3]
#     dry  = [m_s1, m_s2, m_s3]
#     Ve   = [Ve1, Ve2, Ve3]
#     mdot = [mdot1, mdot2, mdot3]

#     m = m_payload + sum(dry) + sum(fuel)
#     V = 0.0
#     t = 0.0

#     times = [t]
#     velocities = [V]

#     stage = 0

#     # сюда запишем моменты отделения ступеней
#     separation_times = []

#     while stage < 3:
#         burn_time = fuel[stage] / mdot[stage]
#         steps = int(np.ceil(burn_time / dt))
#         burned = 0.0

#         for _ in range(steps):
#             if burned >= fuel[stage]:
#                 break

#             dm = mdot[stage] * dt
#             if burned + dm > fuel[stage]:
#                 dm = fuel[stage] - burned

#             burned += dm
#             m -= dm

#             dV = Ve[stage] * (dm / m)
#             V += dV
#             t += dt

#             times.append(t)
#             velocities.append(V)

#         # фиксируем момент отделения ступени
#         separation_times.append((t, V))

#         # отделяем ступень
#         m -= dry[stage]
#         stage += 1

#     return np.array(times), np.array(velocities), separation_times


# # ------------------------------------------------
# # Построение графика с точками отделения
# # ------------------------------------------------
# plt.figure(figsize=(10, 6))

# for i, (mf1, mf2, mf3) in enumerate(fuel_scenarios, start=1):
#     t_arr, V_arr, sep = simulate_three_stage(mf1, mf2, mf3)

#     plt.plot(t_arr, V_arr / 1000, label=f"Сценарий {i}: {mf1}/{mf2}/{mf3} кг")

#     # наносим точки отделения
#     for k, (ts, Vs) in enumerate(sep):
#         plt.scatter(ts, Vs/1000, marker='o', s=40)
#         plt.text(ts, Vs/1000,
#                  f"  Отделение {k+1}-й",
#                  fontsize=8, verticalalignment='bottom')

# plt.title("Зависимость скорости трёхступенчатой ракеты от времени\n"
#           "с отмеченными точками отделения ступеней")
# plt.xlabel("Время, с")
# plt.ylabel("Скорость, км/с")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# g = 9.8

# # --- параметры ракеты ---
# m_c1, m_c2 = 1500, 800       # конструкционные массы
# m_t1, m_t2 = 3000, 2000      # топливо
# m_n = 1000                   # полезная нагрузка

# Ve1, Ve2 = 2800, 3200        # скорости истечения газов

# def initial_velocity():
#     m0 = m_c1 + m_t1 + m_c2 + m_t2 + m_n
#     m1 = m_c2 + m_t2 + m_n
#     m2 = m_n
#     dV1 = Ve1 * np.log(m0 / m1)
#     dV2 = Ve2 * np.log(m1 / m2)
#     return dV1 + dV2

# def range_ballistic(h0, theta_deg):
#     V0 = initial_velocity()
#     theta = np.radians(theta_deg)
#     Vy0 = V0 * np.sin(theta)
#     Vx0 = V0 * np.cos(theta)
#     T = (Vy0 + np.sqrt(Vy0**2 + 2 * g * h0)) / g
#     R = Vx0 * T
#     return R   # м

# # --- исследование дальности от высоты для разных углов ---
# heights = np.linspace(0, 20000, 21)   # 0–20 км
# angles = [10, 20, 30, 45, 60]         # физически осмысленные углы

# plt.figure(figsize=(9, 5))

# for theta_fixed in angles:
#     ranges = [range_ballistic(h0, theta_fixed) / 1000 for h0 in heights]  # км
#     plt.plot(heights / 1000, ranges, marker='o', label=f"{theta_fixed}°")

# plt.title("Зависимость дальности полёта от начальной высоты\nдля разных углов запуска")
# plt.xlabel("Начальная высота, км")
# plt.ylabel("Дальность, км")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------
# ПАРАМЕТРЫ РАКЕТЫ (упрощённая модель)
# --------------------------------------
V0 = 3000  # скорость после работы ступеней
g = 9.81   # ускорение свободного падения

# Модель: скорость зависит от распределения топлива
def final_velocity(alpha):
    return V0 * (1 - 0.3 * alpha)

# Баллистическая дальность
def range_ballistic(h0, theta_deg, alpha):
    theta = np.radians(theta_deg)
    V = final_velocity(alpha)

    t_flight = (V*np.sin(theta) + np.sqrt((V*np.sin(theta))**2 + 2*g*h0)) / g
    R = V * np.cos(theta) * t_flight
    return R / 1000  # км


# -------------------------------------------------------------------------------------
# 1) R(θ) при разных высотах h0 и разных α  (как на твоём графике — три подграфика)
# -------------------------------------------------------------------------------------
angles = np.arange(10, 85, 5)
heights = [0, 10000, 20000]              # 0, 10, 20 км
alphas = [0.2, 0.5, 0.8]

plt.figure(figsize=(18, 6))

for i, h0 in enumerate(heights):
    plt.subplot(1, 3, i+1)
    for alpha in alphas:
        R = [range_ballistic(h0, th, alpha) for th in angles]
        plt.plot(angles, R, marker='o', label=f'α = {alpha}')
    plt.title(f"Начальная высота: {h0/1000} км")
    plt.xlabel("Угол запуска, градусы")
    plt.ylabel("Дальность, км")
    plt.grid(True)
    plt.legend()

plt.suptitle("Зависимость дальности от угла, высоты и распределения топлива", fontsize=16)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------
# 2) R(h0) при разных углах θ и разных α
# -------------------------------------------------------------------------------------
h0_vals = np.linspace(0, 20000, 25)

plt.figure(figsize=(18, 6))

for i, alpha in enumerate(alphas):
    plt.subplot(1, 3, i+1)
    for theta in [20, 30, 40, 45, 60]:
        R = [range_ballistic(h0, theta, alpha) for h0 in h0_vals]
        plt.plot(h0_vals/1000, R, marker='o', label=f'θ = {theta}°')
    plt.title(f"Доля топлива α = {alpha}")
    plt.xlabel("Начальная высота, км")
    plt.ylabel("Дальность, км")
    plt.grid(True)
    plt.legend()

plt.suptitle("Зависимость дальности от высоты для разных углов и α", fontsize=16)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------
# 3) R(α) при разных углах θ и разных высотах h0
# -------------------------------------------------------------------------------------
alpha_vals = np.linspace(0.1, 0.9, 25)

plt.figure(figsize=(18, 6))

for i, h0 in enumerate(heights):
    plt.subplot(1, 3, i+1)
    for theta in [20, 30, 40, 45, 60]:
        R = [range_ballistic(h0, theta, a) for a in alpha_vals]
        plt.plot(alpha_vals, R, marker='o', label=f'θ = {theta}°')
    plt.title(f"Начальная высота: {h0/1000} км")
    plt.xlabel("Доля топлива α")
    plt.ylabel("Дальность, км")
    plt.grid(True)
    plt.legend()

plt.suptitle("Зависимость дальности от распределения топлива при разных углах и высотах", fontsize=16)
plt.tight_layout()
plt.show()
