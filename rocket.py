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


import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# Фиксированные параметры ступеней и нагрузки
# -----------------------------------------------
m_s1 = 15000   # сухая масса 1-й ступени
m_s2 = 6000    # сухая масса 2-й ступени
m_s3 = 3000    # сухая масса 3-й ступени
m_payload = 2000  # ПН

# Скорости истечения газов
Ve1 = 2800
Ve2 = 3200
Ve3 = 3600

# Расходы топлива (постоянные)
mdot1 = mdot2 = mdot3 = 200.0

# Общая масса топлива
total_fuel = 24000

# 5 различных сценариев распределения топлива
fuel_scenarios = [
    (12000, 8000, 4000),   # Сценарий 1: преимущество 1-й ступени
    (8000,  8000, 8000),   # Сценарий 2: равномерное распределение
    (6000,  10000, 8000),  # Сценарий 3: максимум топлива во 2-й
    (4000,  8000, 12000),  # Сценарий 4: акцент на 3-й ступени
    (14000, 6000, 4000),   # Сценарий 5: сильный перекос в пользу 1-й
]

# Проверка суммарного топлива
for mf1, mf2, mf3 in fuel_scenarios:
    assert abs((mf1 + mf2 + mf3) - total_fuel) < 1e-6

def simulate_three_stage(mf1, mf2, mf3, dt=0.1):
    """Расчёт зависимости скорости от времени для трёхступенчатой ракеты."""
    fuel = [mf1, mf2, mf3]
    dry  = [m_s1, m_s2, m_s3]
    Ve   = [Ve1, Ve2, Ve3]
    mdot = [mdot1, mdot2, mdot3]

    # Начальная масса
    m = m_payload + sum(dry) + sum(fuel)
    V = 0.0
    t = 0.0

    times = [t]
    velocities = [V]

    stage = 0

    while stage < 3:
        burn_time = fuel[stage] / mdot[stage]
        steps = int(np.ceil(burn_time / dt))
        burned = 0.0

        for _ in range(steps):
            if burned >= fuel[stage]:
                break

            dm = mdot[stage] * dt
            if burned + dm > fuel[stage]:
                dm = fuel[stage] - burned

            burned += dm
            m_before = m
            m -= dm

            dV = Ve[stage] * (dm / m)
            V += dV
            t += dt

            times.append(t)
            velocities.append(V)

        # отделение ступени
        m -= dry[stage]
        stage += 1

    return np.array(times), np.array(velocities)


# ------------------------------------------------
# Построение графиков V(t) для всех сценариев
# ------------------------------------------------
plt.figure(figsize=(10, 6))

for i, (mf1, mf2, mf3) in enumerate(fuel_scenarios, start=1):
    t_arr, V_arr = simulate_three_stage(mf1, mf2, mf3)
    plt.plot(t_arr, V_arr / 1000, label=f"Сценарий {i}: {mf1}/{mf2}/{mf3} кг")

plt.title("Зависимость скорости трёхступенчатой ракеты от времени\nдля разных распределений топлива")
plt.xlabel("Время, с")
plt.ylabel("Скорость, км/с")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
