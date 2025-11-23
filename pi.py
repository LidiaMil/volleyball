# import random
# import math
# import matplotlib.pyplot as plt

# # ===================================================================
# #  Функции Монте-Карло
# # ===================================================================

# def monte_carlo_full_circle(num_points):
#     inside = 0
#     xs_out, ys_out = [], []
#     xs_in, ys_in = [], []

#     for _ in range(num_points):
#         x = random.uniform(-1, 1)
#         y = random.uniform(-1, 1)

#         if x*x + y*y <= 1:
#             inside += 1
#             xs_in.append(x)
#             ys_in.append(y)
#         else:
#             xs_out.append(x)
#             ys_out.append(y)

#     pi_est = 4 * inside / num_points
#     return pi_est, xs_in, ys_in, xs_out, ys_out


# def monte_carlo_quarter_circle(num_points):
#     inside = 0
#     xs_out, ys_out = [], []
#     xs_in, ys_in = [], []

#     for _ in range(num_points):
#         x = random.uniform(0, 1)
#         y = random.uniform(0, 1)

#         if x*x + y*y <= 1:
#             inside += 1
#             xs_in.append(x)
#             ys_in.append(y)
#         else:
#             xs_out.append(x)
#             ys_out.append(y)

#     pi_est = 4 * inside / num_points
#     return pi_est, xs_in, ys_in, xs_out, ys_out

# N = 5000   # количество точек

# # -----------------------------
# # 1) Полный круг
# # -----------------------------
# pi_full, xs_in, ys_in, xs_out, ys_out = monte_carlo_full_circle(N)

# plt.figure(figsize=(6, 6))
# plt.scatter(xs_out, ys_out, s=2)
# plt.scatter(xs_in, ys_in, s=2)
# plt.title(f"Монте-Карло: Полный круг ({N} точек)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.gca().set_aspect("equal")
# plt.savefig("full_circle_monte_carlo.png", dpi=200)
# plt.show()

# # -----------------------------
# # 2) Четверть круга
# # -----------------------------
# pi_quarter, xs_in_q, ys_in_q, xs_out_q, ys_out_q = monte_carlo_quarter_circle(N)

# plt.figure(figsize=(6, 6))
# plt.scatter(xs_out_q, ys_out_q, s=2)
# plt.scatter(xs_in_q, ys_in_q, s=2)
# plt.title(f"Монте-Карло: Четверть круга ({N} точек)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.gca().set_aspect("equal")
# plt.savefig("quarter_circle_monte_carlo.png", dpi=200)
# plt.show()

# # -----------------------------
# # 3) График сходимости π
# # -----------------------------
# def convergence_test(n_values):
#     estimates = []
#     for n in n_values:
#         pi_est, _, _, _, _ = monte_carlo_full_circle(n)
#         estimates.append(pi_est)
#     return estimates

# n_values = [100, 300, 1000, 3000, 10000, 30000]
# estimates = convergence_test(n_values)

# plt.figure(figsize=(8, 5))
# plt.plot(n_values, estimates)
# plt.axhline(math.pi)
# plt.title("Сходимость метода Монте-Карло к числу π")
# plt.xlabel("Количество случайных точек")
# plt.ylabel("Оценка π")
# plt.savefig("convergence_plot.png", dpi=200)
# plt.show()

# true_pi = math.pi

# # ===================================================================
# #  ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ТОЧНОСТИ ПРИ РАЗНЫХ N
# # ===================================================================

# def analyze_accuracy():
#     true_pi = math.pi
#     test_values = [20, 100, 300, 1000, 3000, 5000, 10000, 50000, 100000]

#     print("\nАнализ точности при разных размерах выборки:")
#     print("Точки (N)   |   Приближение π   |   Абсолютная ошибка")
#     print("---------------------------------------------------------")

#     for n in test_values:
#         pi_est, _, _, _, _ = monte_carlo_full_circle(n)
#         error = abs(pi_est - true_pi)

#         print(f"{n:<11} |   {pi_est:>10.5f}      |      {error:>10.5f}")

#     print("---------------------------------------------------------")
#     print("Замечание: увеличение числа случайных точек уменьшает разброс\n"
#           "результатов и повышает точность оценки числа π.\n")

# # Запускаем анализ точности
# analyze_accuracy()

# print("\nРезультаты оценки π методом Монте-Карло:")
# print(f"Четверть круга:  π ≈ {pi_quarter:.5f}   (ошибка: {abs(pi_quarter - true_pi):.5f})")
# print(f"Полный круг:     π ≈ {pi_full:.5f}     (ошибка: {abs(pi_full - true_pi):.5f})")
# print(f"Точное значение: π = {true_pi:.5f}")






# import random
# import math
# import matplotlib.pyplot as plt


# # -------------------------------------------------------
# # 1. Функция моделирования бросания иглы Бюффона
# # -------------------------------------------------------

# def buffon_needle(num_throws, L=1.0, d=1.0, log=False):
#     """
#     Моделирование задачи Бюффона.
#     :param num_throws: количество бросков
#     :param L: длина иглы
#     :param d: расстояние между линиями (d >= L)
#     :param log: выводить ли подробный лог
#     :return: оценка pi, число пересечений
#     """

#     if L > d:
#         raise ValueError("Условие задачи Бюффона: длина иглы L должна быть <= расстояния между линиями d")

#     hits = 0  # число пересечений линий

#     for i in range(num_throws):
#         # угол от 0 до pi/2
#         theta = random.uniform(0, math.pi / 2)

#         # расстояние от центра иглы до ближайшей линии
#         x = random.uniform(0, d / 2)

#         # условие пересечения: x <= (L/2) * sin(theta)
#         if x <= (L / 2) * math.sin(theta):
#             hits += 1

#     if hits == 0:
#         return None, 0

#     # формула оценки π
#     pi_est = (2 * L * num_throws) / (hits * d)

#     if log:
#         print(f"\n=== ЛОГИ БРОСКОВ (Buffon) ===")
#         print(f"Количество бросков: {num_throws}")
#         print(f"Длина иглы L = {L}")
#         print(f"Расстояние между линиями d = {d}")
#         print(f"Пересечений линий = {hits}")
#         print(f"Оценка π = {pi_est:.5f}")
#         print(f"Ошибка = {abs(pi_est - math.pi):.5f}")

#     return pi_est, hits


# # -------------------------------------------------------
# # 2. Иллюстрация бросков иглы
# # -------------------------------------------------------

# def plot_needles(num_needles=100, L=1.0, d=1.0):
#     """
#     Создание иллюстрации бросков иглы на расчерченной плоскости.
#     """

#     fig, ax = plt.subplots(figsize=(8, 6))

#     # рисуем параллельные линии
#     for y in range(-2, 4):
#         ax.axhline(y * d, color="black", linewidth=1)

#     # генерируем иглы
#     for _ in range(num_needles):
#         theta = random.uniform(0, math.pi)
#         x_center = random.uniform(-2, 2)
#         y_center = random.uniform(-2, 2)

#         dx = (L / 2) * math.cos(theta)
#         dy = (L / 2) * math.sin(theta)

#         x1, y1 = x_center - dx, y_center - dy
#         x2, y2 = x_center + dx, y_center + dy

#         ax.plot([x1, x2], [y1, y2], color="blue", alpha=0.7)

#     ax.set_title("Иллюстрация задачи Бюффона (случайные броски иглы)")
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.grid(False)
#     plt.show()


# # -------------------------------------------------------
# # 3. График зависимости точности расчёта π
# # -------------------------------------------------------

# def plot_pi_convergence(L=1.0, d=1.0):
#     N_values = [50, 100, 300, 500, 1000, 3000, 5000, 10000]
#     pi_values = []
#     errors = []

#     for N in N_values:
#         pi_est, _ = buffon_needle(N, L, d)
#         pi_values.append(pi_est)
#         errors.append(abs(pi_est - math.pi))

#     plt.figure(figsize=(10, 6))
#     plt.plot(N_values, pi_values, marker='o')
#     plt.axhline(math.pi, color='red', linestyle='--', label="Точное значение π")
#     plt.title("Сходимость метода Бюффона при вычислении π")
#     plt.xlabel("Количество бросков")
#     plt.ylabel("Оценка π")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(10, 6))
#     plt.plot(N_values, errors, marker='o', color="purple")
#     plt.title("Ошибка вычисления π в зависимости от количества бросков")
#     plt.xlabel("Количество бросков")
#     plt.ylabel("Абсолютная ошибка")
#     plt.grid(True)
#     plt.show()


# # -------------------------------------------------------
# # 4. Основной запуск с логами
# # -------------------------------------------------------

# if __name__ == "__main__":
#     # иллюстрация
#     plot_needles(num_needles=70, L=1.0, d=1.0)

#     # логирование
#     buffon_needle(3000, L=1.0, d=1.0, log=True)

#     # графики сходимости
#     plot_pi_convergence(L=1.0, d=1.0)



# === Исследование зависимости точности от отношения L/d ===

import numpy as np
import matplotlib.pyplot as plt
import math

def buffon_pi_single_experiment(num_throws, L, d):
    crossings = 0
    for _ in range(num_throws):
        y = np.random.uniform(0, d / 2)             # расстояние от центра иглы до ближайшей прямой
        alpha = np.random.uniform(0, np.pi / 2)     # угол иглы
        if y <= (L / 2) * np.sin(alpha):
            crossings += 1
    if crossings == 0:
        return None
    return (2 * L * num_throws) / (crossings * d)


# ---- параметры ----
N = 5000                     # фиксированное число бросков
d = 1.0                      # расстояние между линиями
ratios = [0.2, 0.4, 0.6, 0.8, 1.0]   # L/d

pi_values = []
errors = []

for r in ratios:
    L = r * d
    pi_est = buffon_pi_single_experiment(N, L, d)
    pi_values.append(pi_est)
    errors.append(abs(pi_est - math.pi))

# ---- График оценки π в зависимости от L/d ----
plt.figure(figsize=(10, 6))
plt.plot(ratios, pi_values, marker='o')
plt.axhline(math.pi, color='r', linestyle='--', label="Точное значение π")
plt.title("Зависимость оценки π от отношения L/d")
plt.xlabel("Отношение L/d")
plt.ylabel("Оценка π")
plt.grid(True)
plt.legend()
plt.show()

# ---- График ошибки ----
plt.figure(figsize=(10, 6))
plt.plot(ratios, errors, marker='o', color="purple")
plt.title("Ошибка вычисления π в зависимости от отношения L/d")
plt.xlabel("Отношение L/d")
plt.ylabel("Абсолютная ошибка")
plt.grid(True)
plt.show()

# ---- Логи ----
print("\n=== ЛОГИ ОТНОШЕНИЯ L/d ===")
for i in range(len(ratios)):
    print(f"L/d = {ratios[i]:.2f} | Оценка π = {pi_values[i]:.5f} | Ошибка = {errors[i]:.5f}")
