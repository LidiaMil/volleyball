import random
import math
import matplotlib.pyplot as plt

# ===================================================================
#  Функции Монте-Карло
# ===================================================================

def monte_carlo_full_circle(num_points):
    inside = 0
    xs_out, ys_out = [], []
    xs_in, ys_in = [], []

    for _ in range(num_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x*x + y*y <= 1:
            inside += 1
            xs_in.append(x)
            ys_in.append(y)
        else:
            xs_out.append(x)
            ys_out.append(y)

    pi_est = 4 * inside / num_points
    return pi_est, xs_in, ys_in, xs_out, ys_out


def monte_carlo_quarter_circle(num_points):
    inside = 0
    xs_out, ys_out = [], []
    xs_in, ys_in = [], []

    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        if x*x + y*y <= 1:
            inside += 1
            xs_in.append(x)
            ys_in.append(y)
        else:
            xs_out.append(x)
            ys_out.append(y)

    pi_est = 4 * inside / num_points
    return pi_est, xs_in, ys_in, xs_out, ys_out

N = 5000   # количество точек

# -----------------------------
# 1) Полный круг
# -----------------------------
pi_full, xs_in, ys_in, xs_out, ys_out = monte_carlo_full_circle(N)

plt.figure(figsize=(6, 6))
plt.scatter(xs_out, ys_out, s=2)
plt.scatter(xs_in, ys_in, s=2)
plt.title(f"Монте-Карло: Полный круг ({N} точек)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.savefig("full_circle_monte_carlo.png", dpi=200)
plt.show()

# -----------------------------
# 2) Четверть круга
# -----------------------------
pi_quarter, xs_in_q, ys_in_q, xs_out_q, ys_out_q = monte_carlo_quarter_circle(N)

plt.figure(figsize=(6, 6))
plt.scatter(xs_out_q, ys_out_q, s=2)
plt.scatter(xs_in_q, ys_in_q, s=2)
plt.title(f"Монте-Карло: Четверть круга ({N} точек)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.savefig("quarter_circle_monte_carlo.png", dpi=200)
plt.show()

# -----------------------------
# 3) График сходимости π
# -----------------------------
def convergence_test(n_values):
    estimates = []
    for n in n_values:
        pi_est, _, _, _, _ = monte_carlo_full_circle(n)
        estimates.append(pi_est)
    return estimates

n_values = [100, 300, 1000, 3000, 10000, 30000]
estimates = convergence_test(n_values)

plt.figure(figsize=(8, 5))
plt.plot(n_values, estimates)
plt.axhline(math.pi)
plt.title("Сходимость метода Монте-Карло к числу π")
plt.xlabel("Количество случайных точек")
plt.ylabel("Оценка π")
plt.savefig("convergence_plot.png", dpi=200)
plt.show()

true_pi = math.pi

# ===================================================================
#  ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ТОЧНОСТИ ПРИ РАЗНЫХ N
# ===================================================================

def analyze_accuracy():
    true_pi = math.pi
    test_values = [20, 100, 300, 1000, 3000, 5000, 10000, 50000, 100000]

    print("\nАнализ точности при разных размерах выборки:")
    print("Точки (N)   |   Приближение π   |   Абсолютная ошибка")
    print("---------------------------------------------------------")

    for n in test_values:
        pi_est, _, _, _, _ = monte_carlo_full_circle(n)
        error = abs(pi_est - true_pi)

        print(f"{n:<11} |   {pi_est:>10.5f}      |      {error:>10.5f}")

    print("---------------------------------------------------------")
    print("Замечание: увеличение числа случайных точек уменьшает разброс\n"
          "результатов и повышает точность оценки числа π.\n")

# Запускаем анализ точности
analyze_accuracy()

print("\nРезультаты оценки π методом Монте-Карло:")
print(f"Четверть круга:  π ≈ {pi_quarter:.5f}   (ошибка: {abs(pi_quarter - true_pi):.5f})")
print(f"Полный круг:     π ≈ {pi_full:.5f}     (ошибка: {abs(pi_full - true_pi):.5f})")
print(f"Точное значение: π = {true_pi:.5f}")
