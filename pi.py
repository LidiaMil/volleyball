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

    return 4 * inside / num_points, xs_in, ys_in, xs_out, ys_out


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

    return 4 * inside / num_points, xs_in, ys_in, xs_out, ys_out


# ===================================================================
#  РИСОВАНИЕ ИЛЛЮСТРАЦИЙ
# ===================================================================

# -----------------------------
# 1) Полный круг
# -----------------------------
pi_full, xs_in, ys_in, xs_out, ys_out = monte_carlo_full_circle(5000)

plt.figure(figsize=(6, 6))
plt.scatter(xs_out, ys_out, s=2)
plt.scatter(xs_in, ys_in, s=2)
plt.title("Монте-Карло: Полный круг (5000 точек)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.savefig("full_circle_monte_carlo.png", dpi=200)
plt.show()


# -----------------------------
# 2) Четверть круга
# -----------------------------
pi_quarter, xs_in_q, ys_in_q, xs_out_q, ys_out_q = monte_carlo_quarter_circle(5000)

plt.figure(figsize=(6, 6))
plt.scatter(xs_out_q, ys_out_q, s=2)
plt.scatter(xs_in_q, ys_in_q, s=2)
plt.title("Монте-Карло: Четверть круга (5000 точек)")
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
