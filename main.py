import numpy as np
import matplotlib.pyplot as plt

# === Параметры задачи ===
g = 9.81               # ускорение свободного падения, м/с^2
V0 = 18                # начальная скорость броска, м/с
alpha = np.radians(40) # угол броска (в градусах -> радианы)
x0, y0 = 0, 2.2        # начальные координаты (м) (рост игрока ~2.2 м)

# параметры площадки и сетки
court_length = 18      # длина волейбольной площадки (м)
net_x = 9              # положение сетки (середина площадки, м)
net_h = 2.43           # высота сетки (м)

# === Функции движения ===
def x(t):
    return x0 + V0 * np.cos(alpha) * t

def y(t):
    return y0 + V0 * np.sin(alpha) * t - 0.5 * g * t**2

# === Численное моделирование ===
t = np.linspace(0, 3, 500)       # массив времени
X, Y = x(t), y(t)

# момент пересечения с землей (y=0)
ground_indices = np.where(Y <= 0)[0]
if len(ground_indices) > 0:
    landing_index = ground_indices[0]
    x_landing = X[landing_index]
    y_landing = Y[landing_index]
else:
    x_landing, y_landing = None, None

# проверка перелета сетки
t_net = net_x / (V0 * np.cos(alpha))  # время, когда мяч над сеткой
y_net = y(t_net)
over_net = y_net > net_h

# проверка попадания на площадку соперника
on_opponent_side = x_landing is not None and net_x < x_landing < court_length

# === Результаты ===
print("Высота над сеткой:", round(y_net, 2), "м")
print("Перелетел сетку:", "Да" if over_net else "Нет")
if x_landing:
    print("Координата падения:", round(x_landing, 2), "м")
    print("Попал на площадку соперника:", "Да" if on_opponent_side else "Нет")

# === Визуализация ===
plt.figure(figsize=(10, 5))
plt.plot(X, Y, label="Траектория мяча")
plt.axhline(0, color="black")  # земля
plt.axvline(net_x, color="red", linestyle="--", label="Сетка")
plt.axhline(net_h, color="red", linestyle=":", label="Высота сетки")
plt.axvline(court_length, color="blue", linestyle="--", label="Конец площадки")

# точка падения
if x_landing:
    plt.scatter(x_landing, 0, color="green", zorder=5, label="Место падения")

plt.title("Траектория мяча (волейболист)")
plt.xlabel("x (м)")
plt.ylabel("y (м)")
plt.legend()
plt.grid(True)
plt.show()
