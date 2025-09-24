import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Параметры площадки и сетки
# -----------------------------
court_length = 18.0    # длина площадки (м)
court_width = 9.0      # ширина площадки (м)
net_x = court_length / 2  # сетка посередине
net_height = 2.43      # высота сетки (м, для мужчин)

# -----------------------------
# Физические параметры
# -----------------------------
g = 9.81  # ускорение свободного падения (м/с^2)

# -----------------------------
# Функции
# -----------------------------
def simulate_throw(x0, y0, v0, alpha, phi, z0):
    """
    Возвращает координаты приземления и факт перелёта сетки
    """
    vx = v0 * np.cos(alpha) * np.cos(phi)
    vy = v0 * np.cos(alpha) * np.sin(phi)
    vz = v0 * np.sin(alpha)

    # Время до падения (решаем z(t)=0)
    A = -0.5 * g
    B = vz
    C = z0
    roots = np.roots([A, B, C])
    roots = roots[np.isreal(roots)]
    t_flight = np.max(roots)  # берём положительное время

    # Координаты приземления
    x_land = x0 + vx * t_flight
    y_land = y0 + vy * t_flight

    # Проверка пересечения сетки
    crosses_net = False
    if vx > 0:
        t_net = (net_x - x0) / vx
        if 0 < t_net < t_flight:
            z_net = z0 + vz * t_net - 0.5 * g * t_net**2
            crosses_net = z_net > net_height

    return x_land, y_land, crosses_net


def check_success(x0, y0, v0, alpha, phi, z0):
    """
    Успех = перелетел сетку и попал в площадку соперника
    """
    x_land, y_land, crosses_net = simulate_throw(x0, y0, v0, alpha, phi, z0)
    lands_in_court = (net_x <= x_land <= court_length) and (0 <= y_land <= court_width)
    return crosses_net and lands_in_court

# -----------------------------
# Основные параметры броска (можно менять)
# -----------------------------
v0 = 15.0              # начальная скорость (м/с)(в среднем от 15 до 30)
alpha = np.deg2rad(45) # угол броска вверх (0-парал. полу, 90 - вверх)
phi = np.deg2rad(0)    # угол в горизонтальной плоскости(0-по центру площадки соперника,  вправо >0 влево <0)
z0 = 2.30              # рост игрока (высота удара, м)(рост + высота замаха)

# -----------------------------
# Сетка начальных позиций (только своя половина)
# -----------------------------
x_positions = np.linspace(0.5, net_x - 0.5, 40)
y_positions = np.linspace(0.5, court_width - 0.5, 40)
XX, YY = np.meshgrid(x_positions, y_positions)
zone = np.zeros_like(XX)

for i in range(len(y_positions)):
    for j in range(len(x_positions)):
        if check_success(x_positions[j], y_positions[i], v0, alpha, phi, z0):
            zone[i, j] = 1  # успех
        else:
            zone[i, j] = -1 # неудача

# -----------------------------
# Визуализация
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Поле целиком
ax.set_xlim(0, court_length)
ax.set_ylim(0, court_width)

# Сетка (вертикальная линия)
ax.axvline(net_x, color="black", linewidth=3, label="Сетка")

# Раскраска начальных позиций
cmap = plt.colormaps.get_cmap("RdYlGn")  # красный-неудача, зелёный-успех
mesh = ax.pcolormesh(XX, YY, zone, cmap=cmap, shading='auto', vmin=-1, vmax=1)

# Подписи
ax.set_title(f"Зона попадания (v0={v0:.1f} м/с, α={np.rad2deg(alpha):.1f}°, φ={np.rad2deg(phi):.1f}°, рост={z0:.2f} м)")
ax.set_xlabel("X0 (м) — начальная позиция игрока")
ax.set_ylabel("Y0 (м)")
ax.legend()
plt.colorbar(mesh, ax=ax, label="-1 — неудача, 1 — успех")

plt.show()