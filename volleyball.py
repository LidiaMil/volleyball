import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Параметры площадки и сетки
# -----------------------------
court_length = 18.0    # длина площадки (м)
court_width = 9.0      # ширина площадки (м)
net_x = court_length / 2
net_height = 2.43      # высота сетки (м)

# -----------------------------
# Физика
# -----------------------------
g = 9.81  # м/с^2

def simulate_throw(x0, y0, v0, alpha, phi, z0, n_points=200):
    """Возвращает траекторию и факт перелёта сетки"""
    vx = v0 * np.cos(alpha) * np.cos(phi)
    vy = v0 * np.cos(alpha) * np.sin(phi)
    vz = v0 * np.sin(alpha)

    # время полёта (решаем z=0)
    A = -0.5 * g
    B = vz
    C = z0
    roots = np.roots([A, B, C])
    roots = roots[np.isreal(roots)]
    t_flight = np.max(roots)  # берём положительный корень

    t = np.linspace(0, t_flight, n_points)
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t - 0.5 * g * t**2

    # высота над сеткой
    crosses_net = False
    if vx > 0:
        t_net = (net_x - x0) / vx
        if 0 < t_net < t_flight:
            z_net = z0 + vz * t_net - 0.5 * g * t_net**2
            crosses_net = z_net > net_height

    return x, y, z, crosses_net, (x[-1], y[-1])

def check_success(x0, y0, v0, alpha, phi, z0):
    """Успех = перелет сетки + приземление в пределах площадки соперника"""
    x, y, z, crosses_net, (x_land, y_land) = simulate_throw(x0, y0, v0, alpha, phi, z0)
    lands_in_court = (net_x <= x_land <= court_length) and (0 <= y_land <= court_width)
    return crosses_net and lands_in_court

# -----------------------------
# Основные параметры (можно менять)
# -----------------------------
v0 = 15.0              # м/с
alpha = np.deg2rad(40) # угол вверх
phi = np.deg2rad(0)   # угол вбок
z0 = 1.8               # м (атака в прыжке)

# -----------------------------
# Сетка начальных позиций
# -----------------------------
x_positions = np.linspace(0.5, net_x - 0.5, 30)
y_positions = np.linspace(0.5, court_width - 0.5, 30)
XX, YY = np.meshgrid(x_positions, y_positions)
zone = np.zeros_like(XX)

for i in range(len(y_positions)):
    for j in range(len(x_positions)):
        if check_success(x_positions[j], y_positions[i], v0, alpha, phi, z0):
            zone[i, j] = 1
        else:
            zone[i, j] = -1

# -----------------------------
# Визуализация
# -----------------------------
fig = plt.figure(figsize=(14, 6))

# 1) Карта успеха
ax1 = fig.add_subplot(1, 2, 1)
cmap = plt.colormaps.get_cmap("RdYlGn")
mesh = ax1.pcolormesh(XX, YY, zone, cmap=cmap, shading='auto', vmin=-1, vmax=1)

ax1.axvline(net_x, color="black", linewidth=3, label="Сетка")
ax1.set_xlim(0, court_length/2)
ax1.set_ylim(0, court_width)
ax1.set_title("Зона успеха (X0,Y0)")
ax1.set_xlabel("X0 (м)")
ax1.set_ylabel("Y0 (м)")
ax1.legend()
plt.colorbar(mesh, ax=ax1, label="-1 — неудача, 1 — успех")

# 2) 3D траектория (берем одну точку, напр. середину своей половины)
x0 = court_length/4
y0 = court_width/2
x, y, z, crosses_net, (x_land, y_land) = simulate_throw(x0, y0, v0, alpha, phi, z0)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(x, y, z, color="blue", label="Траектория")
ax2.scatter([x0], [y0], [z0], color="green", s=50, label="Начало")
ax2.scatter([x_land], [y_land], [0], color="red", s=50, label="Приземление")

# Сетка (плоскость)
yy = np.linspace(0, court_width, 10)
zz = np.linspace(0, net_height, 10)
YY_net, ZZ_net = np.meshgrid(yy, zz)
XX_net = np.ones_like(YY_net) * net_x
ax2.plot_surface(XX_net, YY_net, ZZ_net, color="black", alpha=0.3)

ax2.set_xlim(0, court_length)
ax2.set_ylim(0, court_width)
ax2.set_zlim(0, 6)
ax2.set_xlabel("X (м)")
ax2.set_ylabel("Y (м)")
ax2.set_zlabel("Z (м)")
ax2.set_title("Пример траектории в 3D")
ax2.legend()

plt.tight_layout()
plt.show()