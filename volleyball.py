import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# --- параметры площадки ---
court_length = 18   # длина площадки (м)
court_width = 9     # ширина площадки (м)
net_x = court_length / 2  # координата сетки (по X)
net_height = 2.43   # высота сетки (м)
g = 9.81            # ускорение свободного падения (м/с^2)

# --- начальные параметры ---
x0, y0, z0 = 1.0, 4.5, 2.0   # стартовые координаты (м)
v0 = 15.0                    # начальная скорость (м/с)
alpha = 20.0 * np.pi / 180   # угол к горизонту (рад)
phi = 0.0 * np.pi / 180      # угол в горизонтальной плоскости (рад)

# --- функция траектории ---
def trajectory(x0, y0, z0, v0, alpha, phi, t_max=3.0, n=200):
    t = np.linspace(0, t_max, n)
    vx = v0 * np.cos(alpha) * np.cos(phi)
    vy = v0 * np.cos(alpha) * np.sin(phi)
    vz = v0 * np.sin(alpha)

    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t - 0.5 * g * t**2

    mask = z >= 0
    return x[mask], y[mask], z[mask]

# --- проверка попадания ---
def check_hit(x, y, z):
    for xi, yi, zi in zip(x, y, z):
        if abs(xi - net_x) < 0.1:  # мяч рядом с сеткой
            if zi < net_height:   # не перелетел
                return False
    # проверка: попал ли в площадку соперника
    if x[-1] > net_x and 0 <= y[-1] <= court_width:
        return True
    return False

# --- построение графика ---
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

def update_plot(val=None):
    global v0, alpha, phi, z0
    v0 = slider_v0.val
    alpha = np.deg2rad(slider_alpha.val)
    phi = np.deg2rad(slider_phi.val)
    z0 = slider_z0.val

    ax.clear()
    x, y, z = trajectory(x0, y0, z0, v0, alpha, phi)

    # цвет фона (результат удара)
    hit = check_hit(x, y, z)
    if hit:
        ax.set_facecolor("lightgreen")
    else:
        ax.set_facecolor("lightcoral")

    # --- сетка в виде вертикальной плоскости ---
    y_net = np.linspace(0, court_width, 10)
    z_net = np.linspace(0, net_height, 10)
    Y, Z = np.meshgrid(y_net, z_net)
    X = np.ones_like(Y) * net_x
    ax.plot_surface(X, Y, Z, color="red", alpha=0.4)

    # --- площадка ---
    ax.plot([0, court_length, court_length, 0, 0],
            [0, 0, court_width, court_width, 0],
            [0, 0, 0, 0, 0], "k-")

    # --- траектория ---
    ax.plot(x, y, z, label=f"v0={v0:.1f} м/с, α={np.rad2deg(alpha):.0f}°, φ={np.rad2deg(phi):.0f}°")
    ax.scatter(x0, y0, z0, color="blue", s=60, label="Начальная точка")

    # оформление
    ax.set_xlim(0, court_length)
    ax.set_ylim(0, court_width)
    ax.set_zlim(0, 6)
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_zlabel("Z (м)")
    ax.set_title("Траектория полета волейбольного мяча")
    ax.legend()
    plt.draw()

# --- слайдеры для управления ---
axcolor = "lightgoldenrodyellow"
ax_v0 = plt.axes([0.2, 0.02, 0.65, 0.02], facecolor=axcolor)
ax_alpha = plt.axes([0.2, 0.05, 0.65, 0.02], facecolor=axcolor)
ax_phi = plt.axes([0.2, 0.08, 0.65, 0.02], facecolor=axcolor)
ax_z0 = plt.axes([0.2, 0.11, 0.65, 0.02], facecolor=axcolor)

slider_v0 = Slider(ax_v0, "v0 (м/с)", 1.0, 30.0, valinit=v0)
slider_alpha = Slider(ax_alpha, "α (°)", 1.0, 60.0, valinit=np.rad2deg(alpha))
slider_phi = Slider(ax_phi, "φ (°)", -45.0, 45.0, valinit=np.rad2deg(phi))
slider_z0 = Slider(ax_z0, "z0 (м)", 0.5, 3.0, valinit=z0)

slider_v0.on_changed(update_plot)
slider_alpha.on_changed(update_plot)
slider_phi.on_changed(update_plot)
slider_z0.on_changed(update_plot)

update_plot()
plt.show()
