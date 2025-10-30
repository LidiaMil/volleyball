import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Параметры площадки и физики
# -----------------------
COURT_LENGTH = 18.0      # полная длина площадки (м)
COURT_WIDTH = 9.0        # полная ширина площадки (м)
NET_X = COURT_LENGTH / 2 # положение сетки по X (м)
NET_HEIGHT = 2.43        # высота сетки (м) для мужчин (женщинам 2.24)
G = 9.81                 # ускорение свободного падения (м/с^2)

# -----------------------
# Функции физики
# -----------------------
def flight_time_to_ground(vz, z0):
    """
    Возвращает положительное время до касания земли (z=0) для вертикальной компоненты vz и начальной высоты z0.
    Если корней нет — возвращает None.
    Решает: z0 + vz * t - 0.5 * g * t^2 = 0
    """
    a = -0.5 * G
    b = vz
    c = z0
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)
    ts = [t for t in (t1, t2) if t > 1e-9]
    return max(ts) if ts else None

def simulate_trajectory(x0, y0, z0, v0, alpha_rad, phi_rad, drag_horiz=1.0):
    """
    Симуляция полёта без сопротивления по вертикали, простая модель горизонтального снижения скорости:
      vx = v0 * cos(alpha) * cos(phi) * drag_horiz
      vy = v0 * cos(alpha) * sin(phi) * drag_horiz
      vz = v0 * sin(alpha)
    Возвращает (x_land, y_land, z_net, net_clear(bool), t_land, vx, vy, vz, t_net)
    Если не долетает до сетки / не имеет смысла — возвращает tuple of None/False.
    """
    # компоненты скорости в начале
    cos_a = np.cos(alpha_rad); sin_a = np.sin(alpha_rad)
    cos_p = np.cos(phi_rad);   sin_p = np.sin(phi_rad)
    vx = v0 * cos_a * cos_p * drag_horiz
    vy = v0 * cos_a * sin_p * drag_horiz
    vz = v0 * sin_a

    # если горизонтальная компонента по X направлена не к сетке (vx <= 0), то подача не работает
    if vx <= 1e-9:
        return (None, None, None, False, None, vx, vy, vz, None)

    # время падения
    t_land = flight_time_to_ground(vz, z0)
    if t_land is None:
        return (None, None, None, False, None, vx, vy, vz, None)

    # время когда мяч достигает плоскости сетки X = NET_X
    t_net = (NET_X - x0) / vx
    # если мяч не пересекает плоскость сетки в промежутке (0, t_land), то не подходит
    if not (0 < t_net < t_land):
        return (None, None, None, False, t_land, vx, vy, vz, t_net)

    # высота мяча в момент пересечения плоскости сетки
    z_net = z0 + vz * t_net - 0.5 * G * (t_net**2)
    net_clear = (z_net > NET_HEIGHT + 1e-6)

    # координаты приземления
    x_land = x0 + vx * t_land
    y_land = y0 + vy * t_land

    return (x_land, y_land, z_net, net_clear, t_land, vx, vy, vz, t_net)

def is_successful_serve(x0, y0, z0, v0, alpha_rad, phi_rad, drag_horiz=1.0):
    """Булева проверка: перелетел ли над сеткой и попал ли в площадку соперника."""
    x_land, y_land, z_net, net_clear, t_land, vx, vy, vz, t_net = simulate_trajectory(
        x0, y0, z0, v0, alpha_rad, phi_rad, drag_horiz
    )
    if x_land is None:
        return False
    # попадание в площадку соперника: X между NET_X и COURT_LENGTH, Y между 0 и COURT_WIDTH
    in_opponent = (NET_X <= x_land <= COURT_LENGTH) and (0 <= y_land <= COURT_WIDTH)
    return bool(net_clear and in_opponent)

# -----------------------
# Поиск существует-ли phi для заданной точки подачи (x0,y0)
# -----------------------
def exists_phi_success(x0, y0, z0, v0, alpha_rad,
                       phi_min=-np.pi/3, phi_max=np.pi/3, n_phi=241, drag_horiz=1.0):
    """
    Для заданной позиции подачи (x0,y0,z0), скорости и угла подъёма проверяет,
    существует ли угол направления phi в диапазоне [phi_min, phi_max], при котором подача успешна.
    Возвращает (any_success_bool, best_phi_rad_or_None, best_metrics_dict_or_None)
    """
    phis = np.linspace(phi_min, phi_max, n_phi)
    best_phi = None
    for phi in phis:
        x_land, y_land, z_net, net_clear, t_land, vx, vy, vz, t_net = simulate_trajectory(
            x0, y0, z0, v0, alpha_rad, phi, drag_horiz
        )
        if x_land is None:
            continue
        in_opponent = (NET_X <= x_land <= COURT_LENGTH) and (0 <= y_land <= COURT_WIDTH)
        if net_clear and in_opponent:
            best_phi = phi
            return True, best_phi, {
                'x_land': x_land, 'y_land': y_land, 'z_net': z_net, 't_net': t_net, 't_land': t_land
            }
    return False, None, None

# -----------------------
# Карта зон: по сетке x0 (по длине) и y0 (по ширине) проверяем, где хотя бы один phi даёт успех
# -----------------------
def compute_success_map(v0, alpha_deg, z0,
                        x0_vals=None, y0_vals=None, drag_horiz=1.0,
                        phi_min=-np.pi/3, phi_max=np.pi/3, n_phi=181):
    """
    x0_vals: массив значений по оси X (позиция подачи вдоль длины; обычно на своей половине: [0,..NET_X-min_dist])
    y0_vals: массив значений по оси Y (ширина: 0..COURT_WIDTH)
    Возвращает: X_mesh, Y_mesh, success_mask (1/0)
    """
    if x0_vals is None:
        # допустим, разрешаем позицию подачи от 0.5 м (от задней линии) до NET_X - 0.5
        x0_vals = np.linspace(0.5, NET_X - 0.5, 80)
    if y0_vals is None:
        y0_vals = np.linspace(0.1, COURT_WIDTH - 0.1, 40)

    X, Y = np.meshgrid(x0_vals, y0_vals)
    success_mask = np.zeros_like(X, dtype=int)

    alpha_rad = np.deg2rad(alpha_deg)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0 = X[i, j]; y0 = Y[i, j]
            ok, phi, metrics = exists_phi_success(x0, y0, z0, v0, alpha_rad,
                                                 phi_min=phi_min, phi_max=phi_max,
                                                 n_phi=n_phi, drag_horiz=drag_horiz)
            success_mask[i, j] = 1 if ok else 0
    return X, Y, success_mask

# -----------------------
# Визуализация: слева X-Z (с траекторией), справа X-Y (план) + карта зон подач
# -----------------------
def plot_side_and_top(x0, y0, z0, v0, alpha_deg, phi_deg,
                      X_map=None, Y_map=None, success_mask=None,
                      drag_horiz=1.0, show_phi_marker=True):
    alpha_rad = np.deg2rad(alpha_deg)
    phi_rad = np.deg2rad(phi_deg)

    # симуляция для выбранного угла phi
    traj = simulate_trajectory(x0, y0, z0, v0, alpha_rad, phi_rad, drag_horiz)
    (x_land, y_land, z_net, net_clear, t_land, vx, vy, vz, t_net) = traj

    # вычислим траекторию точками для построения
    if t_land is None:
        t_vals = np.linspace(0, 0.5, 10)
    else:
        t_vals = np.linspace(0, t_land, 300)
    xs = x0 + vx * t_vals
    ys = y0 + vy * t_vals
    zs = z0 + vz * t_vals - 0.5 * G * t_vals**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle(f"Подача: v0={v0:.1f} м/с, α={alpha_deg:.1f}°, φ={phi_deg:.1f}° (drag_horiz={drag_horiz})", fontsize=12)

    # ---- Вид сбоку (X - Z) ----
    ax0 = axes[0]
    ax0.plot(xs, zs, color='dodgerblue', lw=2, label='траектория (проекция)')
    # рисуем сетку как прямоугольник от пола до NET_HEIGHT
    ax0.fill_betweenx([0, NET_HEIGHT], NET_X - 0.08, NET_X + 0.08, color='black', label='сетка (h=2.43m)')
    # поле соперника
    ax0.axvspan(NET_X, COURT_LENGTH, color='lightgreen', alpha=0.25)
    # точки
    ax0.scatter([x0], [z0], color='blue', s=60, label='подача (x0,z0)')
    if x_land is not None:
        ax0.scatter([x_land], [0], color='red', s=50, label='приземление')
        # отметим точку пересечения с сеткой
        if t_net is not None:
            znet = z_net
            ax0.scatter([NET_X], [znet], color='orange', s=40, label=f'z@сетке={znet:.2f}m')
    ax0.set_xlim(0, COURT_LENGTH + 0.5)
    ax0.set_ylim(0, max(4.5, z0 + 1.5))
    ax0.set_xlabel('X (м) — вдоль корта')
    ax0.set_ylabel('Z (м) — высота')
    ax0.set_title('Вид сбоку (X–Z)')
    ax0.grid(alpha=0.3)
    ax0.legend(loc='upper left')

    # ---- Вид сверху / карта зон (X - Y) ----
    ax1 = axes[1]
    # рисуем карту зон, если она передана
    if X_map is not None and Y_map is not None and success_mask is not None:
        # отобразим карту так, что оси будут в метрах: extent задаётся вручную
        im = ax1.pcolormesh(X_map, Y_map, success_mask, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('1 — существует φ для успеха, 0 — нет')
    # границы корта
    ax1.add_patch(plt.Rectangle((0, 0), COURT_LENGTH, COURT_WIDTH, fill=False, edgecolor='k', lw=2))
    # сетка (тонкая полоса)
    ax1.add_patch(plt.Rectangle((NET_X - 0.08, 0), 0.16, COURT_WIDTH, color='black'))
    # отметки подающего и приземления
    ax1.scatter([x0], [y0], color='blue', s=60, label='подача (x0,y0)')
    if x_land is not None and y_land is not None:
        ax1.scatter([x_land], [y_land], color='red', s=50, label='приземление (x_land,y_land)')
        # стрелка направления (показываем в проекции сверху)
        ax1.arrow(x0, y0, x_land - x0, y_land - y0, head_width=0.25, head_length=0.4, fc='navy', ec='navy')
    ax1.set_xlim(0, COURT_LENGTH)
    ax1.set_ylim(0, COURT_WIDTH)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Вид сверху (X–Y), карта зон успешной подачи')
    ax1.grid(alpha=0.25)
    ax1.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# -----------------------
# Набор тест-кейсов + демонстрация логов
# -----------------------
def run_test_cases():
    """
    Набор тестов — разнообразие по высоте z0, скорости v0, углу alpha и месту x0,y0.
    Выводится лог и для каждого кейса строится пара графиков (вид сбоку + вид сверху-карта).
    """
    tests = [
        # (описание, x0, y0, z0, v0, alpha_deg, phi_deg)
        ("Учебная подача (снизу), центр", 4.0, 4.5, 2.0, 14.0, 10.0, 0.0),
        ("Подача сверху (реалистично), центр", 3.5, 4.5, 2.4, 18.0, 8.0, 0.0),
        ("Падающая плоская, мощная", 3.0, 6.0, 3.0, 24.0, 6.0, 0.0),
        ("Угол в сторону (право)", 4.0, 2.0, 2.4, 15.0, 10.0, 12.0),
        ("Угол в сторону (лево)", 4.0, 7.0, 2.4, 15.0, 10.0, -12.0),
        ("Слабая подача, близко к сетке", 7.5, 4.5, 2.0, 10.0, 25.0, 0.0),
        ("Подача в прыжке (высоко)", 2.0, 4.5, 3.1, 26.0, 12.0, 0.0),
        ("Краевая ширина (правый бок)", 5.0, 8.0, 2.4, 16.0, 9.0, 5.0),
    ]

    # параметры карты зон, едино для всех кейсов (можно варьировать)
    x0_vals = np.linspace(0.5, NET_X - 0.5, 60)  # разрешаем старт с задней линии до 0.5 м перед сеткой
    y0_vals = np.linspace(0.1, COURT_WIDTH - 0.1, 40)

    for descr, x0, y0, z0, v0, alpha_deg, phi_deg in tests:
        print("---------------------------------------------------------------")
        print("Кейс:", descr)
        print(f"Параметры: x0={x0:.2f} m, y0={y0:.2f} m, z0={z0:.2f} m, v0={v0:.2f} m/s, alpha={alpha_deg}°, phi={phi_deg}°")

        # прогон нескольких phi для логов (покажем что происходит)
        for test_phi_deg in [phi_deg, 0.0, 10.0, -10.0]:
            alpha_rad = np.deg2rad(alpha_deg)
            phi_rad = np.deg2rad(test_phi_deg)
            out = simulate_trajectory(x0, y0, z0, v0, alpha_rad, phi_rad, drag_horiz=0.9)
            (x_land, y_land, z_net, net_clear, t_land, vx, vy, vz, t_net) = out
            print(f"  phi={test_phi_deg:5.1f}° -> vx={vx:6.2f}, vy={vy:6.2f}, vz={vz:5.2f}, t_land={t_land if t_land else 'N/A'}")
            if x_land is not None:
                print(f"     t_net={t_net:.3f}s, z_net={z_net:.3f}m, x_land={x_land:.2f}, y_land={y_land:.2f}, net_clear={net_clear}")
            else:
                print("     Мяч не пересекает сетку в корректный промежуток (не долетает/неправ.направление)")

        # вычислим карту зон успеха для текущих v0, alpha_deg, z0
        print("  Вычисляем карту зон (это может занять несколько секунд)...")
        Xmap, Ymap, success_mask = compute_success_map(v0, alpha_deg, z0, x0_vals=x0_vals, y0_vals=y0_vals,
                                                      drag_horiz=0.9, phi_min=-np.pi/3, phi_max=np.pi/3, n_phi=181)
        # визуализация (пара графиков)
        plot_side_and_top(x0, y0, z0, v0, alpha_deg, phi_deg, X_map=Xmap, Y_map=Ymap, success_mask=success_mask,
                          drag_horiz=0.9, show_phi_marker=True)

# -----------------------
# Если запускаем напрямую — прогоняем тесты
# -----------------------
if __name__ == "__main__":
    run_test_cases()
