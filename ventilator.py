import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # нужно для 3D-графика
import math


# -------------------------------------------------------
# 1. Модель движения вентилятора на пружине
# -------------------------------------------------------

def blade_tip_trajectory(t, Rad, A, y0, omega_rot, omega_spring, phase_rot=0.0, phase_spring=0.0):
    """
    Координаты конца лопасти вентилятора, подвешенного на пружине.

    Система отсчёта K:
      - ось y направлена вертикально (вверх),
      - ось z – горизонтально (вперёд),
      - ось x – перпендикулярно плоскости yz.

    Вентилятор колеблется вдоль оси y по гармоническому закону,
    а конец лопасти вращается по окружности радиуса Rad.

    :param t: массив времён
    :param Rad: радиус вращения конца лопасти
    :param A: амплитуда колебаний пружины по y
    :param y0: положение равновесия по y
    :param omega_rot: угловая частота вращения лопастей
    :param omega_spring: угловая частота колебаний пружины
    :param phase_rot: начальная фаза вращения
    :param phase_spring: начальная фаза колебаний
    :return: массивы x(t), y(t), z(t)
    """

    # вращение лопасти в плоскости xz
    x = Rad * np.cos(omega_rot * t + phase_rot)
    z = Rad * np.sin(omega_rot * t + phase_rot)

    # вертикальные колебания вентилятора на пружине
    y = y0 + A * np.cos(omega_spring * t + phase_spring)

    return x, y, z


# -------------------------------------------------------
# 2. Визуализация траекторий (пункт 1 задачи)
# -------------------------------------------------------

def plot_trajectory_3d_and_projections():
    # Параметры модели (можешь подправить под свои)
    Rad = 1.0          # радиус вращения конца лопасти
    A = 0.5            # амплитуда колебаний пружины
    y0 = 0.0           # положение равновесия по y

    T_spring = 2.0     # период колебаний пружины
    T_rot = 0.5        # период вращения лопастей
    omega_spring = 2 * np.pi / T_spring
    omega_rot = 2 * np.pi / T_rot

    # Временной интервал: берём несколько общих периодов
    t_max = 6 * T_spring
    t = np.linspace(0, t_max, 4000)

    x, y, z = blade_tip_trajectory(t, Rad, A, y0, omega_rot, omega_spring)

    # --- ЛОГИ ПО ПАРАМЕТРАМ ---
    print("=== МОДЕЛЬ ВЕНТИЛЯТОРА НА ПРУЖИНЕ (траектория конца лопасти) ===")
    print(f"Радиус вращения Rad          = {Rad}")
    print(f"Амплитуда колебаний A        = {A}")
    print(f"Положение равновесия y0      = {y0}")
    print(f"Период колебаний пружины T_s = {T_spring}")
    print(f"Период вращения лопастей T_r = {T_rot}")
    print(f"Отношение периодов T_r / T_s = {T_rot / T_spring:.3f}")
    print("-----------------------------------------------------------------\n")

    # --- 3D траектория ---
    fig = plt.figure(figsize=(14, 4))

    ax3d = fig.add_subplot(131, projection='3d')
    ax3d.plot(x, y, z)
    ax3d.set_title("3D-траектория конца лопасти")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    # --- Проекция на плоскость yz (фигура Лиссажу) ---
    ax_yz = fig.add_subplot(132)
    ax_yz.plot(z, y, linewidth=0.7)
    ax_yz.set_title("Проекция траектории на плоскость y–z")
    ax_yz.set_xlabel("z")
    ax_yz.set_ylabel("y")
    ax_yz.grid(True)

    # --- Проекция на плоскость xz ---
    ax_xz = fig.add_subplot(133)
    ax_xz.plot(x, z, linewidth=0.7)
    ax_xz.set_title("Проекция траектории на плоскость x–z")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.grid(True)

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# 3. Исследование фигур Лиссажу (пункт 2 задачи)
#    При каких соотношениях периодов траектория в плоскости yz – Лиссажу?
# -------------------------------------------------------

def explore_lissajous(period_ratio_list, Rad=1.0, A=0.5, y0=0.0, T_spring=2.0):
    """
    Исследование зависимости формы траектории от отношения периодов
    вращения лопастей и колебаний пружины.

    :param period_ratio_list: список пар (p, q) – рациональные отношения периодов
                              T_rot / T_spring = p / q
    :param Rad, A, y0, T_spring: параметры системы
    """

    omega_spring = 2 * np.pi / T_spring

    print("=== ИССЛЕДОВАНИЕ ФИГУР ЛИССАЖУ (проекция y–z) ===")
    print("Формат: p:q – рациональное отношение периодов T_rot / T_spring")
    print("---------------------------------------------------------------")

    n = len(period_ratio_list)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for idx, (p, q) in enumerate(period_ratio_list):
        # Отношение периодов: T_rot / T_spring = p / q
        T_rot = T_spring * p / q
        omega_rot = 2 * np.pi / T_rot

        # Чтобы увидеть замкнутую фигуру Лиссажу, достаточно нескольких общих периодов
        t_max = q * T_spring
        t = np.linspace(0, t_max, 4000)

        x, y, z = blade_tip_trajectory(t, Rad, A, y0, omega_rot, omega_spring)

        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        ax.plot(z, y, linewidth=0.8)
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_title(f"T_rot / T_spring = {p}/{q}")
        ax.grid(True)

        # --- ЛОГИ ПО ДАННОМУ ОТНОШЕНИЮ ---
        print(f"Отношение T_rot / T_spring = {p}/{q} "
              f"≈ {T_rot / T_spring:.3f} | "
              f"ω_rot / ω_spring = {(omega_rot / omega_spring):.3f}")

    print("---------------------------------------------------------------\n")
    plt.suptitle("Фигуры Лиссажу (проекция траектории конца лопасти на плоскость y–z)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# -------------------------------------------------------
# 4. Дополнительное логирование – сравнение замкнутости траекторий
# -------------------------------------------------------

def closedness_measure(Rad=1.0, A=0.5, y0=0.0, T_spring=2.0, p=1, q=1):
    """
    Грубая оценка "замкнутости" траектории: расстояние
    между начальной и конечной точкой после q периодов пружины.
    Чем меньше значение, тем ближе траектория к замкнутой.
    """

    omega_spring = 2 * np.pi / T_spring
    T_rot = T_spring * p / q
    omega_rot = 2 * np.pi / T_rot

    t_max = q * T_spring
    t = np.linspace(0, t_max, 5000)
    x, y, z = blade_tip_trajectory(t, Rad, A, y0, omega_rot, omega_spring)

    dist = math.sqrt((x[0] - x[-1])**2 + (y[0] - y[-1])**2 + (z[0] - z[-1])**2)
    return dist


def log_closedness_examples():
    ratios = [(1, 1), (2, 1), (3, 2), (5, 3)]
    print("=== ОЦЕНКА ЗАМКНУТОСТИ ТРАЕКТОРИЙ ===")
    print("Мера замкнутости – расстояние между первой и последней точкой за общий период.")
    print("Малое значение -> траектория почти замкнута (классическая фигура Лиссажу).")
    print("---------------------------------------------------------------")
    for p, q in ratios:
        dist = closedness_measure(p=p, q=q)
        print(f"T_rot / T_spring = {p}/{q} | мера замкнутости ≈ {dist:.5e}")
    print("---------------------------------------------------------------\n")


# -------------------------------------------------------
# 5. Основной блок запуска
# -------------------------------------------------------

if __name__ == "__main__":
    # Пункт 1: построение модели траектории и иллюстраций
    plot_trajectory_3d_and_projections()

    # Пункт 2: исследование соотношения периодов (фигуры Лиссажу)
    period_ratios = [
        (1, 1),   # равные периоды
        (2, 1),   # вращение в 2 раза быстрее колебаний
        (3, 2),   # отношение частот 3:2
        (5, 3)    # более сложная фигура
    ]
    explore_lissajous(period_ratios)

    # Дополнительные логи по замкнутости траекторий
    log_closedness_examples()
