import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import logging

# ------------------ ЛОГИРОВАНИЕ ------------------
logging.basicConfig(level=logging.INFO, format='[LOG] %(message)s')
log = logging.getLogger()

log.info("=== МОДЕЛЬ МАЯТНИКА КАПИЦЫ ===")

# ------------------ ПАРАМЕТРЫ ------------------
g = 9.81           # ускорение свободного падения
l = 1.0            # длина стержня
a = 0.1            # амплитуда вибраций точки подвеса
nu = 50.0          # частота вертикальных колебаний (ВЫСОКАЯ!)
w0 = np.sqrt(g/l)  # собственная частота математического маятника

log.info(f"Собственная частота ω0 = {w0:.3f}")
log.info(f"Заданная частота вынужденных колебаний ν = {nu:.1f}  (ν >> ω0)")

# ------------------ УРАВНЕНИЕ МАЯТНИКА КАПИЦЫ ------------------
def kapitza_equations(t, y):
    phi, dphi = y
    ddphi = -(a * nu**2 * np.cos(nu * t) + g) * np.sin(phi) / l
    return [dphi, ddphi]

# ------------------ ЧИСЛЕННОЕ РЕШЕНИЕ ------------------
t_max = 10
t_eval = np.linspace(0, t_max, 5000)
y0 = [0.2, 0.0]   # начальный угол и скорость

sol = solve_ivp(kapitza_equations, [0, t_max], y0, t_eval=t_eval)

phi = sol.y[0]
dphi = sol.y[1]
t = sol.t

# ------------------ ТРАЕКТОРИЯ (x,y) ------------------
x = l * np.sin(phi)
y = -l * np.cos(phi) + a * np.cos(nu * t)

# ------------------ ЭНЕРГИЯ ------------------
E_pot = -g * y
E_kin = 0.5 * (l**2 * dphi**2)
E = E_pot + E_kin

# ------------------ ГРАФИКИ ------------------
plt.figure(figsize=(16, 10))

# φ(t)
plt.subplot(2, 2, 1)
plt.plot(t, phi)
plt.title("Угол φ(t)")
plt.xlabel("t")
plt.ylabel("φ")

# Траектория (x,y)
plt.subplot(2, 2, 2)
plt.plot(x, y)
plt.title("Траектория маятника (x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")

# Энергия
plt.subplot(2, 2, 3)
plt.plot(t, E_pot, label="Потенциальная")
plt.plot(t, E_kin, label="Кинетическая")
plt.plot(t, E, label="Полная")
plt.legend()
plt.title("Энергия маятника Капицы")
plt.xlabel("t")
plt.ylabel("E")

# φ(t) — увеличение на коротком интервале
plt.subplot(2, 2, 4)
plt.plot(t[:1500], phi[:1500])
plt.title("φ(t) (увеличенный участок)")
plt.xlabel("t")
plt.ylabel("φ")

plt.tight_layout()
plt.show()

log.info("=== РАСЧЁТ ЗАВЕРШЕН ===")
