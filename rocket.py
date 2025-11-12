import numpy as np
import matplotlib.pyplot as plt

# ==============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================

def delta_v(Ve, m0, mf):
    """Приращение скорости по уравнению Циолковского."""
    if mf <= 0 or m0 <= mf:
        return 0.0
    return Ve * np.log(m0 / mf)


# ==============================================
# 1. ДВУХСТУПЕНЧАТАЯ РАКЕТА — анализ распределения топлива
# ==============================================

def analyze_two_stage(
    Ve1=3000, Ve2=4500,
    mc1=1500, mc2=1000, m_payload=2000,
    total_fuel=4000
):
    ratios = np.linspace(0.05, 0.95, 19)
    fixed_total_fuel = [3000, 4000, 5000, 6000]  # разные общие массы топлива
    plt.figure(figsize=(9, 5))

    for tf in fixed_total_fuel:
        velocities = []
        for r in ratios:
            mt1 = tf * r
            mt2 = tf * (1 - r)

            # 1-я ступень
            m0_1 = mc1 + mc2 + mt1 + mt2 + m_payload
            mf_1_before_jettison = m0_1 - mt1
            dV1 = delta_v(Ve1, m0_1, mf_1_before_jettison)

            # Отделение 1-й ступени
            m0_2 = mf_1_before_jettison - mc1
            mf_2_before_jettison = m0_2 - mt2
            dV2 = delta_v(Ve2, m0_2, mf_2_before_jettison)

            velocities.append((dV1 + dV2) / 1000)
        plt.plot(ratios, velocities, lw=2, label=f"Топливо всего: {tf/1000:.1f} т")

    plt.title("Двухступенчатая ракета: зависимость итоговой скорости от распределения топлива")
    plt.xlabel("Доля топлива в первой ступени (от общей массы)")
    plt.ylabel("Итоговая скорость, км/с")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================
# 2. ТРЁХСТУПЕНЧАТАЯ РАКЕТА — изменение скорости во времени
# ==============================================

def simulate_three_stage_multi(
    Ve1=3000, Ve2=4000, Ve3=5000,
    mc1=2500, mc2=1200, mc3=600, m_payload=1000,
    mdot1=120, mdot2=80, mdot3=50
):
    """
    Модель скорости трёхступенчатой ракеты во времени для разных масс топлива.
    """
    plt.figure(figsize=(9, 5))
    fuel_sets = [
        (4000, 1500, 500),
        (6000, 2500, 800),
        (8000, 3000, 1000),
        (10000, 3500, 1200)
    ]
    colors = ['purple', 'blue', 'green', 'orange']

    for (m1_fuel, m2_fuel, m3_fuel), color in zip(fuel_sets, colors):
        m0 = mc1 + mc2 + mc3 + m1_fuel + m2_fuel + m3_fuel + m_payload
        t, dt, V, m = 0.0, 0.25, 0.0, m0
        t_list, V_list = [], []

        # --- Первая ступень ---
        burn1 = m1_fuel / mdot1
        for _ in np.arange(0, burn1, dt):
            m -= mdot1 * dt
            V = Ve1 * np.log(m0 / m)
            t += dt
            t_list.append(t)
            V_list.append(V)
        m -= mc1
        m0, V1_end = m, V

        # --- Вторая ступень ---
        burn2 = m2_fuel / mdot2
        for _ in np.arange(0, burn2, dt):
            m -= mdot2 * dt
            V = V1_end + Ve2 * np.log(m0 / m)
            t += dt
            t_list.append(t)
            V_list.append(V)
        m -= mc2
        m0, V2_end = m, V

        # --- Третья ступень ---
        burn3 = m3_fuel / mdot3
        for _ in np.arange(0, burn3, dt):
            m -= mdot3 * dt
            V = V2_end + Ve3 * np.log(m0 / m)
            t += dt
            t_list.append(t)
            V_list.append(V)

        plt.plot(t_list, np.array(V_list)/1000, lw=2, color=color,
                 label=f"{m1_fuel/1000:.1f}т + {m2_fuel/1000:.1f}т + {m3_fuel/1000:.1f}т")

    plt.axhline(7.8, color='gray', linestyle='--', label='Орбитальная скорость LEO ≈ 7.8 км/с')
    plt.title("V(t) трёхступенчатой ракеты при разных масcах топлива")
    plt.xlabel("Время, с")
    plt.ylabel("Скорость, км/с")
    plt.legend(title="Масса топлива ступеней (1+2+3)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==============================================
# 3. АНАЛИЗ ВЛИЯНИЯ СКОРОСТИ ИСТЕКАЮЩИХ ГАЗОВ (V_e)
# ==============================================

def analyze_exhaust_velocity():
    """
    Исследуем, как изменение Ve влияет на итоговую скорость двухступенчатой ракеты.
    """
    plt.figure(figsize=(9, 5))
    Ve_values = [2500, 3000, 4000, 5000]
    colors = ['teal', 'royalblue', 'orange', 'red']

    ratios = np.linspace(0.1, 0.9, 15)
    mc1, mc2, m_payload, total_fuel = 1500, 1000, 2000, 4000

    for Ve, color in zip(Ve_values, colors):
        velocities = []
        for r in ratios:
            mt1 = total_fuel * r
            mt2 = total_fuel * (1 - r)

            m0_1 = mc1 + mc2 + mt1 + mt2 + m_payload
            mf_1_before_jettison = m0_1 - mt1
            dV1 = delta_v(Ve, m0_1, mf_1_before_jettison)
            m0_2 = mf_1_before_jettison - mc1
            mf_2_before_jettison = m0_2 - mt2
            dV2 = delta_v(Ve, m0_2, mf_2_before_jettison)
            velocities.append((dV1 + dV2) / 1000)

        plt.plot(ratios, velocities, lw=2, color=color, label=f"Vₑ = {Ve/1000:.1f} км/с")

    plt.title("Влияние скорости истечения газов на итоговую ΔV (двухступенчатая ракета)")
    plt.xlabel("Доля топлива в первой ступени")
    plt.ylabel("Итоговая скорость, км/с")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==============================================
# 4. СРАВНЕНИЕ 2 И 3 СТУПЕНЕЙ ПРИ РАЗНОМ ТОПЛИВЕ
# ==============================================

def compare_two_and_three_stage():
    """
    Сравнение эффективности 2- и 3-ступенчатой ракеты при одинаковом общем запасе топлива.
    """
    Ve = 3500
    total_fuel = 6000
    m_payload = 2000
    mc1, mc2, mc3 = 1500, 1000, 500

    ratios = np.linspace(0.1, 0.9, 17)
    two_stage, three_stage = [], []

    for r in ratios:
        # Двухступенчатая
        mt1 = total_fuel * r
        mt2 = total_fuel * (1 - r)
        m0_1 = mc1 + mc2 + mt1 + mt2 + m_payload
        mf_1_before_jettison = m0_1 - mt1
        dV1 = delta_v(Ve, m0_1, mf_1_before_jettison)
        m0_2 = mf_1_before_jettison - mc1
        mf_2_before_jettison = m0_2 - mt2
        dV2 = delta_v(Ve, m0_2, mf_2_before_jettison)
        two_stage.append((dV1 + dV2) / 1000)

        # Трёхступенчатая (равное деление топлива по 3 ступеням)
        m1_fuel = total_fuel * 0.5 * r
        m2_fuel = total_fuel * 0.3
        m3_fuel = total_fuel * (0.2 - 0.1 * r)
        m0_1 = mc1 + mc2 + mc3 + m1_fuel + m2_fuel + m3_fuel + m_payload
        mf_1_before_jettison = m0_1 - m1_fuel
        dV1 = delta_v(Ve, m0_1, mf_1_before_jettison)
        m0_2 = mf_1_before_jettison - mc1
        mf_2_before_jettison = m0_2 - m2_fuel
        dV2 = delta_v(Ve, m0_2, mf_2_before_jettison)
        m0_3 = mf_2_before_jettison - mc2
        mf_3_before_jettison = m0_3 - m3_fuel
        dV3 = delta_v(Ve, m0_3, mf_3_before_jettison)
        three_stage.append((dV1 + dV2 + dV3) / 1000)

    plt.figure(figsize=(9, 5))
    plt.plot(ratios, two_stage, 'r-', lw=2, label="2-ступенчатая")
    plt.plot(ratios, three_stage, 'b--', lw=2, label="3-ступенчатая")
    plt.title("Сравнение эффективности 2- и 3-ступенчатой ракеты")
    plt.xlabel("Доля топлива в первой ступени")
    plt.ylabel("Итоговая скорость, км/с")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":
    print("=== Анализ 2-ступенчатой ракеты (разные запасы топлива) ===")
    analyze_two_stage()

    print("\n=== 3-ступенчатая ракета: изменение скорости во времени ===")
    simulate_three_stage_multi()

    print("\n=== Влияние скорости истечения газов ===")
    analyze_exhaust_velocity()

    print("\n=== Сравнение 2 и 3 ступеней ===")
    compare_two_and_three_stage()
