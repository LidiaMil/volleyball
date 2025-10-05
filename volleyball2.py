import numpy as np
import matplotlib.pyplot as plt

# Court and net parameters
court_length = 18.0
court_width = 9.0
net_x = court_length / 2
net_height = 2.43

# Physics
g = 9.81

def flight_time(vz, z0):
    a = -0.5 * g
    b = vz
    c = z0
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)
    ts = [t for t in (t1, t2) if t is not None and t > 0]
    if not ts:
        return None
    return max(ts)

def simulate(x0, y0, v0, alpha, phi, z0):
    vx = v0 * np.cos(alpha) * np.cos(phi)
    vy = v0 * np.cos(alpha) * np.sin(phi)
    vz = v0 * np.sin(alpha)
    if vx <= 0:
        return None, None, False, None
    t_land = flight_time(vz, z0)
    if t_land is None:
        return None, None, False, None
    t_net = (net_x - x0) / vx
    net_cross_in_time = (t_net is not None) and (0 < t_net < t_land)
    if not net_cross_in_time:
        x_land = x0 + vx * t_land
        y_land = y0 + vy * t_land
        return x_land, y_land, False, t_net
    z_net = z0 + vz * t_net - 0.5 * g * t_net**2
    net_clear = (z_net > net_height)
    x_land = x0 + vx * t_land
    y_land = y0 + vy * t_land
    return x_land, y_land, net_clear, t_net

def success(x0, y0, v0, alpha, phi, z0):
    x_land, y_land, net_clear, _ = simulate(x0, y0, v0, alpha, phi, z0)
    if x_land is None:
        return False
    lands_in_opponent = (net_x <= x_land <= court_length) and (0 <= y_land <= court_width)
    return bool(net_clear and lands_in_opponent)

def success_exists_over_phi(x0, y0, v0, alpha, z0, phi_min=-np.deg2rad(45), phi_max=np.deg2rad(45), nphi=241):
    phis = np.linspace(phi_min, phi_max, nphi)
    best_phi = np.nan
    best_dist = None
    any_success = False
    for phi in phis:
        x_land, y_land, net_clear, _ = simulate(x0, y0, v0, alpha, phi, z0)
        if x_land is None:
            continue
        lands_in_opponent = (net_x <= x_land <= court_length) and (0 <= y_land <= court_width)
        if net_clear and lands_in_opponent:
            any_success = True
            dist = abs(y_land - court_width/2.0)
            if (best_dist is None) or (dist < best_dist):
                best_dist = dist
                best_phi = phi
    return any_success, best_phi

if __name__ == "__main__":
    # Example parameters
    v0 = 10.0
    alpha = np.deg2rad(30.0)
    z0 = 2.3
    nx, ny = 50, 30
    x_positions = np.linspace(0.5, net_x - 0.5, nx)
    y_positions = np.linspace(0.25, court_width - 0.25, ny)
    XX, YY = np.meshgrid(x_positions, y_positions)
    success_grid = np.zeros_like(XX, dtype=float)
    best_phi_grid = np.full_like(XX, fill_value=np.nan, dtype=float)
    for i in range(ny):
        for j in range(nx):
            ok, phi_star = success_exists_over_phi(XX[i,j], YY[i,j], v0, alpha, z0)
            success_grid[i,j] = 1.0 if ok else 0.0
            best_phi_grid[i,j] = phi_star
    # Plot success region
    plt.figure(figsize=(8,5))
    plt.pcolormesh(XX, YY, success_grid, shading='auto', vmin=0.0, vmax=1.0)
    plt.axvline(net_x, linewidth=2)
    plt.xlim(0, net_x)
    plt.ylim(0, court_width)
    plt.xlabel("X0 (м)")
    plt.ylabel("Y0 (м)")
    plt.title("Существует ли φ для успешной подачи")
    plt.colorbar(label="1 — есть φ, 0 — нет φ")
    plt.show()
    # Plot best phi map (deg), masked where no success
    phi_deg = np.rad2deg(best_phi_grid)
    phi_deg_masked = np.where(success_grid > 0.5, phi_deg, np.nan)
    plt.figure(figsize=(8,5))
    mesh = plt.pcolormesh(XX, YY, phi_deg_masked, shading='auto')
    plt.axvline(net_x, linewidth=2)
    plt.xlim(0, net_x)
    plt.ylim(0, court_width)
    plt.xlabel("X0 (м)")
    plt.ylabel("Y0 (м)")
    plt.title("Оптимальный азимут φ (°), где успех возможен")
    plt.colorbar(mesh, label="φ (°)")
    plt.show()
