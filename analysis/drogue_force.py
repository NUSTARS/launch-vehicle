import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

shock_cord_length = 12.7 # Shock Cord Length (m)
exit_velocity = 8.32 # Exit Velocity (m/s)
#exit_velocity = 10 # Exit Velocity (m/s)
t_vec = np.arange(0, 2, 0.001) # (s)
m1 = 4.7 # (kg) forward section
m2 = 10.9 # (kg) aft section
p_force = 934 # (N) force from blackpowder
coupler_length = 0.127 # (m)

def initial_dynamics(t, x):
    return np.array([x[2], x[3], -p_force/m1, p_force/m2])

def separation_event(t, y): # stops integration when it returns 0
    return coupler_length - (y[1] - y[0])

separation_event.terminal = True
separation_event.direction = -1

ic_sol = spi.solve_ivp(initial_dynamics, (0, t_vec[-1]), np.array([0, 0, 0, 0]), events = separation_event, method='RK45', t_eval=t_vec, max_step=0.01)
forward_vel = (ic_sol.y.T[:, 2])[-1]
aft_vel = (ic_sol.y.T[:, 3])[-1]

fig1, ax1 = plt.subplots()
ax1.plot(ic_sol.t, ic_sol.y.T[:, 0], label = 'Forward Section Position')
ax1.plot(ic_sol.t, ic_sol.y.T[:, 1], label = 'Aft Section Position')
ax1.set_ylabel('Position (m)')
ax1.set_title('Separation vs Time')
ax1.set_xlabel('Time (s)')
ax1.legend()

x_0 = np.array([0, 0, forward_vel, aft_vel])  # [pos_x, pos_y, vel_x, vel_y]

def get_spring_force(distance, rest_length):

    extension = abs(distance) - rest_length

    if extension < 0:
        return 0

    return (5.588 * np.exp(11.457 * abs(extension)) - 5.588) * np.sign(distance)

def get_xdot(t, x):
    spring_force = get_spring_force(x[1] - x[0], shock_cord_length)
    return np.array([x[2], x[3], spring_force/m1, -spring_force/m2])

sol = spi.solve_ivp(get_xdot, (0, t_vec[-1]), x_0, method='RK45', t_eval=t_vec, max_step=0.01)

output_force = [get_spring_force(pos[1] - pos[0], shock_cord_length) for pos in sol.y.T]
pos_diff = sol.y.T[:, 1] - sol.y.T[:, 0] - shock_cord_length
E_masses = 0.5 * m1 * sol.y.T[:, 2]**2 + 0.5 * m2 * sol.y.T[:, 3]**2

fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

axs[0].plot(sol.t, output_force)
axs[0].set_ylabel('Drogue Force (N)')
axs[0].set_title('Drogue Force vs Time')
axs[0].set_xlabel('Time (s)')

axs[1].plot(sol.t, sol.y.T[:, 2:])
axs[1].plot(sol.t, pos_diff)
axs[1].set_ylabel('States')
axs[1].set_title('State Variables vs Time')
axs[1].legend(['Vel 1 (m/s)', 'Vel 2 (m/s)', 'Pos Diff (m)'])

axs[2].plot(sol.t, E_masses)
axs[2].set_ylabel('Kinetic Energy (J)')
axs[2].set_title('Total Kinetic Energy vs Time')
plt.show()