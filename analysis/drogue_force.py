import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

shock_cord_length = 12.7 # Shock Cord Length (m)
exit_velocity = 8.32 # Exit Velocity (m/s)
#exit_velocity = 10 # Exit Velocity (m/s)
t_vec = np.arange(0, 2, 0.001) # (s)
m1 = 4.7 # (kg)
m2 = 10.9 # (kg)

x_0 = np.array([0, 0, -exit_velocity, 0])  # [pos_x, pos_y, vel_x, vel_y]

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







