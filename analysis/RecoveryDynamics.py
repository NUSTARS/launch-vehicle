import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# PYTHON 3.7

# USER INPUTS
weight = 35  # lbs
apogee = 5000  # ft
drogue_d = 18  # in
main_cd = 2.2
drogue_cd = 1.6
main_alt = 500  # ft
main_d = 120  # in

# CONSTANTS
g = 32.17  # ft/s^2

# COMPUTED VALUES
drogue_A = np.pi * (drogue_d**2) / 4 / 144  # ft^2 (convert from in^2)
main_A = np.pi * (main_d**2) / 4 / 144  # ft^2 (convert from in^2)
mass = weight / g  # slugs

# STRUCTURE FOR SOLVER PARAMETERS
params = {
    "apogee": apogee,
    "drogue_A": drogue_A,
    "main_alt": main_alt,
    "main_A": main_A,
    "drogue_cd": drogue_cd,
    "main_cd": main_cd,
    "mass": mass
}

def recovery_dynamics(t, state, params):
    x, y, vx, vy = state  # Unpack state
    rho = density(y)  # Get air density

    if y > params["main_alt"]:
        # Drogue parachute only
        f_drag = 0.5 * rho * vy**2 * params["drogue_cd"] * params["drogue_A"]
    elif y > 0:
        # Main parachute, with dynamic opening
        height_to_open = 300
        if params["main_alt"] - y < height_to_open:
            computed_area = params["drogue_A"] + (params["main_A"] - params["drogue_A"]) * (params["main_alt"] - y) / height_to_open
        else:
            computed_area = params["main_A"]
        f_drag = 0.5 * rho * vy**2 * params["main_cd"] * computed_area
    else:
        return [0, 0, 0, 0]  # Stop descent at ground level

    f_y = f_drag - params["mass"] * g
    f_x = 0

    ax = f_x / params["mass"]
    ay = f_y / params["mass"]

    return [vx, vy, ax, ay]

def density(h):
    """Air density based on altitude (ft)."""
    if h < 36152:
        T = 59 - 0.00356 * h  # Temperature in Fahrenheit
        p = 2116 * ((T + 459.7) / 518.6)**5.256  # Pressure in lbf/ft^2
        rho = p / (1718 * (T + 459.7))  # Density in slugs/ft^3
    elif h < 82345:
        T = -70
        p = 473.1 * np.exp(1.73 - 0.000048 * h)
        rho = p / (1718 * (T + 459.7))  # Density in slugs/ft^3
    else:
        rho = -1  # Out of atmospheric range
    return rho

def event_function(t, y):
    """Stop integration when altitude reaches 0 (ground level)."""
    return y[1]  # Track altitude (y position)

event_function.terminal = True
event_function.direction = -1

def plot_results(time, states, params):
    """Plot height and velocity against time."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Height vs. Time
    axs[0].plot(time, states[:, 1], "-")
    axs[0].set_ylim(0, params["apogee"])
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Height [ft]")
    axs[0].axhline(params["apogee"], linestyle='--', color='b', label='Apogee')
    axs[0].axhline(params["main_alt"], linestyle='--', color='g', label='Main Deployment')
    axs[0].set_title("Height vs. Time")
    axs[0].grid(True)  # Add grid lines

    # Velocity vs. Time
    axs[1].plot(time, states[:, 3], "--r")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Velocity [ft/s]")
    axs[1].set_title("Velocity vs. Time")
    axs[1].grid(True)  # Add grid lines

    terminal_velocity = min(sol.y[3])
    descent_time = sol.t[-1]
    impact_velocity = sol.y[3, -1]

    result_text = (
        f"Terminal Velocity: {terminal_velocity:.2f} ft/s\n"
        f"Total Descent Time: {descent_time:.2f} s\n"
        f"Main Impact Velocity: {impact_velocity:.2f} ft/s"
    )
    axs[0].text(0.95, 0.9, result_text, fontsize=12, va="top", ha="right",
            transform=axs[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))

    plt.tight_layout()

# Initial conditions [x, y, vx, vy]
initial_conditions = [0, apogee, 0, 0]
t_span = [0, 1e3]

# Solve ODE using `solve_ivp`
sol = solve_ivp(
    lambda t, y: recovery_dynamics(t, y, params),
    t_span,
    initial_conditions,
    method='RK45',
    events=event_function,
    rtol=1e-6
)

# Plot the results
plot_results(sol.t, sol.y.T, params)
plt.show()