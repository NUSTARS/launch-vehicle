import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import mplcursors

# PYTHON 3.7

# USER INPUTS
weight = 31.17 # lbs
apogee = 10000  # ft
drogue_d = 36  # in
main_cd = 2.2
drogue_cd = 1.6
main_alt = 1000  # ft
main_d = 96  # in
airframe_d = 5  # in

# CONSTANTS
g = 32.17  # ft/s^2
n = 4  # fill constant from knacke for solid textile parachutes

# COMPUTED VALUES
drogue_A = np.pi * (drogue_d**2) / 4 / 144  # ft^2 (convert from in^2)
main_A = np.pi * (main_d**2) / 4 / 144  # ft^2 (convert from in^2)
mass = weight / g  # slugs
eta = (np.pi*(((airframe_d/12)**2)/4)) / main_A  # ratio of projected area at line stretch to main area

# STRUCTURE FOR SOLVER PARAMETERS
params = {
    "apogee": apogee,
    "drogue_A": drogue_A,
    "main_alt": main_alt,
    "main_A": main_A,
    "drogue_cd": drogue_cd,
    "main_cd": main_cd,
    "mass": mass,
    "t_fill": None
}


def drogue_dynamics(t, state, params):
    x, y, vx, vy = state  # Unpack state
    rho = density(y)  # Get air density
    # Drogue parachute only
    f_drag = 0.5 * rho * vy**2 * params["drogue_cd"] * params["drogue_A"]
    f_y = f_drag - params["mass"] * g
    f_x = 0

    ax = f_x / params["mass"]
    ay = f_y / params["mass"]

    return [vx, vy, ax, ay]


def inflating_dynamics(t, state, params):
    x, y, vx, vy = state  # Unpack state
    rho = density(y)  # Get air density

    # calculate parachute area for this timestep -- from Knacke, for solid textile parachutes
    drag_area_frac = ((1-eta)*(t / params["t_fill"])**3 + eta)**2
    computed_area = drag_area_frac * params['main_A']

    # calculate drag    
    f_drag = ((0.5 * rho * vy**2 * params["main_cd"] * computed_area) + 
              (0.5 * rho * vy**2 * params["drogue_cd"] * params["drogue_A"]))

    f_y = f_drag - params["mass"] * g
    f_x = 0

    ax = f_x / params["mass"]
    ay = f_y / params["mass"]

    return [vx, vy, ax, ay]


def main_dynamics(t, state, params):
    x, y, vx, vy = state  # Unpack state
    rho = density(y)  # Get air density
    
    # calculate force for this timestep
    f_drag = ((0.5 * rho * vy**2 * params["main_cd"] * params["main_A"]) + 
              (0.5 * rho * vy**2 * params["drogue_cd"] * params["drogue_A"]))
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


def main_dep_event(t, y):
    """Stop integrating with drogue dynamics"""
    return y[1] - params["main_alt"]
main_dep_event.terminal = True
main_dep_event.direction = -1

def full_inflation_event(t, y):
    """Stop integrating with inflating parachute dynamics"""
    return params["t_fill"] - t
full_inflation_event.terminal = True
full_inflation_event.direction = -1

def landing_event(t, y):
    """Stop integration when altitude reaches 0 (ground level)."""
    return y[1]  # Track altitude (y position)
landing_event.terminal = True
landing_event.direction = -1


def plot_results(time, states, drag_area, drag_force, params):
    """Plot height, velocity, dragarea, and drag force against time."""
    fig1, motion_axs = plt.subplots(2, 1, figsize=(10, 8))

    # Height vs. Time
    altitude, = motion_axs[0].plot(time, states[1, :], "-", label="Height [ft]")
    motion_axs[0].set_ylim(0, params["apogee"]+100)
    motion_axs[0].set_xlabel("Time [s]")
    motion_axs[0].set_ylabel("Height [ft]")
    motion_axs[0].axhline(params["apogee"], linestyle='--', color='b', label='Apogee')
    motion_axs[0].axhline(params["main_alt"], linestyle='--', color='g', label='Main Deployment')
    motion_axs[0].set_title("Height vs. Time")
    motion_axs[0].grid(True)  # Add grid lines

    # Velocity vs. Time
    velocity, = motion_axs[1].plot(time, states[3, :], "--r", label="Velocity [ft/s]")
    motion_axs[1].set_xlabel("Time [s]")
    motion_axs[1].set_ylabel("Velocity [ft/s]")
    motion_axs[1].set_title("Velocity vs. Time")
    motion_axs[1].grid(True)  # Add grid lines

    # Find velocity just before the main parachute opens (at `main_alt`)
    idx_main_open = np.argmax(states[1, :] <= params["main_alt"])  # First index where altitude is below `main_alt`
    if idx_main_open > 0:
        terminal_velocity_drogue = states[3, idx_main_open - 1]  # Velocity just before main opens
    else:
        terminal_velocity_drogue = np.nan  # If we don't find such a point

    descent_time = time[-1]
    impact_velocity = states[3, -1]

    #opening_force = 0.5 * density(0) * terminal_velocity_drogue**2 * params["main_cd"] * params["main_A"]   # this is the old way
    #opening_acceleration =(opening_force-params["mass"]*g)/params["mass"]

    result_text = (
        f"Terminal Velocity: {terminal_velocity_drogue:.2f} ft/s\n"
        f"Total Descent Time: {descent_time:.2f} s\n"
        f"Main Impact Velocity: {impact_velocity:.2f} ft/s\n"
        #f"Main Opening Load: {opening_force:.2f} lbf\n"
        #f"Main Opening Acceleration: {opening_acceleration:.2f} ft/s^2"
    )
    motion_axs[0].text(0.95, 0.9, result_text, fontsize=12, va="top", ha="right",
            transform=motion_axs[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))

    # Drag plots
    fig2, drag_axs = plt.subplots(figsize=(10, 8))

    # Drag area vs. time
    dragarea_line, = drag_axs.plot(time, drag_area, label="Drag Area")
    drag_axs.set_xlabel("Time [s]")
    drag_axs.set_ylabel("Drag Area [ft^2], Drag Force [lbf]")
    drag_axs.set_title("Drag Area and Force vs. Time")
    drag_axs.grid(True)

    # find where main is fully open and plot a line there
    idx_inflated = np.argmax(drag_area == params["main_A"]*params["main_cd"])
    drag_axs.axvline(time[idx_inflated], linestyle="--", linewidth=0.75, label="Main Parachute Fully Open")

    # Drag force vs. time
    dragforce_line, = drag_axs.plot(time, drag_force, label="Drag Force")
    drag_axs.legend()
    
    # make hovering over lines display values
    cursor = mplcursors.cursor([altitude, velocity, dragarea_line, dragforce_line], hover=True)

    plt.tight_layout()

# Initial conditions [x, y, vx, vy]
initial_conditions = [0, apogee, 0, 0]
t_span = [0, 1e3]

# Solve ODE using `solve_ivp`
drogue_sol = solve_ivp(
    lambda t, y: drogue_dynamics(t, y, params),
    t_span,
    initial_conditions,
    method='RK45',
    events=main_dep_event,
    rtol=1e-6
)

drogue_v = drogue_sol.y[3, -1]
params['t_fill'] = np.round((n*main_d/12)/(abs(drogue_v)**0.85), decimals=2)

inflating_sol = solve_ivp(
    lambda t, y: inflating_dynamics(t, y, params),
    t_span,
    drogue_sol.y[:,-1],
    method='RK45',
    events=full_inflation_event,
    rtol=1e-6,
    max_step=0.01
)

main_sol = solve_ivp(
    lambda t, y: main_dynamics(t, y, params),
    t_span,
    inflating_sol.y[:,-1],
    method='RK45',
    events=landing_event,
    rtol=1e-6
)

combined_time = np.concatenate((drogue_sol.t, 
                                (inflating_sol.t + drogue_sol.t[-1]), 
                                (main_sol.t + drogue_sol.t[-1] + inflating_sol.t[-1])))

combined_traj = np.concatenate((drogue_sol.y, 
                                inflating_sol.y, 
                                main_sol.y), 
                                axis=1)


# get drag area and drag force v time for whole flight
density_vec = np.vectorize(density)   # needed for vectorized operation

#drogue_dragarea = np.repeat(params["drogue_A"]*params["drogue_cd"], len(drogue_sol.t))
drogue_dragarea = np.zeros(len(drogue_sol.t))   # we are just focusing on drag from the main parachute here
drogue_dragforce = 0.5 * density_vec(drogue_sol.y[1]) * (drogue_sol.y[3])**2 * drogue_dragarea

# I couldn't save data on parachute inflation/drag force bc of how this solver works
# just calculate area (we have t_fill and velocities) and drag force using vectorized operations here, then plot
inflation_curve = ((1-eta)*(inflating_sol.t / params["t_fill"])**3 + eta)**2
inflating_dragarea = inflation_curve * params["main_A"] * params["main_cd"] #+ drogue_dragarea[-1]
inflating_dragforce = 0.5 * density(params["main_alt"]) * (inflating_sol.y[3])**2 * inflating_dragarea

main_dragarea = np.repeat(params["main_A"]*params["main_cd"], len(main_sol.t))
main_dragforce = 0.5 * density_vec(main_sol.y[1]) * (main_sol.y[3])**2 * main_dragarea

# combine
combined_dragarea = np.concatenate((drogue_dragarea,
                                   inflating_dragarea,
                                   main_dragarea))

combined_dragforce = np.concatenate((drogue_dragforce,
                                  inflating_dragforce,
                                  main_dragforce))


# Plot the results
plot_results(combined_time, combined_traj, combined_dragarea, combined_dragforce, params)

plt.show()