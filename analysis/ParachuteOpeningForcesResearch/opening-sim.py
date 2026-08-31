import numpy as np
import matplotlib.pyplot as plt


def integrate(f, xt, dt, current_force):
    # """
    # This function takes in an initial condition x(t) and a timestep dt,
    # as well as a dynamical system f(x) that outputs a vector of the
    # same dimension as x(t). It outputs a vector x(t+dt) at the future
    # time step.

    # Parameters
    # ============
    # dyn: Python function
    #     derivate of the system at a given step x(t),
    #     it can considered as \dot{x}(t) = func(x(t))
    # xt: NumPy array
    #     current step x(t)
    # dt:
    #     step size for integration

    # Return
    # ============
    # new_xt:
    #     value of x(t+dt) integrated from x(t)
    # """
    k1 = dt * f(xt, current_force)
    k2 = dt * f(xt+k1/2., current_force)
    k3 = dt * f(xt+k2/2., current_force)
    k4 = dt * f(xt+k3, current_force)
    new_xt = xt + (1/6.) * (k1+2.0*k2+2.0*k3+k4)
    return new_xt



def simulate(f, x0, drag_area, drag_area_frac_vs_time, rho, g, burnout_mass, tspan, dt, integrate):
    # """
    # This function takes in an initial condition x0, a timestep dt,
    # a time span tspan consisting of a list [min_time, max_time],
    # as well as a dynamical system f(x) that outputs a vector of the
    # same dimension as x0. It outputs a full trajectory simulated
    # over the time span of dimensions (xvec_size, time_vec_size).

    # Parameters
    # ============
    # f: Python function
    #     derivate of the system at a given step x(t),
    #     it can considered as \dot{x}(t) = func(x(t))
    # x0: NumPy array
    #     initial conditions
    # tspan: Python list
    #     tspan = [min_time, max_time], it defines the start and end
    #     time of simulation
    # dt:
    #     time step for numerical integration
    # integrate: Python function
    #     numerical integration method used in this simulation

    # Return
    # ============
    # x_traj:
    #     simulated trajectory of x(t) from t=0 to tf
    # """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0),N))
    force_traj = np.zeros(N)
    current_drag_area = 0

    for i in range(N):
        # get current drag area
        if i < len(drag_area_frac_vs_time):
            current_drag_area = drag_area_frac_vs_time[i] * drag_area
        else:
            current_drag_area = drag_area_frac_vs_time[-1] * drag_area
        
        # calculate instantaneous drag force
        current_force = 0.5*rho*(x[1])**2*current_drag_area - burnout_mass*g
        force_traj[i] = current_force + burnout_mass*g

        # integrate for new state
        xtraj[:,i]=integrate(f,x,dt, current_force)
        x = np.copy(xtraj[:,i])
    return xtraj, tvec, force_traj

###############################################################################################################

# constants
rho = 0.002378 # slg/ft^3
g = 32.17 # ft/s^2
cd = 2.2
main_diam = 4 # ft
main_s = np.pi*(main_diam**2)/4 # ft^2
drag_area = cd*main_s # ft^2
v_drogue = -74 # ft/sec
burnout_mass = 0.3511255846 # slg
main_alt = 550 # feet
eta = (np.pi*((4/12)**2)/4) / main_s # fraction of area at line stretch to full area
dt = 0.001
n = 4 # fill constant

# find fill time
t_fill = np.round((n*main_diam)/(abs(v_drogue)**0.85), decimals=2)

# get drag area vs. time
fill_times = np.linspace(0, t_fill, int(t_fill/dt))
drag_area_frac = ((1-eta)*(fill_times/t_fill)**3 + eta)**2
#plt.plot(fill_times, drag_area_frac*drag_area)


# start from line stretch
current_force = 0 # initialize
x0 = [main_alt, v_drogue]

def dyn(s, current_force):
    acc = current_force / burnout_mass
    xdot = np.array([s[1], acc])
    return xdot

traj, times, force_time = simulate(f=dyn, 
                                   x0=x0, 
                                   drag_area=drag_area, 
                                   drag_area_frac_vs_time=drag_area_frac, 
                                   rho=rho,
                                   g=g,
                                   burnout_mass=burnout_mass,
                                   tspan=[0,5], 
                                   dt=dt, 
                                   integrate=integrate)



plt.plot(times, force_time, label='force')
#plt.plot(times, traj[0,:], label='altitude')
plt.plot(times, traj[1,:], label='velocity')
plt.legend()
plt.show()


