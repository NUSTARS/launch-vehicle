# quick plotting script
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

def plot_altus(filename, metric=False):
    flight_data = pd.read_csv(filename)

    if not metric:
        flight_data['height'] = flight_data['height'] * 3.28084
        flight_data['speed'] = flight_data['speed'] * 3.28084
        flight_data['acceleration'] = flight_data['acceleration'] * 3.28084

    fig, motion_graph = plt.subplots(3,1, sharex=True)

    # Plot altitude and speed and acc with markers
    alt_line, = motion_graph[0].plot(flight_data['time'], flight_data['height'], label='Altitude')
    spd_line, = motion_graph[1].plot(flight_data['time'], flight_data['speed'], label='Velocity')
    acc_line, = motion_graph[2].plot(flight_data['time'], flight_data['acceleration'], label='Acceleration')

    motion_graph[0].set_ylabel('Altitude [ft]' if not metric else 'Altitude (m)')
    motion_graph[1].set_ylabel('Vertical Speed [ft/s]' if not metric else "Velocity (m/s)")
    motion_graph[2].set_ylabel('Vertical Acceleration [ft/s^2]' if not metric else 'Acceleration (m/s^2)')

    motion_graph[2].set_xlabel("Time (s)")
    motion_graph[0].set_title("Altus Altimeter Flight Data")
    motion_graph[0].grid(True)
    motion_graph[1].grid(True)
    motion_graph[2].grid(True)

    # Enable hover on both data series
    cursor = mplcursors.cursor([alt_line, spd_line, acc_line], hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.2f}s\nval = {sel.target[1]:.2f}"))

    #plt.show()

    return flight_data


############################################################################################################################################


def plot_blueraven(LR_filename, HR_filename):
    LR_data = pd.read_csv(LR_filename)
    HR_data = pd.read_csv(HR_filename)

    fig1, LR_graph = plt.subplots()

    # Plot altitude and speed with markers
    blr_alt_line = LR_graph.scatter(LR_data['Flight_Time_(s)'], LR_data['Baro_Altitude_AGL_(feet)'], label='Baro Altitude', marker='.')
    inert_spd_line = LR_graph.scatter(LR_data['Flight_Time_(s)'], LR_data['Velocity_Up'], label='Velocity', marker='.')
    inert_alt_line = LR_graph.scatter(LR_data['Flight_Time_(s)'], LR_data['Inertial_Altitude'], label='Inertial Altitude', marker='.')

    LR_graph.set_xlabel("Time (s)")
    LR_graph.set_ylabel("Altitude (ft), Velocity (ft/s)")
    LR_graph.set_title("Blue Raven Flight Data")
    LR_graph.grid(True)
    LR_graph.legend()


    fig2, HR_graph = plt.subplots()
    x_accel, = HR_graph.plot(HR_data['Flight_Time_(s)'], HR_data['Accel_X'], label='X Acceleration')
    y_accel, = HR_graph.plot(HR_data['Flight_Time_(s)'], HR_data['Accel_Y'], label='Y Acceleration')
    z_accel, = HR_graph.plot(HR_data['Flight_Time_(s)'], HR_data['Accel_Z'], label='Z Acceleration')

    HR_graph.set_xlabel("Time [s]")
    HR_graph.set_ylabel("Acceleration [g's]")
    HR_graph.set_title("Blue Raven HR Accelerometer Flight Data")
    HR_graph.grid(True)
    HR_graph.legend()


    fig3, voltage_graph = plt.subplots()
    apo_ch_line, = voltage_graph.plot(LR_data['Flight_Time_(s)'], LR_data['Apo_Volts'], label='Apo Volts')
    main_ch_line, = voltage_graph.plot(LR_data['Flight_Time_(s)'], LR_data['Main_Volts'], label='Main Volts')

    voltage_graph.set_xlabel("Flight Time")
    voltage_graph.set_ylabel("Volts")
    voltage_graph.set_title("Voltage on Deployment Channels")
    voltage_graph.legend()

    # Enable hover on both data series
    cursor = mplcursors.cursor([blr_alt_line, 
                                inert_spd_line, 
                                inert_alt_line, 
                                apo_ch_line, 
                                main_ch_line, 
                                x_accel, 
                                y_accel, 
                                z_accel], 
                                hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.2f}s\nval = {sel.target[1]:.2f}"))

    #plt.show()
    return LR_data, HR_data


if __name__ == "__main__":
    plot_altus("data-2026/FT2/2026-04-19-serial-16162-flight-0004.csv", False)
    #plot_altus("plotting\\data-2026\\FT2\\2026-04-19-serial-16162-flight-0004.csv", False)
    #plot_altus("plotting\\data-2026\\FT1\\2026-03-08-serial-16162-flight-0003-via-12200.csv", False)
    plt.show()



