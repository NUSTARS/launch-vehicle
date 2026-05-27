# quick plotting script
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

def plot_altus(filename, title, metric=False):
    flight_data = pd.read_csv(filename)

    if not metric:
        flight_data['height'] = flight_data['height'] * 3.28084
        flight_data['speed'] = flight_data['speed'] * 3.28084
        flight_data['acceleration'] = flight_data['acceleration'] * 3.28084

    # Flight profile graph
    fig1, alt = plt.subplots()

    # Plot altitude and speed and acc with markers
    alt_line, = alt.plot(flight_data['time'], flight_data['height'], label='Altitude')
    alt.set_xlabel("Time [s]")
    alt.set_ylabel('Altitude [ft]' if not metric else 'Altitude [m]')
    alt.grid(True)

    speed = alt.twinx()
    spd_line, = speed.plot(flight_data['time'], flight_data['speed'], label='Velocity', color='tab:orange')
    speed.set_ylabel('Vertical Speed [ft/s]' if not metric else 'Velocity [m/s]')

    acc = alt.twinx()
    acc.spines["right"].set_position(("outward", 60))   # push spine out
    acc_line, = acc.plot(flight_data['time'], flight_data['acceleration'], label='Acceleration', color='tab:green')
    acc.set_ylabel('Vertical Acceleration [ft/s^2]' if not metric else 'Acceleration [m/s^2]')

    alt.legend(handles=[alt_line, spd_line, acc_line], loc='upper right')
    plt.title(f"Altus: {title}")

    # Enable hover on both data series
    cursor = mplcursors.cursor([alt_line, spd_line, acc_line], hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.2f}s\nval = {sel.target[1]:.2f}"))


    # Voltage graph
    fig2, alt = plt.subplots()
    alt_line, = alt.plot(flight_data['time'], flight_data['height'], label='Altitude')
    alt.set_xlabel("Time [s]")
    alt.set_ylabel('Altitude [ft]' if not metric else 'Altitude [m]')
    alt.grid(True)

    volt = alt.twinx()
    drogue, = volt.plot(flight_data['time'], flight_data['drogue_voltage'], label='Drogue Voltage', color='tab:orange')
    main, = volt.plot(flight_data['time'], flight_data['main_voltage'], label='Main Voltage', color='tab:green')
    volt.set_ylabel('Voltage')
    plt.title(f"Altus: {title}")




    #plt.show()

    return flight_data


############################################################################################################################################


def plot_blueraven(LR_filename, HR_filename, title):
    LR_data = pd.read_csv(LR_filename)
    HR_data = pd.read_csv(HR_filename)

    fig1, alt = plt.subplots()

    # Plot altitude and speed with markers
    blr_alt_line, = alt.plot(LR_data['Flight_Time_(s)'], LR_data['Baro_Altitude_AGL_(feet)'], label='Baro Altitude')
    #inert_alt_line = LR_graph.scatter(LR_data['Flight_Time_(s)'], LR_data['Inertial_Altitude'], label='Inertial Altitude', marker='.')
    alt.set_xlabel("Time (s)")
    alt.set_ylabel("Altitude (ft)")
    alt.grid(True)

    spd = alt.twinx()
    spd_line, = spd.plot(LR_data['Flight_Time_(s)'], LR_data['Velocity_Up'], label='Velocity', color='tab:orange')
    spd.set_ylabel('Vertical Speed (ft/s)')

    acc = alt.twinx()
    acc.spines["right"].set_position(("outward", 60))   # push spine out
    acc_line, = acc.plot(HR_data['Flight_Time_(s)'], np.sqrt(HR_data['Accel_X']**2 + HR_data['Accel_Y']**2 + HR_data["Accel_Z"]**2))
    acc.set_ylabel("Acceleration (ft/s^2)")
    

    plt.title(f"Blue Raven: {title}")
    alt.legend(handles=[blr_alt_line, spd_line])


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

    alt = voltage_graph.twinx()
    alt_line, = alt.plot(LR_data['Flight_Time_(s)'], LR_data['Baro_Altitude_AGL_(feet)'])
    alt.set_ylabel("Altitude [ft]")

    voltage_graph.set_xlabel("Flight Time")
    voltage_graph.set_ylabel("Volts")
    voltage_graph.set_title(f"Raven Voltage: {title}")
    voltage_graph.legend()

    # Enable hover on both data series
    cursor = mplcursors.cursor([blr_alt_line,
                                spd_line,
                                acc_line,  
                                apo_ch_line, 
                                main_ch_line, 
                                x_accel, 
                                y_accel, 
                                z_accel], 
                                hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.2f}s\nval = {sel.target[1]:.2f}"))

    #plt.show()
    return LR_data#, HR_data


if __name__ == "__main__":
    plot_altus('plotting\data-2026\FT3\Rocket_TeleMega.csv', 'FT3 Rocket')
    plot_blueraven('plotting\data-2026\FT3\Rocket_BlRv_LR.csv', 'plotting\data-2026\FT3\Rocket_BlRv_HR.csv', "FT3 Rocket")    
    plot_altus('plotting\data-2026\FT3\Payload_TeleMega.csv', 'FT3 Payload')
    plot_blueraven('plotting\data-2026\FT3\Payload_BlRv_LR.csv', 'plotting\data-2026\FT3\Payload_BlRv_HR.csv', "FT3 Payload")

    
    plt.show()



