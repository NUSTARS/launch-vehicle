# quick plotting script
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import numpy as np

def plot_altus(filename, metric=True):
    flight_data = pd.read_csv(filename)

    if not metric:
        flight_data['height'] = flight_data['height'] * 3.28084
        flight_data['speed'] = flight_data['speed'] * 3.28084

    fig, motion_graph = plt.subplots()

    # Plot altitude and speed with markers
    alt_line = motion_graph.scatter(flight_data['time'], flight_data['height'], label='Altitude')
    spd_line = motion_graph.scatter(flight_data['time'], flight_data['speed'], label='Velocity')

    motion_graph.set_xlabel("Time (s)")
    motion_graph.set_ylabel("Altitude (ft), Velocity (ft/s)" if not metric else "Altitude (m), Velocity (m/s)")
    motion_graph.set_title("Altus Altimeter Flight Data")
    motion_graph.grid(True)
    motion_graph.legend()

    # Enable hover on both data series
    cursor = mplcursors.cursor([alt_line, spd_line], hover=True)
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


##########################################################################################33#######################


def plot_loadcell(file_path):
    # read to a dataframe
    cell_data = pd.read_csv(file_path, names=['time', 'force'], header=None)
    cell_data['time'] = cell_data['time']/1000

    # basic metrics
    num_samples = len(cell_data['time'])
    time_rec = cell_data['time'].iloc[-1] - cell_data['time'].iloc[0]
    avg_sample_rate = num_samples / time_rec
    print('Samples:', num_samples)
    print('Time [s]:', time_rec)
    print('Avg. Sample Rate [Hz]:', avg_sample_rate)
    loop_times = cell_data['time'].diff().dropna()

    # plot
    fig1, force_graph = plt.subplots()
    force_series = force_graph.scatter(cell_data['time'], cell_data['force'], marker='.', color='tab:red')
    force_line = force_graph.plot(cell_data['time'], cell_data['force'], color='tab:red')
    force_graph.set_xlabel('time [s]')
    force_graph.set_ylabel('Force [lbf]')
    force_graph.set_title('Force vs. Time')
    force_graph.grid(True)

    fig2, sample_graph = plt.subplots()
    sample_series = sample_graph.scatter(range(len(loop_times)), loop_times)
    sample_graph.set_ylabel('Time [s]')
    sample_graph.set_xlabel('Sample #')
    sample_graph.set_title('Time Between Samples')

    # Enable hover on both data series
    cursor = mplcursors.cursor(force_series, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.2f}s\nforce = {sel.target[1]:.2f}lbf"))

    cursor = mplcursors.cursor(sample_series, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"sample = {sel.target[0]:.2f}\ntime = {sel.target[1]:.4f}s"))

    #plt.show()

    return cell_data


#########################################################################################################################################


def plot_overlay(cell_dat_path, blr_LR_path, blr_HR_path, telemega_path):
    # read to a dataframes
    cell_data = pd.read_csv(cell_dat_path, names=['time', 'force'], header=None)
    cell_data['time'] = cell_data['time']/1000
    blr_data = pd.read_csv(blr_LR_path)
    blr_data_HR = pd.read_csv(blr_HR_path)
    blr_data_HR['Accel_tot'] = np.sqrt((blr_data_HR["Accel_X"])**2 + 
                                        (blr_data_HR["Accel_Y"])**2 + 
                                        (blr_data_HR["Accel_Z"])**2)
    telemega_data = pd.read_csv(telemega_path)
    telemega_data['height_imperial'] = telemega_data['height'] * 3.28084

    # open figure and set up first axes
    fig1, force_ax = plt.subplots()
    force_series = force_ax.scatter(cell_data['time'], 
                                    cell_data['force'], 
                                    marker='.', 
                                    color='red', 
                                    s=2, 
                                    label="Load Cell Force")
    
    force_line, = force_ax.plot(cell_data['time'], 
                                cell_data['force'], 
                                label="Load Cell Force", 
                                color='tab:red')
    force_ax.set_xlabel('Time [s]')
    force_ax.set_ylabel('Force [lbf]')
    force_ax.set_title("Load Cell Data Overlaid with Altimeter")
    force_ax.grid(True)

    # set up second axes
    alt_ax = force_ax.twinx()
    blr_alt_line, = alt_ax.plot(blr_data['Flight_Time_(s)'], 
                                blr_data['Baro_Altitude_AGL_(feet)'], 
                                label='Blue Raven Baro Altitude',
                                color='tab:blue')
        #blr_alt_line = motion_ax.scatter(blr_data['Flight_Time_(s)'], blr_data['Baro_Altitude_AGL_(feet)'], label='Baro Altitude', marker='.', s=2)
    #inert_spd_line, = motion_ax.plot(blr_data['Flight_Time_(s)'], 
    #                                blr_data['Velocity_Up'], 
    #                                label='Velocity',
    #                                color = 'tab:orange')
    #inert_alt_line = motion_ax.scatter(blr_data['Flight_Time_(s)'], blr_data['Inertial_Altitude'], label='Inertial Altitude', marker='.')
    telemega_alt_line, = alt_ax.plot(telemega_data['time'],
                                     telemega_data['height_imperial'],
                                     label='Telemega Altitude',
                                     color='tab:cyan')

    alt_ax.set_ylabel("Altitude [ft]")
    
    # accel_ax = force_ax.twinx()
    # accel_line, = accel_ax.plot(blr_data_HR["Flight_Time_(s)"], 
    #                             blr_data_HR['Accel_tot'],
    #                             label="Inertial Acceleration (total)",
    #                             color="tab:orange")
    # accel_ax.set_ylabel("Acceleration [g's]")
    # accel_ax.spines["right"].set_position(("outward", 60))


    

    # set up hover
    cursor = mplcursors.cursor(force_series, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.2f}s\nforce = {sel.target[1]:.2f}lbf"))
    
    cursor = mplcursors.cursor([blr_alt_line, telemega_alt_line], hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.2f}s\nval = {sel.target[1]:.2f}"))

    lines = [force_line, blr_alt_line, telemega_alt_line]
    labels = [l.get_label() for l in lines]
    force_ax.legend(lines, labels)
    #plt.show()



if __name__ == "__main__":
    plot_overlay('data/flight_1_trimmed.txt', 'Rocket Name LR_08-16-2025_12_28_06.csv', 'Rocket Name HR_08-16-2025_12_28_06.csv', '2025-08-16-serial-16162-flight-0002.csv')
    plot_loadcell('data/flight_1_trimmed.txt')
    #plot_blueraven('Rocket Name LR_08-16-2025_12_28_06.csv', 'Rocket Name HR_08-16-2025_12_28_06.csv')
    #plot_altus('bernie_backup.csv', False)
    #plot_loadcell("data/flight_sim.txt")
    #plot_altus('2025-08-16-serial-16162-flight-0002.csv', False)
    plt.show()



