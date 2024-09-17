import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
import math
from PIL import Image
import matplotlib.ticker as ticker
#import seaborn as sns
# import scipy as spi
# from mpl_toolkits.mplot3d import Axes3D

input_is_metric = True
output_is_metric = False

plt.style.use('seaborn-v0_8-dark-palette')

def clean_data(df, alpha):
    assert alpha > 0 and alpha <= 1, 'Alpha must be between 0 and 1'
    assert 'altitude' in df.columns, 'Altitude data not found in DataFrame'
    assert 'height' in df.columns, 'Height data not found in DataFrame'
    assert 'speed' in df.columns, 'Speed data not found in DataFrame'
    assert 'acceleration' in df.columns, 'Acceleration data not found in DataFrame'

    df['smoothed_altitude'] = df['altitude'].ewm(alpha=alpha, min_periods=1).mean()
    df['smoothed_height'] = df['height'].ewm(alpha=alpha, min_periods=1).mean()
    df['smoothed_velocity'] = df['speed'].ewm(alpha=alpha, min_periods=1).mean()
    df['smoothed_acceleration'] = df['acceleration'].ewm(alpha=alpha, min_periods=1).mean()

    return df

def convert_to_imperial(df):
    if input_is_metric and output_is_metric == False: ### how andy likes to deal with data
        df['smoothed_altitude_ft'] = df['smoothed_altitude'] * 3.28084  # 1 meter = 3.28084 feet
        df['smoothed_height_ft'] = df['smoothed_height'] * 3.28084
        df['smoothed_velocity_ft/s'] = df['smoothed_velocity'] * 3.28084
        df['smoothed_acceleration_ft/s^2'] = df['smoothed_acceleration'] * 3.28084 
    return df

def add_fluid_properties(df, properties):
    try:
        # all of these are in imperial units
        df['density'] = np.vectorize(density_of_air)(df['smoothed_altitude_ft'])
        df['local_speed_of_sound'] = np.vectorize(speed_of_sound)(df['smoothed_altitude_ft'])
        df['kinematic_viscosity'] = np.vectorize(kinematic_viscosity)(df['smoothed_altitude_ft'])
        df['dynamic_viscosity'] = df['density'] * df['kinematic_viscosity']
        df['dynamic_pressure'] = 0.5 * df['density'] * df['smoothed_velocity_ft/s']**2

        # experimental stuff that depends on viscosity
        df['reynolds_number'] = (df['smoothed_velocity_ft/s'] * properties['length_scale']) / df['kinematic_viscosity']
        # df['mach_number'] = df['velocity_ft/s'] / df['speed_of_sound']
        # df['stagnation_pressure'] = 0.5 * df['density'] * df['velocity_ft/s']**2
        # df['stagnation_temperature'] = df['temperature'] + (df['velocity_ft/s']**2) / (2 * 1716)
        # df['stagnation_density'] = df['stagnation_pressure'] / (1716 * df['stagnation_temperature'])
        return df
    except:
        print('Error adding fluid properties')

def smoothing_comparison_plot(df):
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 9), sharex=True)

        main_parachute_index_primary = df.index[df['state_name'].str.strip() == 'main'].tolist()[0]
        time_main_primary = df.loc[main_parachute_index_primary, 'time']
        N_before = 1
        M_after = 3
        start_index_primary = df[df['time'] <= time_main_primary - N_before].index[-1]
        end_index_primary = df[df['time'] >= time_main_primary + M_after].index[0]
        isolated_df = df.loc[start_index_primary:end_index_primary]

        ax1.set_ylabel('Height (ft)', color='blue')
        ax1.tick_params('y', colors='blue')
        ax1.grid(True)
        ax1.set_title('Primary Data')
        
        ax2.set_ylabel('Velocity (ft/s)', color='orange')
        ax2.tick_params('y', colors='orange')
        ax2.grid(True)

        ax3.set_ylabel('Acceleration (ft/s^2)', color='green')
        ax3.tick_params('y', colors='green')
        ax3.grid(True)
        ax3.set_xlabel('Time (s)')
        
        time = isolated_df['time']

        for a in np.arange(0.1, 1, 0.1):
            height = isolated_df['height_ft'].ewm(alpha=a, min_periods=1).mean()
            velocity = isolated_df['velocity_ft/s'].ewm(alpha=a, min_periods=1).mean()
            acceleration = isolated_df['acceleration_ft/s^2'].ewm(alpha=a, min_periods=1).mean()
            
            ax1.plot(time, height, label=f"{a:.1f}")
            ax2.plot(time, velocity, label=f"{a:.1f}")
            ax3.plot(time, acceleration, label=f"{a:.1f}")
        
        ax1.legend()
        ax2.legend()
        ax3.legend()

    except:
      print('Error generating smoothing comparison plot')

def gps_plot(df):
    assert 'latitude' in df.columns, 'Latitude data not found in DataFrame'
    assert 'longitude' in df.columns, 'Longitude data not found in DataFrame'

    try:
        # Calculate displacement from the origin for latitude and longitude
        df['lat_displacement'] = (df['latitude'] - df['latitude'].iloc[0]) * 111139  # 1 degree latitude ~= 111139 meters
        df['long_displacement'] = (df['longitude'] - df['longitude'].iloc[0]) * 111139 * np.cos(np.deg2rad(df['latitude'].iloc[0]))  # 1 degree longitude ~= 111139 meters * cos(latitude)

        # Convert displacement to feet
        df['lat_displacement_ft'] = df['lat_displacement'] * 3.28084
        df['long_displacement_ft'] = df['long_displacement'] * 3.28084

        max_height_index = df['smoothed_height_ft'].idxmax()

        # Plot trajectory in 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=df['long_displacement_ft'],
            y=df['lat_displacement_ft'],
            z=df['smoothed_height_ft'],
            mode='markers',
            marker=dict(
                size=4,
                color=df['smoothed_velocity_ft/s'],  
                colorscale='Viridis',  
                opacity=0.8
            )
        )])

        fig.update_layout(
            title='3D Trajectory of GPS coordinates with Altitude',
            scene=dict(
                xaxis=dict(title='Longitude (feet from origin)'),
                yaxis=dict(title='Latitude (feet from origin)'),
                zaxis=dict(title='Altitude (feet)'),
            )
        )

        fig.add_trace(go.Scatter3d(
            x=[df['long_displacement_ft'][max_height_index]],
            y=[df['lat_displacement_ft'][max_height_index]],
            z=[df['smoothed_height_ft'][max_height_index]],
            mode='markers',
            marker=dict(
                size=8,
                color='red',  
                opacity=1
            )
        ))

        fig.show()
    except:
        print('Error plotting GPS data')

    # fig.add_trace(go.Scatter(
    # x=df['longitude'],
    # y=df['latitude'],
    # mode='markers',
    # marker=dict(
    #     size=9,
    #     color=df['time'],  # Color points based on the time column
    #     colorscale='Viridis',  # Choose a color scale
    #     colorbar=dict(title='Time')  # Add a color bar
    # ),
    # name='GPS Data'
    # ))

    # fig.add_trace(go.Scatter(
    # x=[df['longitude'].iloc[0]],
    # y=[df['latitude'].iloc[0]],
    # mode='markers',
    # marker=dict(size=12, color='blue'),
    # name='Initial Position'
    # ))

    # # Highlight final position
    # fig.add_trace(go.Scatter(
    #     x=[df['longitude'].iloc[-1]],
    #     y=[df['latitude'].iloc[-1]],
    #     mode='markers',
    #     marker=dict(size=12, color='red'),
    #     name='Final Position'
    # ))

    # # Set the layout
    # fig.update_layout(
    #     title="GPS Data Plot",
    #     xaxis=dict(title="Longitude"),
    #     yaxis=dict(title="Latitude"),
    # )

    # # Show the figure
    # fig.show()

def profile_plot(df):
    try:
        time = df['time']
        height = df['smoothed_height_ft']
        velocity = df['smoothed_velocity_ft/s']
        acceleration = df['smoothed_acceleration_ft/s^2']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot height vs time on the left axis
        ax1.plot(time, height, label='Height (ft)', color='blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Height (ft)', color='blue')
        ax1.tick_params('y', colors='blue')
        ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # Adjust the number as needed
        ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # Adjust the number as needed

        # Adjust the grid to include minor ticks
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)  # Adjust linewidth and linestyle as needed

        max_index = height.idxmax()
        max_height = height[max_index]
        max_time = time[max_index]

        # Annotate the maximum point on the plot
        plt.scatter(max_time, max_height, color='red', s=100)
        plt.text(max_time + 0.1, max_height, 'Max Height: {:.2f}'.format(max_height), color='black', fontsize=10)

        # Create a second y-axis for velocity and acceleration on the right
        ax2 = ax1.twinx()

        # Plot velocity vs time on the right axis
        ax2.plot(time, velocity, label='Velocity (ft/s)', color='orange')
        ax2.set_ylabel('Velocity (ft/s)', color='orange')
        ax2.tick_params('y', colors='orange')

        # Plot acceleration vs time on the right axis
        ax2.plot(time, acceleration, label='Acceleration (ft/s^2)', color='green')
        ax2.set_ylabel('Velocity (ft/s) and Acceleration (ft/s^2)', color='orange')

        # Add legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        plt.title('Flight Profile Plot')

        return fig

        #plt.savefig('flight_profile_plot.png', dpi=300)
    except:
        print('Error plotting flight data')

def RRC3_plots(df):
    try:
        time = df['Time']
        height = df['Altitude']
        velocity = df['Velocity']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot height vs time on the left axis
        ax1.plot(time, height, label='Height (ft)', color='blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Height (ft)', color='blue')
        ax1.tick_params('y', colors='blue')
        ax1.grid(True)

        max_index = height.idxmax()
        max_height = height[max_index]
        max_time = time[max_index]

        # Annotate the maximum point on the plot
        plt.scatter(max_time, max_height, color='red', s=100)
        plt.text(max_time + 0.1, max_height, 'Max Height: {:.2f}'.format(max_height), color='black', fontsize=10)

        # Create a second y-axis for velocity and acceleration on the right
        ax2 = ax1.twinx()

        # Plot velocity vs time on the right axis
        ax2.plot(time, velocity, label='Velocity (ft/s)', color='orange')
        ax2.set_ylabel('Velocity (ft/s)', color='orange')
        ax2.tick_params('y', colors='orange')

        # Add legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        plt.title('Flight Profile Plot')
    except:
        print('Error plotting RRC3 Flight Data')

def comparison_plot(df_primary, df_backup):
    try:
        time_primary = df_primary['time']
        height_primary = df_primary['smoothed_height_ft']
        velocity_primary = df_primary['smoothed_velocity_ft/s']
        acceleration_primary = df_primary['smoothed_acceleration_ft/s^2']

        time_backup = df_backup['time']
        height_backup = df_backup['smoothed_height_ft']
        velocity_backup = df_backup['smoothed_velocity_ft/s']
        acceleration_backup = df_backup['smoothed_acceleration_ft/s^2']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot primary data on the top subplot
        ax1.plot(time_primary, height_primary, label='Primary Height (ft)', color='blue')
        ax1.set_ylabel('Height (ft)', color='blue')
        ax1.tick_params('y', colors='blue')
        ax1.grid(True)
        ax1.set_title('Primary Data')

        max_index_primary = height_primary.idxmax()
        max_height_primary = height_primary[max_index_primary]
        max_time_primary = time_primary[max_index_primary]

        # Annotate the maximum point on the primary plot
        ax1.scatter(max_time_primary, max_height_primary, color='red', s=100)
        ax1.text(max_time_primary + 0.1, max_height_primary, 'Max Height: {:.2f}'.format(max_height_primary), color='black', fontsize=10)

        # Plot velocity vs time on the top subplot
        ax1_v = ax1.twinx()
        ax1_v.plot(time_primary, velocity_primary, label='Primary Velocity (ft/s)', color='orange')
        ax1_v.plot(time_primary, acceleration_primary, label='Primary Acceleration (ft/s^2)', color='green')
        ax1_v.set_ylabel('Velocity (ft/s) and Acceleration (ft/s^2)', color='black')

        # Add legend for primary data
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1_v, labels1_v = ax1_v.get_legend_handles_labels()
        ax1_v.legend(lines1 + lines1_v, labels1 + labels1_v, loc='best')

        # Plot backup data on the bottom subplot
        ax2.plot(time_backup, height_backup, label='Backup Height (ft)', color='blue')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Height (ft)', color='blue')
        ax2.tick_params('y', colors='blue')
        ax2.grid(True)
        ax2.set_title('Backup Data')

        max_index_backup = height_backup.idxmax()
        max_height_backup = height_backup[max_index_backup]
        max_time_backup = time_backup[max_index_backup]

        # Annotate the maximum point on the backup plot
        ax2.scatter(max_time_backup, max_height_backup, color='red', s=100)
        ax2.text(max_time_backup + 0.1, max_height_backup, 'Max Height: {:.2f}'.format(max_height_backup), color='black', fontsize=10)

        # Plot velocity vs time on the bottom subplot
        ax2_v = ax2.twinx()
        ax2_v.plot(time_backup, velocity_backup, label='Backup Velocity (ft/s)', color='orange')
        ax2_v.plot(time_backup, acceleration_backup, label='Backup Acceleration (ft/s^2)', color='green')
        ax2_v.set_ylabel('Velocity (ft/s) and Acceleration (ft/s^2)', color='black')

        # Add legend for backup data
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2_v, labels2_v = ax2_v.get_legend_handles_labels()
        ax2_v.legend(lines2 + lines2_v, labels2 + labels2_v, loc='best')

        plt.suptitle('Flight Profile — Primary and Backup Computers', y=0.95)
        #plt.tight_layout()
    except:
        print('Error plotting comparison data')

def drag_coefficient_calculation(df, properties):
    try:
        isolated_df = df[(df['state_name'].str.strip() == 'fast') | (df['state_name'].str.strip() == 'coast')]

        time = isolated_df['time']
        velocity = isolated_df['smoothed_velocity_ft/s']
        acceleration = isolated_df['smoothed_acceleration_ft/s^2']
        height = isolated_df['smoothed_height_ft']

        W = properties['weight']
        mass = W/32.17
        A = properties['area']
        drag_force = abs(mass * acceleration)

        q = isolated_df['dynamic_pressure']
        drag_coefficients = drag_force / (q * A)

        reynolds_number = isolated_df['reynolds_number']

        A_value = 0 # lower bound on Cd
        B_value = 1 # upper bound on Cd, values above this are from low speeds and likely innacurate

        drag_coefficients_clipped = drag_coefficients[(drag_coefficients >= A_value) & (drag_coefficients <= B_value)]
        time_clipped = time[(drag_coefficients >= A_value) & (drag_coefficients <= B_value)]
        height_clipped = height[(drag_coefficients >= A_value) & (drag_coefficients <= B_value)]
        velocity_clipped = velocity[(drag_coefficients >= A_value) & (drag_coefficients <= B_value)]
        reynolds_number_clipped = reynolds_number[(drag_coefficients >= A_value) & (drag_coefficients <= B_value)]

        drag_data = {'time': time_clipped, 'height': height_clipped, 'velocity': velocity_clipped, 'cd': drag_coefficients_clipped}
        drag_data_df = pd.DataFrame(drag_data)    

        fig, ax = plt.subplots(2, 1, figsize=(6, 8))

        # Cd vs V plot
        ax[0].scatter(velocity_clipped, drag_coefficients_clipped)
        ax[0].set_xlabel('Velocity (ft/s)')
        ax[0].set_ylabel('Cd')
        ax[0].set_title('Cd vs. Velocity')

        # Cd vs Re plot
        ax[1].scatter(reynolds_number_clipped, drag_coefficients_clipped)
        ax[1].set_xscale('log')  # Use log scale for better visualization
        ax[1].set_xlabel('Reynolds Number')
        ax[1].set_ylabel('Cd')
        ax[1].set_title('Cd vs. Reynolds number')
        ax[1].grid(True)

        # Add vertical line at speed of sound
        speed_of_sound_at_sea_level = speed_of_sound(0)
        max_velocity = max(velocity_clipped)
        if max_velocity > speed_of_sound_at_sea_level:
            ax[0].axvline(x=speed_of_sound_at_sea_level, color='r', linestyle='--', label='Speed of Sound') 

        fig.suptitle('Coefficient of Drag Analysis')

        plt.tight_layout()
        
        return drag_data_df
    except:
        print('Error calculating drag coefficient')

def kinematic_viscosity(height_ft):
    # Constants for Sutherland's formula
    nu_0 = 1.568e-5  # kinematic viscosity at sea level (ft^2/s)
    T0 = 518.67  # reference temperature at sea level (R)
    S = 110.4  # Sutherland's constant (R)

    # Calculate temperature at the given height (assuming standard lapse rate)
    T = 518.67 - 0.003566 * height_ft  # standard lapse rate: 0.003566 R/ft

    # Calculate kinematic viscosity using Sutherland's formula
    nu = nu_0 * (T / T0)**(3/2) * (T0 + S) / (T + S)

    #     # Constants for Sutherland's formula
    # nu_0 = 1.568e-5  # kinematic viscosity at sea level (m^2/s)
    # T0 = 288.15  # reference temperature at sea level (K)
    # S = 110.4  # Sutherland's constant (K)

    # # Calculate temperature at the given height (assuming standard lapse rate)
    # T = 288.15 - 0.0065 * height_m  # standard lapse rate: 0.0065 K/m

    # # Calculate kinematic viscosity using Sutherland's formula
    # nu = nu_0 * (T / T0)**(3/2) * (T0 + S) / (T + S)

    # if is_imperial:
    #     nu *= 10.7639  # Convert from m^2/s to ft^2/s

    # return nu

    return nu

def density_of_air(height_ft):
    # Constants for ISA model
    R = 1716  # gas constant for dry air (ft*lbf/(lbm*R))
    T0 = 518.67  # temperature at sea level (Rankine)
    P0 = 2116.2  # pressure at sea level (lbf/ft^2)
    L = 0.003566  # temperature lapse rate (Rankine/ft)

    # Compute temperature at the given height
    T = T0 - L * height_ft

    # Compute pressure at the given height using the ISA pressure equation
    P = P0 * (1 - L * height_ft / T0) ** (32.17 / (R * L))

    # Compute density using the ideal gas law
    rho = P / (R * T)

    return rho

def rotation_matrix(roll, pitch, yaw):
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    R_z = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    R_x = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    
    return np.dot(np.dot(R_z, R_y), R_x)

def calculate_rotated_line(roll_angle, pitch_angle, yaw_angle):
    unit_line1 = np.array([[-0.25, 0, 0], [0.25, 0, 0]])  # Red line along X-axis
    unit_line2 = np.array([[0, -0.25, 0], [0, 0.25, 0]])  # Green line along Y-axis
    unit_line3 = np.array([[0, 0, -0.5], [0, 0, 0.5]])  # Blue line along Z-axis
    
    # Apply the rotation matrix to the unit lines
    rotated_line1 = np.dot(unit_line1, rotation_matrix(roll_angle, pitch_angle, yaw_angle))
    rotated_line2 = np.dot(unit_line2, rotation_matrix(roll_angle, pitch_angle, yaw_angle))
    rotated_line3 = np.dot(unit_line3, rotation_matrix(roll_angle, pitch_angle, yaw_angle))

    return(rotated_line1, rotated_line2, rotated_line3)

def update_plot(frame, roll_data, pitch_data, yaw_data, data, ax):
    ax.cla()  # Clear the previous frame
    
    # Get the roll, pitch, and yaw angles for the current frame
    roll_angle = roll_data[frame]
    pitch_angle = pitch_data[frame]
    yaw_angle = yaw_data[frame]
    
    # Calculate the rotated line for the current frame
    rotated_line, rotated_line2, rotated_line3 = calculate_rotated_line(roll_angle, pitch_angle, yaw_angle)
    
    # Plot the rotated unit line
    ax.plot(rotated_line[:, 0], rotated_line[:, 1], rotated_line[:, 2], 'r', label='X-axis')
    ax.plot(rotated_line2[:, 0], rotated_line2[:, 1], rotated_line2[:, 2], 'g', label='Y-axis')
    ax.plot(rotated_line3[:, 0], rotated_line3[:, 1], rotated_line3[:, 2], 'b', label='Z-axis')
    
    # Set axis labels and a title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Unit Line Animation')
    ax.legend()
    
    # Set axis limits for a better view
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Add frame count
    ax.text(-0.9, -0.9, -1.1, f'Frame {frame}/{len(roll_data)-1}', fontsize=12, color='blue')

    # phase = data['state_name'][frame]
    # ax.text(-0.4, -0.9, -1.1, f'Phase: {phase}', fontsize=12, color='blue')

    # altitude = data['height'][frame]
    # ax.text(0.1, -0.9, -1.1, f'Altitude: {altitude}', fontsize=12, color='blue')

    # velocity = data['speed'][frame]
    # ax.text(0.6, -0.9, -1.1, f'Velocity: {velocity}', fontsize=12, color='blue')

def imu_position_calculation(df):
    assert 'gyro_roll' in df.columns, 'Gyro Roll data not found in DataFrame'
    assert 'gyro_pitch' in df.columns, 'Gyro Pitch data not found in DataFrame'
    assert 'gyro_yaw' in df.columns, 'Gyro Yaw data not found in DataFrame'
    
    try:
        isolated_df = df[(df['state_name'].str.strip() == 'boost') | (df['state_name'].str.strip() == 'fast') | (df['state_name'].str.strip() == 'coast')]

        threshold = 0.1

        isolated_df = isolated_df[isolated_df['gyro_roll'].abs() >= threshold]
        isolated_df = isolated_df[isolated_df['gyro_pitch'].abs() >= threshold]
        isolated_df = isolated_df[isolated_df['gyro_yaw'].abs() >= threshold]

        time = isolated_df['time']  # Calculate time difference between consecutive rows
        dt = isolated_df['time'].diff().fillna(0)  # Calculate time difference between consecutive rows

        # Smoothing factor
        alpha = 0.6
        
        # Gyro Roll Rates
        roll_rate = (isolated_df['gyro_roll'] * dt).ewm(alpha=alpha, min_periods=1).mean()
        pitch_rate = (isolated_df['gyro_pitch'] * dt).ewm(alpha=alpha, min_periods=1).mean()
        yaw_rate = (isolated_df['gyro_yaw'] * dt).ewm(alpha=alpha, min_periods=1).mean()

        roll = np.cumsum(roll_rate.values * dt.values)
        pitch = np.cumsum(pitch_rate.values * dt.values)
        yaw = np.cumsum(yaw_rate.values * dt.values)

        plt.figure()
        plt.plot(time, roll, label='Roll')
        plt.plot(time, pitch, label='Pitch')
        plt.plot(time, yaw, label='Yaw')
        plt.xlabel('Time')
        plt.ylabel('Euler Angles')
        plt.title('Euler Angles vs. Time')
        plt.legend()

        #fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})

        # frames = []
        # for i in range(len(roll)-1):
        #     print("On frame", i, "of", len(roll)-1)
        #     ax.cla()  # Clear the previous frame
        #     update_plot(i, roll, pitch, yaw, isolated_df, ax)  # Update the plot for the current frame
        #     fig.canvas.draw()  # Redraw the canvas
        #     frame = np.array(fig.canvas.renderer.buffer_rgba())  # Capture the frame as an array
        #     frames.append(Image.fromarray(frame))  # Append the frame to the list of frames
        # file_path_gif = '/Users/andrewwehmeyer/Library/Mobile Documents/com~apple~CloudDocs/NUSTARS/Coding/Visualization/test_animation_1234.gif'
        # frames[0].save(file_path_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)

        # file_path = '/Users/andrewwehmeyer/Library/Mobile Documents/com~apple~CloudDocs/NUSTARS/Coding/Visualization/test_animation_1234.mp4'
        #ani.save(file_path, writer='pillow')

        #Create an animation
        #ani = FuncAnimation(fig, update_plot, frames=len(roll)-1, fargs=(roll, pitch, yaw, isolated_df, ax), repeat=False, interval=1)
        #plt.show()
        
        # # Acceleration
        # accel_x = isolated_df['accel_y'].rolling(window=N, min_periods=1).mean() * 3.28084
        # accel_y = isolated_df['accel_z'].rolling(window=N, min_periods=1).mean() * 3.28084
        # accel_z = isolated_df['accel_x'].rolling(window=N, min_periods=1).mean() * 3.28084

        # velocity_x = spi.integrate.cumulative_trapezoid(accel_x, dt, initial=0)
        # velocity_y = spi.integrate.cumulative_trapezoid(accel_y, dt, initial=0)
        # velocity_z = spi.integrate.cumulative_trapezoid(accel_z, dt, initial=0)

        # velocities = np.vstack((velocity_x, velocity_y, velocity_z))

        # # Rotate velocity vectors
        # rotated_velocities = rotate_vectors(velocities, roll, pitch, yaw)

        # # Integrate rotated velocities to get incremental position changes
        # rotated_position_changes = spi.integrate.cumulative_trapezoid(rotated_velocities.T, dt, initial=0).T
    except:
        print('Error calculating IMU position')

def tilt_plot(df):
    assert 'tilt' in df.columns, 'Tilt data not found in DataFrame'

    try:
        isolated_df = df[(df['state_name'].str.strip() == 'boost') | (df['state_name'].str.strip() == 'fast') | (df['state_name'].str.strip() == 'coast')]
        tilt = isolated_df['tilt'].rolling(window=10, min_periods=1).mean()
        height = isolated_df['smoothed_height_ft']
        
        plt.figure()
        plt.plot(tilt, height, color='purple')
        plt.xlabel("Tilt (º)")
        plt.ylabel("Height (ft)")
        plt.title("Tilt vs. Height")
    except:
        print('Error generating tilt plot')
    
def voltage_plot(df):
    assert 'drogue_voltage' in df.columns, 'Drogue voltage data not found in DataFrame'
    assert 'main_voltage' in df.columns, 'Main voltage data not found in DataFrame'

    try:
        time = df['time']
        height = df['smoothed_height_ft']
        drogue_voltage = df['drogue_voltage']
        main_voltage = df['main_voltage']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot height vs. time on the first subplot (top)
        ax1.plot(time, height, 'b-')
        ax1.set_ylabel('Height (ft)')
        ax1.grid(True)

        # Plot both voltages vs. time on the second subplot (bottom)
        ax2.plot(time, drogue_voltage, 'g-', label='Drogue Voltage')
        ax2.plot(time, main_voltage, 'r-', label='Main Voltage')
        ax2.set_ylabel('Voltage')
        ax2.set_xlabel('Time (s)')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        # Highlight abrupt changes in voltage data with vertical lines
        threshold = 0.04  # You can adjust this threshold as needed

        drogue_diff = np.abs(np.diff(drogue_voltage))
        main_diff = np.abs(np.diff(main_voltage))

        drogue_index = np.argmax(drogue_diff > threshold) + 1  # Adding 1 to align with original indices
        main_index = np.argmax(main_diff > threshold) + 1  # Adding 1 to align with original indices

        # Plot vertical lines and annotate on both subplots
        if drogue_index > 0:
            ax1.axvline(x=time[drogue_index], color='gray', linestyle='--', linewidth=1)
            ax2.axvline(x=time[drogue_index], color='gray', linestyle='--', linewidth=1)
            ax1.text(time[drogue_index], ax1.get_ylim()[1], f'{time[drogue_index]:.1f}', color='gray', ha='center', va='bottom')

        if main_index > 0:
            ax1.axvline(x=time[main_index], color='gray', linestyle='--', linewidth=1)
            ax2.axvline(x=time[main_index], color='gray', linestyle='--', linewidth=1)
            ax1.text(time[main_index], ax1.get_ylim()[1], f'{time[main_index]:.1f}', color='gray', ha='center', va='bottom')

        fig.suptitle('Height and Voltage Plot')
    
    except:
       print('Error generating voltage plot')

def sat_count_plot(df):
    assert 'nsat' in df.columns, 'Satellite count data not found in DataFrame'
    
    try:
        isolated_df = df[(df['state_name'].str.strip() == 'boost') | (df['state_name'].str.strip() == 'fast') | (df['state_name'].str.strip() == 'coast')]
        time = isolated_df['time']
        velocity = isolated_df['smoothed_velocity_ft/s']
        nsat = isolated_df['nsat']
        fig, ax1 = plt.subplots()

        # Plot height on the left y-axis
        ax1.plot(time, velocity, '-', color='orange')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (ft/s)', color='orange')

        # Create a secondary y-axis for sat count
        ax2 = ax1.twinx()
        ax2.plot(time, nsat, 'g-')
        ax2.set_ylabel('Satellites in Solution', color='g')

        plt.grid(True)
        plt.title('Satellite Count vs. Time')
    except:
        print('Error generating satellite plot')

def mach_plot(df):
    try:
        isolated_df = df[(df['state_name'].str.strip() == 'boost') | (df['state_name'].str.strip() == 'fast') | (df['state_name'].str.strip() == 'coast')]
        time = isolated_df['time']
        velocity = isolated_df['smoothed_velocity_ft/s']
        sea_level_mach = velocity / np.vectorize(speed_of_sound)(0)
        local_mach = velocity / isolated_df['local_speed_of_sound']

        peak_index = local_mach.argmax()
        peak_time = time[peak_index]
        peak_mach = local_mach[peak_index]

        # Define the time range for the zoomed-in plot
        zoom_start = max(0, peak_index - 50)  # Adjust as needed to include enough data points before the peak
        zoom_end = min(len(time), peak_index + 50)  # Adjust as needed to include enough data points after the peak

        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot sea level Mach number
        ax.plot(time, sea_level_mach, marker='o', color='blue', linestyle='None', label='Sea Level Mach #', markersize=2)
        ax.plot(time, local_mach, marker='o', color='orange', linestyle='--', label='Local Mach #', markersize=2)
        ax.set_ylabel('Mach #')
        ax.set_xlabel('Time (s)')

        ax.scatter(peak_time, peak_mach, color='red', s=100)
        ax.text(peak_time + 0.1, peak_mach, 'Mach {:.2f}'.format(peak_mach), color='black', fontsize=10)

        ax.legend()

        # Add a zoomed-in plot for the peak Mach number
        ins1 = ax.inset_axes([0.35, 0.6, 0.4, 0.3])
        ins1.plot(time[zoom_start:zoom_end], sea_level_mach[zoom_start:zoom_end], marker='o', color='blue', linestyle='-', markersize=2)
        ins1.plot(time[zoom_start:zoom_end], local_mach[zoom_start:zoom_end], marker='o', color='orange', linestyle='-', markersize=2)
        #ins1.set_xlabel('Time (s)')
        #ins1.set_ylabel('Mach #')
        ins1.set_title('Zoomed-In')
        ins1.set_xlim(time[zoom_start], time[zoom_end])
        ins1.set_ylim(local_mach[zoom_start], local_mach[zoom_end]*1.1)

        ax.axhline(y=1, color='red', linestyle='--')

        # Identify where the Mach number crosses M=1
        mach_crossing = np.diff((local_mach >= 1).astype(int))

        # Find the indices where the crossing occurs
        cross_indices = np.where(mach_crossing != 0)[0]

        # Plot dashed grey bars for Mach crossing M=1
        for index in cross_indices:
            ax.axvline(x=time[index], color='gray', linestyle='--')
            ax.text(time[index], ax.get_ylim()[1], f'{time[index]:.1f}', color='gray', ha='center', va='bottom')

        fig.suptitle('Mach # vs. Time')
    except:
        print('Error generating Mach plot')

def speed_of_sound(altitude_ft):
    # Constants for ISA model
    gamma = 1.4  # specific heat ratio for dry air
    R = 1716  # specific gas constant for dry air (ft*lbf/(lbm*R))
    T0 = 518.67  # temperature at sea level (Rankine)
    L = 0.003566  # temperature lapse rate (Rankine/ft)

    # Compute temperature at the given height
    T = T0 - L * altitude_ft

    # Compute speed of sound using the formula
    c = (gamma * R * T) ** 0.5

    return c

def data_between_gaps(df, key='velocity_ft/s', threshold=0.2):

    constant_indicies = np.where(np.diff(df[key]) < threshold)[0]
    constant_indicies = np.concatenate((constant_indicies, [len(df) - 1]))
    
    gaps = np.diff(constant_indicies)
    threshold = 10  # Adjust as needed based on your data
    gap_indices = np.where(gaps > threshold)[0]
    gap_start_indices = constant_indicies[gap_indices]
    gap_end_indices = constant_indicies[gap_indices + 1]

    averages = []

    start_index = 0
    end_index = gap_start_indices[0] - 1
    average_velocity = df[key].iloc[start_index:end_index + 1].mean()
    averages.append(average_velocity)

    for i in range(len(gap_start_indices) - 1):
        start_index = gap_end_indices[i] + 1
        end_index = gap_start_indices[i + 1] - 1
        average_velocity = df[key].iloc[start_index:end_index + 1].mean()
        averages.append(average_velocity)

    # Calculate the average velocity for the region after the second gap
    start_index = gap_end_indices[-1] + 1
    end_index = len(df) - 1
    average_velocity = df[key].iloc[start_index:end_index + 1].mean()
    averages.append(average_velocity)

    return averages

def report(df):
    try:
        max_height = df['smoothed_height_ft'].max()
        max_height_time = df.loc[df['smoothed_height_ft'].idxmax(), 'time']

        max_velocity = df['smoothed_velocity_ft/s'].max()
        max_velocity_time = df.loc[df['smoothed_velocity_ft/s'].idxmax(), 'time']

        max_acceleration = df['smoothed_acceleration_ft/s^2'].max()
        max_acceleration_time = df.loc[df['smoothed_acceleration_ft/s^2'].idxmax(), 'time']

        # Find time of apogee based on state_name column
        apogee_time = df[df['state_name'].str.strip() == 'drogue']['time'].iloc[0]
        apogee_height = df[df['state_name'].str.strip() == 'drogue']['smoothed_height_ft'].iloc[0]

        # Find time of touchdown based on state_name column, only when state is "main"
        touchdown_df = df[(df['smoothed_height_ft'] <= 0) & (df['state_name'].str.strip() == 'main')]
        touchdown_time = touchdown_df['time'].iloc[0]

        drogue_df = df[df['state_name'].str.strip() == 'drogue']
        drogue_df_trimmed = drogue_df.iloc[50:-50]
        drogue_descent_velocity = drogue_df_trimmed['smoothed_velocity_ft/s'].mean()
        # drogue_descent_velocity = data_between_gaps(drogue_df, key='smoothed_velocity_ft/s', threshold=0.1)[1]

        main_df = df[df['state_name'].str.strip() == 'main']
        main_df_trimmed = main_df.iloc[50:-250]
        main_descent_velocity = main_df_trimmed['smoothed_velocity_ft/s'].mean()
        #main_descent_velocity = data_between_gaps(main_df, key='smoothed_velocity_ft/s', threshold=0.1)[1]

        # if debug:
        #     #trying to find gaps to identify apogee-touchdown times
        #     data = data_between_gaps(df, key='smoothed_velocity_ft/s', threshold=1)
        #     plt.figure()
        #     plt.scatter(range(len(data)), data)
        #     plt.show()

        time_between_apogee_touchdown = touchdown_time - apogee_time

        # initial_lat = df['latitude'].iloc[0]
        # initial_long = df['longitude'].iloc[0]
        # final_lat = df['latitude'].iloc[-1]
        # final_long = df['longitude'].iloc[-1]

        # R = 20925721.785  # Earth radius in ft
        # lat_distance = np.radians(final_lat - initial_lat)
        # long_distance = np.radians(final_long - initial_long)
        # a = np.sin(lat_distance / 2) * np.sin(lat_distance / 2) + np.cos(np.radians(initial_lat)) * np.cos(np.radians(final_lat)) * np.sin(long_distance / 2) * np.sin(long_distance / 2)
        # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        # drift_distance = R * c

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.1, 0.9, f'Max Height: {max_height:.2f} ft at t = {max_height_time:.2f} s', fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.8, f'Max Velocity: {max_velocity:.2f} ft/s at t = {max_velocity_time:.2f} s', fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.7, f'Max Acceleration: {max_acceleration:.2f} ft/s^2 at t = {max_acceleration_time:.2f} s', fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.6, f'Drogue Descent Velocity: {drogue_descent_velocity:.2f} ft/s', fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.5, f'Main Descent Velocity: {main_descent_velocity:.2f} ft/s', fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.3, f'Apogee at t = {apogee_time:.2f} s', fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.2, f'Touchdown at t = {touchdown_time:.2f} s', fontsize=12, transform=ax.transAxes)
        ax.text(0.1, 0.1, f'Time Between Apogee and Touchdown: {time_between_apogee_touchdown:.2f} s', fontsize=12, transform=ax.transAxes)
        # ax.text(0.1, 0, f'Drift Distance: {drift_distance:.2f} ft', fontsize=12, transform=ax.transAxes)

        ax.axis('off')
        plt.grid(False)
    except:
        print("Error generating report")

def main_parachute_plot(df_primary,df_backup,properties):
    assert 'state_name' in df_primary.columns, 'State Name column not found in DataFrame'

    try:
        main_parachute_index_primary = df_primary.index[df_primary['state_name'].str.strip() == 'main'].tolist()[0]
        main_parachute_index_backup = df_backup.index[df_backup['state_name'].str.strip() == 'main'].tolist()[0]

        time_main_primary = df_primary.loc[main_parachute_index_primary, 'time']
        time_main_backup = df_backup.loc[main_parachute_index_backup, 'time']

        # Calculate the time difference in seconds
        N_before = 1
        M_after = 8
        start_index_primary = df_primary[df_primary['time'] <= time_main_primary - N_before].index[-1]
        end_index_primary = df_primary[df_primary['time'] >= time_main_primary + M_after].index[0]
        start_index_backup = df_backup[df_backup['time'] <= time_main_backup - N_before].index[-1]
        end_index_backup = df_backup[df_backup['time'] >= time_main_backup + M_after].index[0]

        # Extract the rows within the specified time range for primary DataFrame
        isolated_df_primary = df_primary.loc[start_index_primary:end_index_primary]

        # Extract the rows within the specified time range for backup DataFrame
        isolated_df_backup = df_backup.loc[start_index_backup:end_index_backup]

        time_primary = isolated_df_primary['time']
        height_primary = isolated_df_primary['smoothed_height_ft']
        velocity_primary = isolated_df_primary['smoothed_velocity_ft/s']
        acceleration_primary = isolated_df_primary['smoothed_acceleration_ft/s^2']

        time_backup = isolated_df_backup['time']
        height_backup = isolated_df_backup['smoothed_height_ft']
        velocity_backup = isolated_df_backup['smoothed_velocity_ft/s']
        acceleration_backup = isolated_df_backup['smoothed_acceleration_ft/s^2']

        fig, ((ax1_primary, ax1_backup), (ax2_primary, ax2_backup), (ax3_primary, ax3_backup)) = plt.subplots(3, 2, figsize=(20, 9), sharex=True)

        # Plot height vs time on the first subfigure for primary and backup
        min_height = min(min(height_primary), min(height_backup))
        max_height = max(max(height_primary), max(height_backup))
        ax1_primary.set_ylim(min_height, max_height)
        ax1_backup.set_ylim(min_height, max_height)

        ax1_primary.plot(time_primary, height_primary, label='Height (ft)', color='blue')
        ax1_backup.plot(time_backup, height_backup, label='Height (ft)', color='blue')
        ax1_primary.set_ylabel('Height (ft)', color='blue')
        ax1_primary.tick_params('y', colors='blue')
        ax1_primary.grid(True)
        ax1_primary.legend()
        ax1_backup.set_ylabel('Height (ft)', color='blue')
        ax1_backup.tick_params('y', colors='blue')
        ax1_backup.grid(True)
        ax1_backup.legend()
        ax1_primary.set_title('Primary Data')
        ax1_backup.set_title('Backup Data')

        # Plot velocity vs time on the second subfigure for primary and backup
        ax2_primary.plot(time_primary, velocity_primary, label='Velocity (ft/s)', color='orange')
        ax2_backup.plot(time_backup, velocity_backup, label='Velocity (ft/s)', color='orange')
        ax2_primary.set_ylabel('Velocity (ft/s)', color='orange')
        ax2_primary.tick_params('y', colors='orange')
        ax2_primary.grid(True)
        ax2_primary.legend()
        ax2_backup.set_ylabel('Velocity (ft/s)', color='orange')
        ax2_backup.tick_params('y', colors='orange')
        ax2_backup.grid(True)
        ax2_backup.legend()

        # Plot acceleration vs time on the third subfigure for primary and backup
        ax3_primary.plot(time_primary, acceleration_primary, label='Acceleration (ft/s^2)', color='green')
        ax3_backup.plot(time_backup, acceleration_backup, label='Acceleration (ft/s^2)', color='green')
        ax3_primary.set_ylabel('Acceleration (ft/s^2)', color='green')
        ax3_primary.tick_params('y', colors='green')
        ax3_primary.grid(True)
        ax3_primary.legend()
        ax3_backup.set_ylabel('Acceleration (ft/s^2)', color='green')
        ax3_backup.tick_params('y', colors='green')
        ax3_backup.grid(True)
        ax3_backup.legend()
        ax3_primary.set_xlabel('Time (s)')
        ax3_backup.set_xlabel('Time (s)')

        H_primary = properties.get('primary_main_parachute_height')
        height_threshold_index_primary = isolated_df_primary.index[height_primary < H_primary].tolist()[0]
        threshold_crossing_time_primary = time_primary[height_threshold_index_primary]
        ax1_primary.axvline(x=threshold_crossing_time_primary, color='red', linestyle='--', linewidth=1)
        ax1_primary.axhline(y=H_primary, color='red', linestyle='--', linewidth=1)
        ax1_primary.text(threshold_crossing_time_primary, ax1_primary.get_ylim()[1], f'{threshold_crossing_time_primary:.1f}', color='red', ha='center', va='bottom')
        ax2_primary.axvline(x=threshold_crossing_time_primary, color='red', linestyle='--', linewidth=1)
        ax3_primary.axvline(x=threshold_crossing_time_primary, color='red', linestyle='--', linewidth=1)

        H_backup = properties.get('backup_main_parachute_height')
        height_threshold_index_backup = isolated_df_backup.index[height_backup < H_backup].tolist()[0]
        threshold_crossing_time_backup = time_backup[height_threshold_index_backup]
        ax1_backup.axvline(x=threshold_crossing_time_backup, color='red', linestyle='--', linewidth=1)
        ax1_backup.axhline(y=H_backup, color='red', linestyle='--', linewidth=1)
        ax1_backup.text(threshold_crossing_time_backup, ax1_backup.get_ylim()[1], f'{threshold_crossing_time_backup:.1f}', color='red', ha='center', va='bottom')
        ax2_backup.axvline(x=threshold_crossing_time_backup, color='red', linestyle='--', linewidth=1)
        ax3_backup.axvline(x=threshold_crossing_time_backup, color='red', linestyle='--', linewidth=1)

        main_voltage_primary = df_primary['main_voltage']
        main_voltage_threshold = 0.3  # You can adjust this threshold as needed
        main_diff_primary = np.abs(np.diff(main_voltage_primary))
        main_voltage_index_primary = np.argmax(main_diff_primary > main_voltage_threshold) + 1
        if main_voltage_index_primary > 0:
            ax1_primary.axvline(x=time_primary[main_voltage_index_primary], color='purple', linestyle='--', linewidth=1)
            ax1_primary.text(time_primary[main_voltage_index_primary], ax1_primary.get_ylim()[0], f'{time_primary[main_voltage_index_primary]:.1f}', color='purple', ha='center', va='bottom')
            ax2_primary.axvline(x=time_primary[main_voltage_index_primary], color='purple', linestyle='--', linewidth=1)
            ax3_primary.axvline(x=time_primary[main_voltage_index_primary], color='purple', linestyle='--', linewidth=1)

        main_voltage_backup = df_backup['main_voltage']
        main_diff_backup = np.abs(np.diff(main_voltage_backup))
        main_voltage_index_backup = np.argmax(main_diff_backup > main_voltage_threshold) + 1
        if main_voltage_index_backup > 0:
            ax1_backup.axvline(x=time_backup[main_voltage_index_backup], color='purple', linestyle='--', linewidth=1)
            ax1_backup.text(time_backup[main_voltage_index_backup], ax1_backup.get_ylim()[0], f'{time_backup[main_voltage_index_backup]:.1f}', color='purple', ha='center', va='bottom')
            ax2_backup.axvline(x=time_backup[main_voltage_index_backup], color='purple', linestyle='--', linewidth=1)
            ax3_backup.axvline(x=time_backup[main_voltage_index_backup], color='purple', linestyle='--', linewidth=1)

        acceleration_spike_index_primary = np.argmax(acceleration_primary)
        acceleration_spike_time_primary = time_primary.iloc[acceleration_spike_index_primary]
        ax1_primary.axvline(x=acceleration_spike_time_primary, color='green', linestyle='--', linewidth=1)
        ax2_primary.axvline(x=acceleration_spike_time_primary, color='green', linestyle='--', linewidth=1)
        ax3_primary.axvline(x=acceleration_spike_time_primary, color='green', linestyle='--', linewidth=1)
        ax1_primary.text(acceleration_spike_time_primary, ax1_primary.get_ylim()[1], f'{acceleration_spike_time_primary:.1f}', color='green', ha='center', va='bottom')

        acceleration_spike_index_backup = np.argmax(acceleration_backup)
        acceleration_spike_time_backup = time_backup.iloc[acceleration_spike_index_backup]
        ax1_backup.axvline(x=acceleration_spike_time_backup, color='green', linestyle='--', linewidth=1)
        ax2_backup.axvline(x=acceleration_spike_time_backup, color='green', linestyle='--', linewidth=1)
        ax3_backup.axvline(x=acceleration_spike_time_backup, color='green', linestyle='--', linewidth=1)
        ax1_backup.text(acceleration_spike_time_backup, ax1_backup.get_ylim()[1], f'{acceleration_spike_time_backup:.1f}', color='green', ha='center', va='bottom')

        # Define the text for the callouts
        callout_text_primary = f"Main Crossing: {threshold_crossing_time_primary:.1f}\nVoltage Drop: {time_primary[main_voltage_index_primary]:.1f}\nAccel. Spike: {acceleration_spike_time_primary:.1f}"
        callout_text_backup = f"Main Crossing: {threshold_crossing_time_backup:.1f}\nVoltage Drop: {time_backup[main_voltage_index_backup]:.1f}\nAccel. Spike: {acceleration_spike_time_backup:.1f}"

        # Add a text block for primary data
        ax1_primary.text(0.65, 0.5, callout_text_primary, transform=ax1_primary.transAxes, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

        # Add a text block for backup data
        ax1_backup.text(0.65, 0.5, callout_text_backup, transform=ax1_backup.transAxes, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
        
    except:
        print("Error generating main parachute plot")

def main_parachute_plot_backup_only(df_backup, properties):
    assert 'state_name' in df_backup.columns, 'State Name column not found in DataFrame'

    try:
        main_parachute_index_backup = df_backup.index[df_backup['state_name'].str.strip() == 'main'].tolist()[0]

        time_main_backup = df_backup.loc[main_parachute_index_backup, 'time']

        # Calculate the time difference in seconds
        N_before = 1
        M_after = 5
        start_index_backup = df_backup[df_backup['time'] <= time_main_backup - N_before].index[-1]
        end_index_backup = df_backup[df_backup['time'] >= time_main_backup + M_after].index[0]

        # Extract the rows within the specified time range for backup DataFrame
        isolated_df_backup = df_backup.loc[start_index_backup:end_index_backup]

        time_backup = isolated_df_backup['time']
        height_backup = isolated_df_backup['smoothed_height_ft']
        velocity_backup = isolated_df_backup['smoothed_velocity_ft/s']
        acceleration_backup = isolated_df_backup['smoothed_acceleration_ft/s^2']

        fig, ((ax1_backup), (ax2_backup), (ax3_backup)) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Plot height vs time on the first subfigure for backup
        ax1_backup.plot(time_backup, height_backup, label='Height (ft)', color='blue')
        ax1_backup.set_ylabel('Height (ft)', color='blue')
        ax1_backup.tick_params('y', colors='blue')
        ax1_backup.grid(True)
        ax1_backup.legend()
        ax1_backup.set_title('Backup Data')

        # Plot velocity vs time on the second subfigure for backup
        ax2_backup.plot(time_backup, velocity_backup, label='Velocity (ft/s)', color='orange')
        ax2_backup.set_ylabel('Velocity (ft/s)', color='orange')
        ax2_backup.tick_params('y', colors='orange')
        ax2_backup.grid(True)
        ax2_backup.legend()

        # Plot acceleration vs time on the third subfigure for backup
        ax3_backup.plot(time_backup, acceleration_backup, label='Acceleration (ft/s^2)', color='green')
        ax3_backup.set_ylabel('Acceleration (ft/s^2)', color='green')
        ax3_backup.tick_params('y', colors='green')
        ax3_backup.grid(True)
        ax3_backup.legend()
        ax3_backup.set_xlabel('Time (s)')


        H_backup = properties.get('backup_main_parachute_height')
        height_threshold_index_backup = isolated_df_backup.index[height_backup < H_backup].tolist()[0]
        threshold_crossing_time_backup = time_backup[height_threshold_index_backup]
        ax1_backup.axvline(x=threshold_crossing_time_backup, color='red', linestyle='--', linewidth=1)
        ax1_backup.axhline(y=H_backup, color='red', linestyle='--', linewidth=1)
        ax1_backup.text(time_backup.iloc[-1], H_backup, f'{H_backup:.1f} ft', color='red', ha='left', va='bottom')
        ax2_backup.axvline(x=threshold_crossing_time_backup, color='red', linestyle='--', linewidth=1)
        ax3_backup.axvline(x=threshold_crossing_time_backup, color='red', linestyle='--', linewidth=1)

        main_voltage_backup = df_backup['main_voltage']
        main_voltage_threshold = 0.2  # You can adjust this threshold as needed
        main_diff_backup = np.abs(np.diff(main_voltage_backup))
        main_voltage_index_backup = np.argmax(main_diff_backup > main_voltage_threshold) + 1
        if main_voltage_index_backup > 0:
            ax1_backup.axvline(x=time_backup[main_voltage_index_backup], color='purple', linestyle='--', linewidth=1)
            ax2_backup.axvline(x=time_backup[main_voltage_index_backup], color='purple', linestyle='--', linewidth=1)
            ax3_backup.axvline(x=time_backup[main_voltage_index_backup], color='purple', linestyle='--', linewidth=1)

            voltage_drop_height = height_backup[main_voltage_index_backup]
            ax1_backup.axhline(y=voltage_drop_height, color='purple', linestyle='--', linewidth=1)
            ax1_backup.text(time_backup.iloc[-1], voltage_drop_height, f'{voltage_drop_height:.1f} ft', color='purple', ha='left', va='bottom')

        acceleration_spike_index_backup = np.argmax(acceleration_backup)
        acceleration_spike_time_backup = time_backup.iloc[acceleration_spike_index_backup]
        ax1_backup.axvline(x=acceleration_spike_time_backup, color='green', linestyle='--', linewidth=1)
        ax2_backup.axvline(x=acceleration_spike_time_backup, color='green', linestyle='--', linewidth=1)
        ax3_backup.axvline(x=acceleration_spike_time_backup, color='green', linestyle='--', linewidth=1)
        acceleration_spike_height_backup = height_backup.iloc[acceleration_spike_index_backup]

        # Plot a horizontal line for the acceleration spike height on ax1_backup
        ax1_backup.axhline(y=acceleration_spike_height_backup, color='green', linestyle='--', linewidth=1)

        # Add a text callout for the acceleration spike height
        ax1_backup.text(time_backup.iloc[-1], acceleration_spike_height_backup, f'{acceleration_spike_height_backup:.1f} ft', color='green', ha='left', va='bottom')


        ax1_backup.text(time_backup.iloc[-1], acceleration_spike_time_backup, f'{acceleration_spike_time_backup:.1f} s', color='green', ha='left', va='bottom')

        # Define the text for the callouts
        callout_text_backup = f"Main Crossing:\n  Time: {threshold_crossing_time_backup:.1f} s\n  Height: {H_backup:.1f} ft\nVoltage Drop:\n  Time: {time_backup[main_voltage_index_backup]:.1f} s\n  Height: {voltage_drop_height:.1f} ft\nAccel. Spike:\n  Time: {acceleration_spike_time_backup:.1f} s\n  Height: {acceleration_spike_height_backup:.1f} ft"

        # Add a text block for backup data in the middle plot (velocity vs. time)
        ax2_backup.text(0.75, 0.5, callout_text_backup, transform=ax2_backup.transAxes, verticalalignment='center', horizontalalignment='left',bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.9'))

    except Exception as e:
        print(f"Error generating backup parachute plot: {e}")

def dynamic_pressure_plot(df):
    try:
        isolated_df = df[(df['state_name'].str.strip() == 'boost') | (df['state_name'].str.strip() == 'fast') | (df['state_name'].str.strip() == 'coast')]
        time = isolated_df['time']
        q = isolated_df['dynamic_pressure']

        # Plotting
        plt.figure()
        plt.plot(time, q)
        plt.xlabel('Time')
        plt.ylabel('Dynamic Pressure (lbf/ft^2)')
        plt.title('Dynamic Pressure vs. Time')

    except:
        print("Error generating dynamic pressure plot")

def generic_plot(xaxis_key, yaxis_key, df, title, xlabel, ylabel, color='b'):
    try:
        plt.figure()
        plt.plot(df[xaxis_key], df[yaxis_key], color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    except:
        print("Error generating generic plot")

def main(file_name, properties):
    # DATA PREPPING
    path_start = "/Users/andrewwehmeyer/Library/Mobile Documents/com~apple~CloudDocs/NUSTARS/Coding/Visualization/data/"
    df_primary = pd.read_csv(path_start + file_name + "_primary.csv")
    df_backup = pd.read_csv(path_start + file_name + "_backup.csv")

    # CLEANING DATA AND CONVERTING TO IMPERIAL
    alpha = 0.6
    df_primary = convert_to_imperial(clean_data(df_primary,alpha))
    df_backup = convert_to_imperial(clean_data(df_backup,alpha))

    # ADDING FLUID PROPERTIES
    df_primary = add_fluid_properties(df_primary, properties)
    df_backup = add_fluid_properties(df_backup, properties)

    # FOR TESTING
    specific_run = False
    if specific_run:

        print("Specific Run")

        profile_plot(df_backup)
        main_parachute_plot_backup_only(df_backup, properties)
        report(df_backup)
        #voltage_plot(df_backup)
        #drag_coefficient_calculation(df_backup, properties)
        #eport(df_backup)

        #(df_primary)
        #voltage_plot(df_primary)
        #print("HI ANDY")
    
    # GENERAL PLOTS
    default_run = True
    if default_run:
        gps_plot(df_primary)
        comparison_plot(df_primary, df_backup)
        profile_plot(df_primary)
        voltage_plot(df_primary)
        sat_count_plot(df_primary)
        mach_plot(df_primary)
        drag_coefficient_calculation(df_primary, properties)
        tilt_plot(df_primary)
        main_parachute_plot(df_primary, df_backup, properties)
        report(df_primary)

    # EXPERIMENTAL PLOTS
    experimental_run = False
    if experimental_run:
        imu_position_calculation(df_primary)
        RRC3_plots(df_primary)
        dynamic_pressure_plot(df_primary)
        generic_plot('Time', 'Altitude', df_primary, 'Height vs. Time', 'Time (s)', 'Height (ft)')

    # DEBUGGING AND TESTING FEATURES
    debug = False
    if debug:
        smoothing_comparison_plot(df_primary)

    save_figures = False
    if save_figures:
        for i in plt.get_fignums():
            plt.figure(i).savefig(f'/Users/andrewwehmeyer/Library/Mobile Documents/com~apple~CloudDocs/NUSTARS/Coding/Visualization/Figures/figure_{i}.png', dpi=300)

    plt.show()

file_name = "supersonic"
properties = {"weight": 36.77, "area": 0.22, "length_scale":6.17, "primary_main_parachute_height":600, "backup_main_parachute_height":550} # FT7
#properties = {"weight": 4.85, "area": 0.03587714, "length_scale":2.242,"main_parachute_height":500}

main(file_name, properties)

# notes for future
# add ways to detect failed import and improve error checking
# add way to convert legacy rrc3 files into usable data -> differentiate to find acceleration
# add some way to switch to metric bc reasons
# probably default everything to metric and add if statement to convert to imperial at END of calculations when displaying stuff
# idk how to deal with properties but 
# add way to determine or set input units