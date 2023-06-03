from pandas import read_csv

class RRC3:
    def __init__(self,csv_path: str):
        all_data_pd = read_csv(csv_path) 
        self.time = all_data_pd['Time'].to_numpy()
        self.altitude = all_data_pd['Altitude'].to_numpy()
        self.pressure = all_data_pd['Pressure'].to_numpy()
        self.velocity = all_data_pd['Velocity'].to_numpy()
        self.temperature = all_data_pd['Temperature'].to_numpy()
        self.events = all_data_pd['Events'].to_numpy()
        self.voltages = all_data_pd['Voltages'].to_numpy()
