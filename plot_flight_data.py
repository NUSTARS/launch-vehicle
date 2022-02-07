'''
Program to read CSV Flight Data and generate plots.

First you must download the altimeter flight data and save export 
it as a CSV file using the MDACS software from MissileWorks

NOTE: Apogee is not necesarily the maximum value in the array of altitudes.
There are some weird pressure affects after drogue deploy. Take a loot at the plot,
you can see when it levels out and afterwards there appears to be some further fluctuation/increase
that probably is noise.

'''
import numpy as np 
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog



# Import flight data
root = tk.Tk()
root.withdraw()
dataPath = filedialog.askopenfilename() # Ask user to select data
print('\n\nPath to flight data: %s\n' % dataPath)
root.destroy()

# Read in the data
t =[]       # Time (s)
a = []      # Altitude (ft.)
p = []      # Pressure
v = []      # Velocity (ft/s)
temp = []   # Temperature (F)
volts = []  # Voltage (V)
colHeaders = []
count = 0
with open(dataPath) as d:
    for row in d:
        row = row.split(",")    
        if count > 0:               # Append the data accordingly
            t.append(row[0])
            a.append(row[1])
            p.append(row[2])
            v.append(row[3])
            temp.append(row[4])

        elif count == 0:            # Get the table headers
            c = 0
            for i in row:
                if c == 0:
                    i = i[3:]       # Remove weird characters before 'time'
                elif c == 6:
                    i = i[0:8]      # Remove the '\n' after 'voltage'
                colHeaders.append(i)
                c = c+1
        count += 1

# print('Table column headers: %s\n' % colHeaders)

# Convert data to floats
t = [float(i) for i in t]  
a = [float(i) for i in a]
p = [float(i) for i in p]
v = [float(i) for i in v]
# temp = [float(i) for i in temp]
# volts = [float(i) for i in volts]

# Altitude plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(t,a,'-b')
ax.set(xlabel='time (s)', ylabel='Altitude (ft)',title='Altitude vs Time')
ax.grid()
plt.show()

# Velocity Plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(t,v,'-r')
ax.set(xlabel='time (s)', ylabel='Velocity (ft/s)',title='Velocity vs Time')
ax.grid()
plt.show()








