#!/usr/bin/env python3 
'''
Program to read CSV Flight Data and generate plots.

First you must download the altimeter flight data and save export 
it as alt CSV file using the MDACS software from MissileWorks

NOTE: Apogee is not necesarily the maximum value in the array of altitudes.
There are some weird pressure affects after drogue deploy. Take a loot at the plot,
you can see when it levels out and afterwards there appears to be some further fluctuation/increase
that probably is noise.

'''
import numpy as np 
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog

plt.style.use("ggplot") # Set matplotlib style


def mav(x,y,num_avg): 
    '''
    mav(x,y,num_avg) computes the moving average of the specified number of points
    of the inputted y and returns it as the new x values and new y values as two lists.
    '''
    ynew = []
    i = 0
    while i < (len(y) - num_avg + 1):
        reg = y[i:i+num_avg]
        mav = sum(reg)/num_avg
        ynew.append(mav)
        i += 1

    lost = (x[-1] - x[-num_avg])
    xnew = np.linspace(0,x[-1] - lost,len(ynew))

    return xnew,ynew


# Import flight data
root = tk.Tk()
root.withdraw()
dataPath = filedialog.askopenfilename() # Ask user to select data
print('\n\nPath to flight data: %s\n' % dataPath)
root.destroy()

# Read in the data
t =[]       # Time (s)
alt = []    # Altitude (ft.)
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
            alt.append(row[1])
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
alt = [float(i) for i in alt]
p = [float(i) for i in p]
v = [float(i) for i in v]
# temp = [float(i) for i in temp]
# volts = [float(i) for i in volts]

# Apply moving average filters to the data
talt, alt = mav(t,alt,20)
tv, v = mav(t,v,20)
tp, p = mav(t,p,20)

# Altitude plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(talt,alt,'-b')
ax.set(xlabel='time (s)', ylabel='Altitude (ft)',title='Altitude vs Time')
plt.show()

# Velocity plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(tv,v,'-r')
ax.set(xlabel='time (s)', ylabel='Velocity (ft/s)',title='Velocity vs Time')
plt.show()

