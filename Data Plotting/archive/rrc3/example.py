from rrc3py import RRC3
from matplotlib import pyplot as plt
import numpy as np

# import the csv data to a RRC3 object
# the RRC3 object "primary" has the following fields
# time, altitdue, pressure, velocity, temperature, events, voltages
primary = RRC3("primary.csv")


# Make an altitude vs time plot
plt.style.use("ggplot")
fig, ax = plt.subplots()
ax.plot(primary.time, primary.altitude, '-b')
ax.set(xlabel="Time (s)", ylabel="Altitude (ft)")
ax.set(title="Altitude vs. Time")
ax.set_xticks(np.arange(0,105,5))
ax.set_yticks(np.arange(0,4200,200))
plt.show()

