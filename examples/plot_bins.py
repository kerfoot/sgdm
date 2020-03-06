"""Example of glidertools.plot.bin_size plotting function using the Dba class.
Depth bins are calculated based on dba.depth_sensor

IMPORTANT: make sure you have created an instance of the Dba class first!"""
import glidertools as gt
import matplotlib as mpl

# Load the dbas before proceeding!

# Grab the depths and remove all NaNs
depths = dba.data[dba.depth_sensor].dropna()
# Find the max delta depth to scale the x-axis
delta_z_max = depths.diff().max()

# Plot the bin histogram (from GliderTools)
ax = gt.plot.bin_size(depths, cmap=mpl.cm.Blues)

# Format the bin size line
line = ax.get_children()[1]
line.set_linewidth(2)
line.set_color('orange')

# Legend viz
legend = ax.get_children()[-2]
legend.set_visible(False)

# Set the xlimits from 0 to delta_z_max + 0.5 meters
ax.set_xlim([0, delta_z_max+0.5])

# Show the plot
mpl.pyplot.show()
