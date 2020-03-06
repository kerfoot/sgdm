"""Example of glidertools.plot.bin_size plotting function using the Dba class.
Depth bins are calculated based on dba.depth_sensor

IMPORTANT: make sure you have created an instance of the Dba class first!"""
import glidertools as gt
import matplotlib as mpl

# Load the dbas

depths = dba.data[dba.depth_sensor].dropna()
ax = gt.plot.bin_size(depths, cmap=mpl.cm.Blues)

delta_z_max = depths.diff().max()

line = ax.get_children()[1]
line.set_linewidth(2)
line.set_color('orange')

legend = ax.get_children()[-2]
legend.set_visible(False)

ax.set_xlim([0, delta_z_max+0.5])

mpl.pyplot.show()
