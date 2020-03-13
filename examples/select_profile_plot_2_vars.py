"""Pandas dataframe example of plotting 2 variable profiles on the same viewing axis
2020-03-13: added a method to the Dba class that does this."""
from sgdm import Dba
from sgdm.osdba import ls_dbas
import datetime
import logging
import math
import matplotlib.pyplot as plt
from pprint import pprint as pp

log_level = getattr(logging, 'INFO')
log_format = '%(asctime)s:%(module)s:%(levelname)s:%(message)s [line %(lineno)d]'
logging.basicConfig(format=log_format, level=log_level)

dt0 = datetime.datetime(2017, 4, 24, 14, 0, 0)
dt1 = datetime.datetime(2017, 4, 24, 15, 0, 0)

dba_dir = '/Users/kerfoot/datasets/gliders/rucool/deployments/2017/ru28-20170424T1310/data/in/ascii/dbd'
dba_dir = '../data/ru28-20170424T1310'
dbas = ls_dbas(dba_dir, dt0=dt0, dt1=dt1)

process_gps = True
process_ctd = True
index_profiles = True

start_time = datetime.datetime.now()
dba = Dba(dbas, keep_gld_dups=False, gps=process_gps, ctd=process_ctd, profiles=index_profiles)
end_time = datetime.datetime.now()

# Set the row
profile_number = 0
# Pull out the profile
profile = dba.slice_profile_by_id(profile_number)

# Sensors to plot
sensor1 = 'temperature_raw'
sensor2 = 'practical_salinity_raw'

# Get the units
units1 = dba.column_defs[sensor1]['attrs'].get('units', '?')
units2 = dba.column_defs[sensor2]['attrs'].get('units', '?')
zunits = dba.column_defs[dba.depth_sensor]['attrs'].get('units', '?')

# Data frame containing the plot sensors and dba.depth_sensor
data1 = profile[[sensor1, dba.depth_sensor]].dropna(axis='index', how='any')
data2 = profile[[sensor2, dba.depth_sensor]].dropna(axis='index', how='any')

props1 = {'marker': 'o',
          'markerfacecolor': 'None',
          'markeredgecolor': 'b',
          'color': 'b'}
props2 = {'marker': 'o',
          'markerfacecolor': 'None',
          'markeredgecolor': 'r',
          'color': 'r'}

axes1_color = props1['color']
axes2_color = props2['color']

# Create the figure and axis
fig, ax1 = plt.subplots()
# Figure size
fig.set_size_inches(8.5, 11)
# Add a second axis for plotting sensor2
ax2 = plt.twiny(ax1)

# Plot data1 profile
m1 = ax1.plot(data1[sensor1], data1[dba.depth_sensor], **props1)
# Plot data2 profile
m2 = ax2.plot(data2[sensor2], data2[dba.depth_sensor], **props2)
# Share the y-axis between both axes
ax1.get_shared_y_axes().join(ax1, ax2)
# Pretty up the y-limits
ax1.set_ylim([0, math.ceil(ax1.get_ylim()[1])])
# Reverse the y-axis direction
ax1.invert_yaxis()

# color the ticks and tick labels for both axes
ax1.tick_params(axis='x', colors=axes1_color)
ax2.tick_params(axis='x', colors=axes2_color)
# color the x-axis labels
ax1.xaxis.label.set_color(axes1_color)
ax2.xaxis.label.set_color(axes2_color)

# Color the x-axis lines
ax1.spines['bottom'].set_edgecolor(axes1_color)
ax2.spines['top'].set_edgecolor(axes2_color)
# Set ax2 bottom x-axis line to None so that the ax1 color shows
ax2.spines['bottom'].set_edgecolor('None')

# Label the axes
ax1.set_xlabel('{:} ({:})'.format(sensor1, units1))
ax2.set_xlabel('{:} ({:})'.format(sensor2, units2))
ax1.set_ylabel('{:} ({:})'.format(dba.depth_sensor, zunits))

plt.tight_layout()
plt.savefig('/Users/kerfoot/dev/sgdm/profiles.png')
